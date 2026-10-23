"""Read-only CPU candidate for a yaw-rotated reference and canonical observations.

The physical world stays unchanged. Commands, action/rhythm/wait semantics and
frozen policy weights stay unchanged. A physical runner is not admitted here.
"""
import numpy as np
from continuous_scoop import AttachedResidualEnv,write
from manual_residual_sac_env import LIMITS_SI,OBS_SIZE

VERSION='yaw_reference_and_observation_v1_cpu'
ANCHOR=np.array([5.85,-.05,.65])
# Native free-joint qpos has 19 entries and qvel has 18 (quaternion versus
# angular velocity). Therefore bucket xyz starts at19+18+19=56, not57.
BUCKET_START=56
WORLD_VECTOR_STARTS=(59,62,65,68)

def rotation(angle):
 if not np.isfinite(angle) or abs(angle)>np.pi:raise ValueError('invalid_reference_yaw')
 c,s=np.cos(angle),np.sin(angle)
 return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

def canonical_values(values,angle,R):
 """Transform the UNCLIPPED vector; clip exactly once after transforming."""
 values=np.asarray(values,dtype=float)
 if values.shape!=(OBS_SIZE,) or not np.isfinite(values).all():raise ValueError('yaw_observation_contract')
 if angle==0:return values
 result=values.copy();result[7]-=angle
 result[BUCKET_START:BUCKET_START+3]=R.T@(result[BUCKET_START:BUCKET_START+3]+ANCHOR)-ANCHOR
 for first in WORLD_VECTOR_STARTS:result[first:first+3]=R.T@result[first:first+3]
 # Joint velocities and joint/reference differences are scalars in their
 # joint coordinates. Four-axis requests and wait errors keep their SI units.
 return result

def height_grid(cloud,bucket,R,angle):
 relative=cloud[:,:2]-bucket[:2]
 if angle!=0:relative=relative@R[:2,:2]
 indices=np.floor((relative+.8)/.2).astype(int)
 inside=((indices>=0)&(indices<8)).all(1)
 indices=indices[inside];heights=(cloud[inside,2]-bucket[2])/3
 height=np.full(64,-1.,dtype=np.float32);known=np.zeros(64,dtype=np.float32)
 flat=indices[:,1]*8+indices[:,0]
 np.maximum.at(height,flat,heights);known[np.unique(flat)]=1
 return height,known

class YawTransferEnv(AttachedResidualEnv):
 """Observation/reference adapter only; .reset and physical .step reject."""
 def __init__(self,*args,yaw_offset_rad,**kwargs):
  self.yaw_offset_rad=float(yaw_offset_rad);self.canonical_rotation=rotation(self.yaw_offset_rad)
  super().__init__(*args,**kwargs)
  if self.reference_q.shape!=(1171,19) or self.reference_targets.shape!=(1171,4):
   raise ValueError('yaw_reference_contract')
  self.reference_q=self.reference_q.copy();self.reference_targets=self.reference_targets.copy()
  if self.yaw_offset_rad!=0:
   self.reference_q[:,7]+=self.yaw_offset_rad;self.reference_targets[:,0]+=self.yaw_offset_rad
 def attach(self,*args,**kwargs):
  obs=super().attach(*args,**kwargs)
  self.report.update(reference_transfer_version=VERSION,yaw_offset_rad=self.yaw_offset_rad,
     policy_observation_frame='canonical_training_yaw',physical_logging_frame='original_world',
     physical_admission=False,reference_geometry='heading rotation only, original cylinders/commands/timebase')
  write(self.directory/'report.json',self.report)
  return obs
 def step(self,*args,**kwargs):
  raise RuntimeError('yaw_transfer_cpu_only_not_admitted_for_physics')
 def observation(self):
  # Field order follows the SHA-bound original 221D encoder. Zero-offset
  # equality and virtual rotation equivalence are checked against that encoder.
  state=self.runtime.bridge.state;q=np.asarray(state['mechanism_qpos'])
  phase=min(int(self.cursor),1170);reference=self.reference_q[phase]
  bucket=np.asarray(state['bucket_origin']);force=self.runtime.captured.get('wrenches',{}).get('bucket')
  wrench=np.zeros(6) if force is None else np.r_[force['force'],force['torque']]/1e5
  points=[self.runtime.fixture.state.particle_q.numpy()]
  from mixed_material_r0 import quaternion_matrix_wxyz
  for vertices,rock in zip(self.vertices,state['rocks']):
   pose=np.asarray(rock['qpos']);points.append(vertices@quaternion_matrix_wxyz(pose[3:]).T+pose[:3])
  cloud=np.concatenate(points)
  height,known=height_grid(cloud,bucket,self.canonical_rotation,self.yaw_offset_rad)
  base=np.zeros(4) if self.hold is not None else self.commands[min(phase,1169)]/LIMITS_SI
  values=np.r_[q,np.asarray(state['mechanism_qvel']),q-reference,
   bucket-ANCHOR,state['bucket_linear_velocity_world_m_s'],state['bucket_angular_velocity_world_rad_s'],
   wrench,base,self.previous_applied/LIMITS_SI,self.previous_action,
   self.cursor/1170,self.runtime.frames/1560,float(force is not None),height,known,
   np.asarray(self.report['last_alignment']['wait_target_error_si'])/LIMITS_SI,
   self.alignment.measured/1170,float(self.report['last_alignment']['mode'].startswith('wait'))]
  values=canonical_values(values,self.yaw_offset_rad,self.canonical_rotation)
  return np.clip(values,-10,10).astype(np.float32)
