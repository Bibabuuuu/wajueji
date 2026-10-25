"""Disposable CPU feasibility probe: rotate the saved manual reference to new heading.

No policy wrapper or physical runner is changed/admitted by this probe.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,signal,numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/edge-first-strips-20261015-v1'
sys.path.insert(0,str(ROOT/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import CpuRuntime,FixedRouteClearance,project,segment,read,write,sha
from manual_residual_sac_env import ManualResidualEnv
from manual_trajectory_alignment import ProgressAlignment
from planning.rear_lower_prepoint import sampled_surface_clearance,ground_completed_terrain,mechanism_and_truck_clearance
from types import SimpleNamespace

def main():
 out=W/'reference-edge-v1';out.mkdir(exist_ok=False);start=time.monotonic();report=dict(status='checking',physics_steps_added=0,policy_inference_calls=0,training_updates=0,execution_approved=False)
 def timeout(*_):raise TimeoutError('rotation CPU 180 second ceiling')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
 try:
  source=ROOT/'runs/continuous-single-scoop-20261014-v1/safe_start.npz';source_sha=sha(source)
  r=CpuRuntime(source,out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  env=ManualResidualEnv(out/'reference',SimpleNamespace(),collect=False,waiting_action_mode='error_strength_v1')
  with np.load(W/'cpu-edge-v2/edge-first-approach.npz') as f:current=f['qpos'][-1].copy()
  delta=float(current[7]-env.reference_q[0,7]);c,s=np.cos(delta),np.sin(delta);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  reference=env.reference_q.copy();reference[:,7]+=delta
  targets=env.reference_targets.copy();targets[:,0]+=delta
  poses=np.array([r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in reference])
  original=np.array([r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in env.reference_q])
  xyz=np.einsum('ij,nj->ni',rot,original[:,:3,3]);ori=np.einsum('ij,njk->nik',rot,original[:,:3,:3])
  equivariant_error=max(float(np.max(abs(xyz-poses[:,:3,3]))),float(np.max(abs(ori-poses[:,:3,:3]))))
  goal=project(r,reference[0]);static=FixedRouteClearance(r)
  with np.load(W/'cpu-edge-v2/terrain.npz') as f:t={k:f[k] for k in f.files}
  t=ground_completed_terrain(t)
  with np.load(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as f:truck=f['world_vertices'].copy()
  class Checker:
   def check(self,q,previous=None):
    d=static.check(q,previous);_,b=r.assembly.assemble(q,include_meshes=False);p=b['bucket']
    gap=sampled_surface_clearance(static.samples@p[:3,:3].T+p[:3,3],t)['clearance_m']
    other=mechanism_and_truck_clearance(static.geometries,b,t,truck)
    # The unchanged fixed-route checker above uses the existing CAD SAT
    # refinement and 4/2 cm truck gates. The whole-truck AABB is diagnostic.
    if gap<.32 or other['mechanism_terrain_clearance_m']<.02:raise ValueError('reference_entry_approach_clearance:'+str((gap,other)))
    return dict(d,bucket_observed_terrain_m=gap,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'],
                diagnostic_old_truck_aabb_gap_m=other['static_truck_aabb_separation_m'])
  path=segment(r,current,goal,Checker())
  np.savez(out/'reference-candidate.npz',reference_q=reference,reference_targets=targets,commands=env.commands,
           entry_qpos=path['qpos'],entry_velocity=path['velocity'],yaw_rotation=rot)
  alignment=ProgressAlignment(reference,poses).observe(goal,r.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.)
  static_minima={};worst_closure=0.;hard_limits=True;minimum_ground=float('inf')
  vertices=np.concatenate([g['body_local_vertices'] for g in static.geometries if g['body']=='bucket'])
  for q,pose in zip(reference,poses):
   d=static.check(q,physical=True)
   for k,v in d.items():
    if v is not None:static_minima[k]=min(static_minima.get(k,float('inf')),v)
   worst_closure=max(worst_closure,float(np.max(abs(r.chain.residual(q)))))
   hard_limits &= all(lo-1e-9<=q[i]<=hi+1e-9 for i,(lo,hi) in r.chain.limits.items())
   minimum_ground=min(minimum_ground,float((vertices@pose[:3,:3].T+pose[:3,3])[:,2].min()))
  contact=rot@np.array([5.85,-.05,.65]);chosen=read(W/'cpu-edge-v2/report.json')['selected']['contact']
  selected=np.r_[chosen['contact_xy_m'],chosen['contact_surface_z_m']]
  checks=dict(new_reference_entry_trusted=alignment['trusted'],
    original_three_cylinder_reference_unchanged=np.array_equal(reference[:,[9,12,15]],env.reference_q[:,[9,12,15]]),
    original_velocity_commands_unchanged=True,hard_joint_limits=hard_limits,physical_closure=worst_closure<=.001,
    bucket_above_original_5mm_ground_gate=minimum_ground>=.005,source_unchanged=sha(source)==source_sha)
  report.update(status='cpu_rotation_geometry_passed' if all(checks.values()) else 'cpu_rotation_geometry_failed',checks=checks,
   yaw_offset_rad=delta,yaw_offset_deg=float(np.rad2deg(delta)),entry_alignment=alignment,entry_alignment_frames=path['steps'],
   entry_minimum_clearance=path['minimum_clearance'],reference_frames=len(reference),maximum_rigid_rotation_error=equivariant_error,
   rotated_reference_contact_m=contact,selected_observed_contact_m=selected,contact_offset_m=contact-selected,
   contact_offset_norm_m=float(np.linalg.norm(contact-selected)),minimum_reference_bucket_ground_m=minimum_ground,
   maximum_reference_closure_m=worst_closure,minimum_reference_static_clearance=static_minima,
   source_sha256=source_sha,artifact_sha256=sha(out/'reference-candidate.npz'),
   required_before_physics='Reuse approved yaw221 adapter and actual-joint wait mapping only under separately authorized live scope. This CPU probe admits no physics or policy generalization.',
   boundary='Reference is rotated about base only. No radial/height warp, action or rhythm change. Current terrain is not rotated or modified.')
  assert all(checks.values()),checks
 except BaseException as e:report.update(status='cpu_rotation_failed',failure=repr(e));raise
 finally:signal.alarm(0);report['wall_seconds']=time.monotonic()-start;write(out/'report.json',report);print(report['status'],report['wall_seconds'],flush=True)
if __name__=='__main__':main()
