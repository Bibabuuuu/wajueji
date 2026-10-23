"""CPU oracle/metamorphic checks; real cached states and synthetic rotations.

Synthetic rotated clouds are unit-test inputs, never saved as physical states.
No native context, physical step, optimizer or training experience is created.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,hashlib,signal,copy
from types import SimpleNamespace as S
import numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sys.path[:0]=[str(W),str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import CpuRuntime,MODEL,MODEL_SHA,sha,read,write
from manual_residual_sac_env import ManualResidualEnv,LIMITS_SI,control_mode,command_at
from yaw_transfer import YawTransferEnv,rotation,canonical_values,VERSION
import run_layered_47_sparse_manual_contact_episode as manual

class Array:
 def __init__(self,data):self.data=data
 def numpy(self):return self.data

def fixture(state,cloud,vertices,reference,*,angle,mode,force,commands):
 cursor=1170. if mode=='hold' else 100.25
 actual=94.25 if mode=='wait' else cursor
 return S(runtime=S(bridge=S(state=state),captured={} if force is None else {'wrenches':{'bucket':force}},
    fixture=S(state=S(particle_q=Array(cloud))),frames=1200 if mode=='hold' else 120),
   cursor=cursor,reference_q=reference,vertices=vertices,commands=commands,hold=0 if mode=='hold' else None,
   previous_applied=np.array([.01,-.002,.003,.004]),previous_action=np.array([.2,-.5,.3,-.1,.8]),
   report={'last_alignment':dict(wait_target_error_si=[.01,-.002,.003,-.005],mode=mode,reference_limit=actual+6)},
   alignment=S(measured=actual),yaw_offset_rad=angle,canonical_rotation=rotation(angle))

def main():
 out=W/'adapter-cpu-v2';out.mkdir(exist_ok=False);start=time.monotonic();checks={};decisions=0
 report=dict(status='checking',physics_steps_added=0,native_initializations=0,training_updates=0,physical_admission=False,version=VERSION)
 def check(k,v):checks[k]=bool(v);assert v,k
 def timeout(*_):raise TimeoutError('adapter CPU 180 second ceiling')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
 try:
  angle=read(W/'rotation-probe-v2/report.json')['yaw_offset_rad']
  old=ManualResidualEnv(out/'legacy',S(),collect=False,waiting_action_mode='error_strength_v1')
  zero=YawTransferEnv(out/'zero',S(),yaw_offset_rad=0.,collect=False,waiting_action_mode='error_strength_v1')
  new=YawTransferEnv(out/'rotated',S(),yaw_offset_rad=angle,collect=False,waiting_action_mode='error_strength_v1')
  check('zero reference bytes unchanged',zero.reference_q.tobytes()==old.reference_q.tobytes() and zero.reference_targets.tobytes()==old.reference_targets.tobytes())
  check('original velocity commands bytes unchanged',new.commands.tobytes()==old.commands.tobytes())
  check('three cylinder references bytes unchanged',new.reference_q[:,[9,12,15]].tobytes()==old.reference_q[:,[9,12,15]].tobytes())
  check('only reference swing and target swing offset',np.array_equal(new.reference_q[:,np.arange(19)!=7],old.reference_q[:,np.arange(19)!=7]) and np.max(abs(new.reference_q[:,7]-old.reference_q[:,7]-angle))<1e-15 and np.array_equal(new.reference_targets[:,1:],old.reference_targets[:,1:]))
  r=CpuRuntime(ROOT/'runs/continuous-single-scoop-20261014-v1/final.npz',out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  vertices=manual._cache_rock_vertices(r.root,r.records)
  cache=ROOT/'runs/continuous-single-scoop-20261014-v1/display'
  before={n:(cache/n).stat().st_mtime_ns for n in ['states.json','particles.npy']}
  states=read(cache/'states.json');selected=[(i,states[i]) for i in [5626,5926,6526,6832]];del states
  particles=np.load(cache/'particles.npy',mmap_mode='r')
  source_arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in r.arrays.items()}
  for i,state in selected:
   state_hash=json.dumps(state,sort_keys=True);cloud=particles[i];cloud_hash=hashlib.sha256(cloud.tobytes()).hexdigest()
   for mode in ['follow','wait','hold']:
    runtime_state=dict(state,bucket_origin=state['bucket_origin_world_m'])
    obj=fixture(runtime_state,cloud,vertices,old.reference_q,angle=0.,mode=mode,force=None,commands=old.commands)
    a=ManualResidualEnv.observation(obj);b=YawTransferEnv.observation(obj)
    check(f'real cache zero-angle221 exact frame{i} {mode}',a.tobytes()==b.tobytes())
   check('actual cached state unchanged '+str(i),state_hash==json.dumps(state,sort_keys=True) and cloud_hash==hashlib.sha256(cloud.tobytes()).hexdigest())
  check('physical arrays unchanged',source_arrays=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in r.arrays.items()})
  check('display files unchanged',before=={n:(cache/n).stat().st_mtime_ns for n in before})
  import torch
  from stable_baselines3 import SAC
  torch.set_num_threads(1);model=SAC.load(MODEL,device='cpu');updates=model._n_updates
  check('frozen221/5 network loaded unchanged',sha(MODEL)==MODEL_SHA and model.observation_space.shape==(221,) and model.action_space.shape==(5,))
  rng=np.random.default_rng(183);max_obs=0.;max_action=0.;saved=[]
  # Deliberately asymmetric clouds, + and - yaw, hold/wait/follow and values
  # outside clipping bounds. This catches direction, row/column and clip-order bugs.
  for angle2 in [angle,-angle,np.deg2rad(37),np.deg2rad(-37)]:
   R=rotation(angle2)
   for mode in ['follow','wait','hold']:
    for has_force in [False,True]:
     bucket=np.array([18.,-17.,7.]) if has_force else np.array([6.,-.1,1.])
     q=old.reference_q[100].copy();qvel=rng.normal(0,.01,18)
     state=dict(mechanism_qpos=q.tolist(),mechanism_qvel=qvel.tolist(),bucket_origin=bucket.tolist(),
       bucket_linear_velocity_world_m_s=[12.,-15.,.7],bucket_angular_velocity_world_rad_s=[-13.,17.,-.2],
       rocks=[{'qpos':[*(bucket+np.array([.11,.34,.2])),1.,0.,0.,0.]}])
     cloud=bucket+rng.uniform([-.799,-.799,-.5],[.799,.799,.9],size=(137,3))
     local=[np.array([[0.,0.,0.],[.03,.01,.05],[-.04,-.02,.01]])]
     force={'force':[1.6e6,2.7e6,-3e6],'torque':[-4.1e6,5.3e6,7e6]} if has_force else None
     base=fixture(state,cloud,local,old.reference_q,angle=0.,mode=mode,force=force,commands=old.commands)
     expected=ManualResidualEnv.observation(base)
     moved=copy.deepcopy(state);moved['mechanism_qpos'][7]+=angle2
     for key in ['bucket_origin','bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s']:moved[key]=(R@np.array(state[key])).tolist()
     moved['rocks'][0]['qpos']=[*(R@np.array(state['rocks'][0]['qpos'][:3])),np.cos(angle2/2),0.,0.,np.sin(angle2/2)]
     moved_force=None if force is None else {k:(R@np.array(v)).tolist() for k,v in force.items()}
     ref=old.reference_q.copy();ref[:,7]+=angle2
     candidate=fixture(moved,cloud@R.T,local,ref,angle=angle2,mode=mode,force=moved_force,commands=old.commands)
     actual=YawTransferEnv.observation(candidate);err=float(np.max(abs(expected-actual)));max_obs=max(max_obs,err)
     check(f'rotation221 oracle {angle2:.5f} {mode} force{has_force}',err<=2e-6)
     check(f'mode semantics invariant {angle2:.5f} {mode} force{has_force}',control_mode(base.cursor,base.report['last_alignment'],base.hold)==control_mode(candidate.cursor,candidate.report['last_alignment'],candidate.hold))
     a=model.predict(expected,deterministic=True)[0];b=model.predict(actual,deterministic=True)[0];decisions+=2
     err_a=float(np.max(abs(a-b)));max_action=max(max_action,err_a)
     check(f'frozen action equivariance {angle2:.5f} {mode} force{has_force}',err_a<=2e-6)
     saved.append(dict(angle=angle2,mode=mode,has_force=has_force,maximum_observation_error=err,maximum_action_error=err_a))
  for env in [zero,new]:
   try:env.step(np.zeros(5))
   except RuntimeError as e:check('physical step rejected '+str(env.yaw_offset_rad),'cpu_only' in str(e))
   else:raise AssertionError('physical step admitted')
  for a in [float('nan'),float('inf'),4.]:
   try:rotation(a)
   except ValueError:pass
   else:raise AssertionError('invalid yaw accepted')
  check('invalid yaw rejected',True)
  for pace in [-1.,0.,1.]:
   a=np.array([.1,-.2,.3,-.4,pace])
   for error in [None,np.array([.001,-.002,.003,-.004])]:
    kw=dict(reference_limit=106.,wait_target_error=error,waiting_action_mode='error_strength_v1')
    aa=command_at(old.commands,100.,a,**kw);bb=command_at(new.commands,100.,a,**kw)
    check(f'unchanged command pace{pace} wait{error is not None}',all(np.array_equal(x,y) for x,y in zip(aa,bb)))
  check('no optimizer update or model change',model._n_updates==updates and sha(MODEL)==MODEL_SHA)
  write(out/'rotation-cases.json',saved)
  report.update(status='cpu_adapter_passed',checks=checks,check_count=len(checks),real_cached_states=4,zero_angle_real_observations=12,
   synthetic_rotation_cases=len(saved),maximum_observation_error=max_obs,maximum_frozen_action_error=max_action,
   cpu_frozen_predict_calls=decisions,neural_updates=0,model_sha256=MODEL_SHA,original_model_updates=updates,
   code_sha256={n:sha(W/n) for n in ['yaw_transfer.py','check_adapter_cpu.py']},
   unchanged_legacy_encoder_sha256=sha(ROOT/'src/v4/manual_residual_sac_env.py'),
   limits='Synthetic rotation equality is an interface property; no current second-scoop physical policy evaluation or real terrain generalization is established.')
 except BaseException as e:
  import traceback
  report.update(status='cpu_adapter_failed',failure=repr(e),traceback=traceback.format_exc(),checks=checks,cpu_frozen_predict_calls=decisions)
  raise
 finally:signal.alarm(0);report['wall_seconds']=time.monotonic()-start;write(out/'report.json',report);print(report['status'],report.get('check_count'),report['wall_seconds'],flush=True)
if __name__=='__main__':main()
