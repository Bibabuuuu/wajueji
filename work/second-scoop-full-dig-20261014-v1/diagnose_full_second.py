"""CPU-only stall diagnosis and existing wait-action counterfactuals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';sys.path.insert(0,str(W))
import run_full_second as m
from manual_residual_sac_env import command_at,LIMITS_SI
import numpy as np
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]

def main():
 D=m.OUT;p=m.read(D/'report.json');actions=rows(D/'policy/episode-0001/actions.ndjson');decisions=rows(D/'policy-decisions.ndjson');states=m.read(D/'display/states.json')
 r=m.CpuRuntime(D/'zero.npz',W/('diagnosis-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 env=m.YawTransferEnv(W/'diagnosis-reference',None,collect=False,waiting_action_mode='error_strength_v1',yaw_offset_rad=m.CANDIDATE['yaw_offset_rad'])
 last=actions[-1];al=last['alignment'];cursor=last['cursor_before'];q=np.asarray(states[-1]['mechanism_qpos']);target=np.asarray(states[-1]['controller_target']);phase=int(cursor);window=actions[-120:]
 virtual=[]
 for strength in [0.,.5,1.]:
  action=np.r_[np.full(4,2*strength-1),0.]
  base,delta,request,pace=command_at(env.commands,cursor,action,.1,reference_limit=max(cursor,al['reference_limit']),wait_target_error=al['wait_target_error_si'],waiting_action_mode='error_strength_v1')
  virtual.append(dict(strength=strength,request_si=np.asarray(request),pace=pace))
 def bucket(q):return r.assembly.assemble(q,include_meshes=False)[1]['bucket'][:3,3]
 positions=np.asarray([bucket(x) for x in env.reference_q[60:101]])
 result=dict(status='cpu_stall_diagnosed',run_status=p['status'],failure=p.get('failure'),frames=p['frames'],reference_cursor=p.get('reference_cursor'),
  final_actual_progress=al['actual_progress'],last_mode=al['mode'],last_wait_target_error_si=al['wait_target_error_si'],
  last_actual_q_minus_reference_q_si=q[r.chain.axes]-env.reference_q[phase,r.chain.axes],
  last_native_target_minus_actual_q_si=target-q[r.chain.axes],reference_target_minus_reference_q_si=env.reference_targets[phase]-env.reference_q[phase,r.chain.axes],
  last_native_target_minus_reference_target_si=target-env.reference_targets[phase],
  last120_request_max_abs_si=np.max(abs(np.array([x['request_si'] for x in window])),axis=0),last120_applied_max_abs_si=np.max(abs(np.array([x['applied_si'] for x in window])),axis=0),
  last120_policy_strength_mean=np.mean((np.asarray([x['policy_action'][:4] for x in window])+1)/2,axis=0),
  last120_effective_strength_mean=np.mean((np.asarray([x['effective_action'][:4] for x in window])+1)/2,axis=0),
  last120_invalidated_count=sum(x['action_invalidated'] for x in window),
  waiting_frames=sum(x['pace']==0 and x['hold_frame'] is None for x in actions),first_wait_frame=next((x['frame'] for x in actions if x['pace']==0),None),
  first_reference76_frame=next((x['frame'] for x in actions if x['cursor_before']>=76),None),
  final_origin_error_to_reference76_m=float(np.linalg.norm(bucket(q)-bucket(env.reference_q[phase]))),
  reference70_to76_origin_displacement_m=float(np.linalg.norm(bucket(env.reference_q[76])-bucket(env.reference_q[70]))),
  reference70_to76_axis_displacement_si=env.reference_q[76,r.chain.axes]-env.reference_q[70,r.chain.axes],
  reference70_to76_target_displacement_si=env.reference_targets[76]-env.reference_targets[70],
  reference_commands60to100_max_abs_si=np.max(abs(env.commands[60:101]),axis=0),virtual_wait_actions=virtual,
  original_guards_unchanged=True,physics_steps_by_diagnosis=0,policy_predictions=0,training_updates=0,
  limitation='CPU counterfactual command mapping only; no claim a modified wait controller can physically recover. No experiment or interface changed.')
 m.write(D/'stall-diagnosis.json',result);np.savez(D/'stall-reference-window.npz',reference_q=env.reference_q[60:101],reference_targets=env.reference_targets[60:101],commands=env.commands[60:101],origin_world_m=positions)
 print(m.encode(result),flush=True)
if __name__=='__main__':main()
