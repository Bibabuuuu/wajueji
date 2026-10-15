"""CPU timeline evidence; no changes to alignment/controller/physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';sys.path.insert(0,str(W))
import run_long_qerror as m
import numpy as np
def main():
 D=m.OUT;p=m.read(D/'report.json');a=[json.loads(s) for s in (D/'policy/episode-0001/actions.ndjson').read_text().splitlines()];states=m.read(D/'display/states.json')
 e=m.YawTransferEnv(W/'diagnosis-reference',None,collect=False,waiting_action_mode='error_strength_v1',yaw_offset_rad=m.CANDIDATE['yaw_offset_rad'])
 wait=[x for x in a if x['execution_mode']=='wait'];blocks=[]
 for row in wait:
  if not blocks or row['frame']!=blocks[-1][-1]['frame']+1:blocks.append([])
  blocks[-1].append(row)
 summaries=[]
 for block in blocks:
  errors=np.array([x['alignment']['actual_joint_error_si'] for x in block]);q=np.array([states[x['frame']-1]['mechanism_qpos'] for x in block])[:,[7,9,12,15]];targets=np.array([states[x['frame']-1]['controller_target'] for x in block]);requests=np.array([x['request_si'] for x in block]);applied=np.array([x['applied_si'] for x in block])
  # A fixed-reference block is particularly useful for diagnosing native
  # target/actual tracking; do not conflate it with a new reference command.
  same_cursor=len(set(x['cursor_before'] for x in block))==1
  summaries.append(dict(first_frame=block[0]['frame'],last_frame=block[-1]['frame'],frames=len(block),first_cursor=block[0]['cursor_before'],last_cursor=block[-1]['cursor_before'],fixed_reference=same_cursor,follow_resumed=block[-1]['frame']<len(a),first_actual_joint_error_si=errors[0],last_actual_joint_error_si=errors[-1],maximum_abs_actual_joint_error_si=np.max(abs(errors),axis=0),first_native_target_minus_actual_q_si=targets[0]-q[0],last_native_target_minus_actual_q_si=targets[-1]-q[-1],native_target_motion_si=targets[-1]-targets[0],actual_q_motion_si=q[-1]-q[0],max_abs_request_si=np.max(abs(requests),axis=0),max_request_applied_difference=float(np.max(abs(requests-applied))),error_sign_changes=np.sum(errors[1:]*errors[:-1]<0,axis=0)))
 maxblock=max(blocks,key=len) if blocks else []
 last=a[-1];al=last['alignment'];cursor=last['cursor_before'];i=min(int(cursor),1170);j=min(i+1,1170);f=cursor-i;ref=e.reference_q[i]*(1-f)+e.reference_q[j]*f
 result=dict(status='cpu_timeline_diagnosed',run_status=p['status'],failure=p.get('failure'),frames=p['frames'],reference_cursor=p.get('reference_cursor'),actual_progress=al['actual_progress'],recovered_reference76=p.get('reference_cursor',0)>76,waiting_frames=len(wait),follow_frames=sum(x['execution_mode']=='follow' for x in a),hold_frames=sum(x['execution_mode']=='hold' for x in a),waiting_blocks=summaries,last_actual_q_minus_reference_q_si=np.asarray(states[-1]['mechanism_qpos'])[[7,9,12,15]]-ref[[7,9,12,15]],last_native_target_minus_reference_q_si=np.asarray(states[-1]['controller_target'])-ref[[7,9,12,15]],last_native_target_minus_actual_q_si=np.asarray(states[-1]['controller_target'])-np.asarray(states[-1]['mechanism_qpos'])[[7,9,12,15]],max_reference_lead=max(x['cursor_before']-x['alignment']['actual_progress'] for x in a),new_physics_steps=0,training_updates=0,limitation='Timeline measurements only. Wait recovery does not prove full scoop. More frames or control law changes require a separate confirmed scope; no new controller or phase rule is implemented here.')
 m.write(D/'wait-timeline-diagnosis.json',result)
 print(m.encode(result),flush=True)
if __name__=='__main__':main()
