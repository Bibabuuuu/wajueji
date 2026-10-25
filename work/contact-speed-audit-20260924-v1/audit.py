"""Read frozen logs only. Distinguish request steps from physical response."""
import json, hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent; ROOT=W.parents[1]
run=ROOT/'runs/contact-geometric-probe-20260924-v7/scoop1500'
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
a=rows(run/'actions.ndjson');r=rows(run/'physics/frames.ndjson');phase=rows(run/'phase-trace.ndjson')
assert len(a)==len(r)==len(phase)==1500
assert all(x['frame']==y['frame']==i+1 and x['applied_command_si']==y['applied_command_si'] for i,(x,y) in enumerate(zip(a,r)))
fields={'bc_normalized':np.array([x['action_normalized'] for x in a]),'requested_si':np.array([x['requested_command_si'] for x in a]),'applied_si':np.array([x['applied_command_si'] for x in a]),'predicted_task_velocity':np.array([x['feedback']['predicted_task_velocity'] for x in a]),'actual_axis_qvel_si':np.array([np.array(x['after_state']['mechanism_qvel'])[[6,8,11,14]] for x in r]),'frame_mean_axis_velocity_si':np.diff(np.array([a[0]['before_state']['mechanism_qpos']]+[x['after_state']['mechanism_qpos'] for x in r]),axis=0)[:,[7,9,12,15]]*60,'controller_target_velocity_si':np.array([x['after_state']['controller_target_velocity'] for x in r]),'actual_bucket_linear_m_s':np.array([x['after_state']['bucket_linear_velocity_world_m_s'] for x in r])}
def sample(i):
 x=a[i];s=r[i]['after_state']
 return dict(frame=i+1,phase=x['phase'],fields={k:v[i].tolist() for k,v in fields.items()},feedback=x['feedback'],bucket_integral_n=s['bucket_integral_effort'][3],bucket_active_n=s['bucket_active_curl_effort'][3],previous_drive_load=x['previous_drive_load'],rock_bucket_substeps=r[i]['rock_bucket_contact_substeps'],soil_force_n=phase[i]['bucket_mpm_force_norm_n'])
stats={}
for k,v in fields.items():
 d=np.abs(np.diff(v,axis=0));idx=np.argsort(d.max(axis=1))[-10:][::-1]+1
 stats[k]=dict(peak_absolute_per_axis=np.abs(v).max(axis=0).tolist(),max_step_per_axis=d.max(axis=0).tolist(),top_changes=[dict(frame=int(i+1),delta=(v[i]-v[i-1]).tolist(),before=v[i-1].tolist(),after=v[i].tolist()) for i in idx])
indices=sorted(set(i for n in [423,711,824,981,1088] for i in range(n-3,n+2))|set(range(973,1006)))
result=dict(source_run=str(run),physics_steps_added=0,axes=['swing_rad_s','boom_m_s','arm_m_s','bucket_m_s'],qpos_addresses=[7,9,12,15],qvel_addresses=[6,8,11,14],projection_modified_frames=sum(x['requested_command_si']!=x['applied_command_si'] for x in a),statistics=stats,windows=[sample(i) for i in indices],bindings=[dict(path=str(run/f),sha256=hashlib.sha256((run/f).read_bytes()).hexdigest()) for f in ['actions.ndjson','physics/frames.ndjson','phase-trace.ndjson','report.json']],findings=['423: feedback target changes at confirmed contact; requested boom changes +0.00773 to -0.01480 m/s','711: root_ready removes pitch upper inequality; bucket request changes +0.03138 to +0.05417 m/s','824: feedback to BC handoff changes arm and bucket requests without a slew contract','981: safety latch forces new coupled action; safety cannot be delayed by post-projection filtering','980: physical bucket-cylinder frame-average velocity is 0.21611 m/s before latch; alternating loads and rock contact present, not causal proof of a unique actuator defect','1088: lift transition has small command change; not a dominant discontinuity'],limitations=['No 4800Hz substep contact/force causal isolation','No new physical trajectory, no off-soil or fill qualification'])
(W/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['windows','statistics']},indent=2))
