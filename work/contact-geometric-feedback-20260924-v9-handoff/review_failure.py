"""Read-only failure timing and command/response separation."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
W=Path(__file__).resolve().parent;ROOT=W.parents[1];run=ROOT/'runs/contact-geometric-probe-20260924-v9-handoff/contact1200'
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
a=rows(run/'actions.ndjson');p=rows(run/'physics/frames.ndjson');g=rows(run/'ground-guard.ndjson');report=json.loads((run/'report.json').read_text())
windows=[]
for i in range(949,len(a)):
 x=a[i];before=x['before_state'];after=p[i]['after_state'];q=np.array(after['mechanism_qpos'])[[7,9,12,15]];previous=np.array(before['mechanism_qpos'])[[7,9,12,15]]
 windows.append(dict(frame=i+1,phase=x['phase'],bc=x['action_normalized'],command_si=x['applied_command_si'],predicted_task_velocity=x['feedback']['predicted_task_velocity'],qpos=q.tolist(),endpoint_qvel=np.array(after['mechanism_qvel'])[[6,8,11,14]].tolist(),frame_mean_velocity=((q-previous)*60).tolist(),controller_target=after['controller_target'],tracking_error=(np.array(after['controller_target'])-q).tolist(),bucket_integral_n=after['bucket_integral_effort'][3],bucket_active_n=after['bucket_active_curl_effort'][3],previous_drive_load=x['previous_drive_load'],rock_bucket_substeps=p[i]['rock_bucket_contact_substeps'],slew_override=x['feedback'].get('slew_safety_override')))
assert len(a)==len(p)==report['frames_completed']==972
with np.load(run/'physics/failure.npz') as d:assert np.array_equal(d['mechanism_qpos'],p[-1]['after_state']['mechanism_qpos'])
r=dict(physics_steps_added=0,frames=972,failed_decision_frame=973,failed_command_dispatched=False,minimum_ground_m=g[-1]['clearance_m'],physical_guard_minimum_m=.005,preview_minimum_m=.03,events=report['events'],windows=windows,findings=['Alternating measured bucket-cylinder velocity grows before BC handoff at frame960; handoff alone is not a proven cause','At frame965 commanded bucket velocity is about0.0273m/s, actual endpoint velocity about0.4137m/s','The active curl effort is already zero from frame963 onward while alternation persists; cannot blame an instantaneous active-assist switch alone','From frame961 geometric requested tip vertical velocity is upward while actual descent persists','At frame973 actual CAD minimum29.5766mm is already below30mm preview floor, so even zero preview cannot meet it; refusal occurs before dispatch','All returned physical frames pass5mm ground/hard/material/sweep guards; depth overshoot55.069mm fails original30mm behavior gate'],limits=['Correlated60Hz loads/rock contacts do not isolate4800Hz contact dynamics versus staggered MPM coupling','No bottom-layer gains/physics changes; no off-soil/fill qualification'],bindings=[dict(path=str(run/f),sha256=sha(run/f)) for f in ('report.json','actions.ndjson','physics/frames.ndjson','physics/failure.npz','physics/failure.json')])
(W/'failure-diagnosis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='windows'},indent=2))
