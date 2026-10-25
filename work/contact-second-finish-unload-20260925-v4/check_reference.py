"""CPU counterfactual command checks on saved v2 states; no new physics."""
import json,sys,ast
from pathlib import Path
import numpy as np
from controller import CarryFeedback,LIMITS,bounded_preview
from runtime import ROOT,W,SOURCE,read
from layered_excavation_approach_planner import _load_kinematics
from contact_bc_ground_guard import BucketGroundGuard
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
for p in W.glob('*.py'):ast.parse(p.read_text())
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
old=ROOT/'runs/contact-second-finish-unload-20260925-v2/physical'
rows=[json.loads(x) for x in (old/'actions.ndjson').read_text().splitlines()]
ctrl=read(SOURCE.parent/'control-state.json');q=np.array(rows[0]['before_state']['mechanism_qpos']);prev=np.array(ctrl['last_action'])
fb=CarryFeedback(a,c,BucketGroundGuard(a,q),q,prev,frame_end=ctrl['frame'],initial_soil_top_m=1.346285343170166);p=ContactCommandProjector(c)
results=[];maximum_change=0.
try:
 for row in rows:
  st=row['before_state'];q=np.array(st['mechanism_qpos']);oldcmd=fb.previous_command.copy()
  cmd,detail=fb.command(frame=row['frame'],qpos=q,projector=p,native_target=st['controller_target'],native_velocity=st['controller_target_velocity'])
  change=float(np.max(abs((cmd-oldcmd)/LIMITS)));maximum_change=max(maximum_change,change)
  assert change<=2/60+1e-9 and detail['zero_curl_center_velocity_m_s']>=-1e-7
  if len(results)%60==0:print('CHECK',len(results)+1,flush=True)
  results.append(dict(frame=row['frame'],command=cmd.tolist(),hold_latched=detail['hold_latched'],native_headroom_m=detail['native_headroom_m'],zero_curl_center_velocity_m_s=detail['zero_curl_center_velocity_m_s']))
 result=dict(passed=True,physics_steps_added=0,counterfactual_saved_states_only=True,frames=len(results),maximum_normalized_command_change=maximum_change,
             first_hold_frame=next((x['frame'] for x in results if x['hold_latched']),None),samples=results[::60]+results[-3:])
except Exception as e:
 result=dict(passed=False,physics_steps_added=0,frames=len(results),failure=repr(e),last=results[-1:] );raise
finally:
 (W/'reference-review.json').write_text(json.dumps(result,indent=2)+'\n')
