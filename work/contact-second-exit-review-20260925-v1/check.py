"""CPU-only reproduction of the rejected transition, no physics or mutations."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/contact-second-finish-unload-20260925-v1'))
from controller import CarryFeedback,LIMITS,yaw_preview
from layered_excavation_approach_planner import _load_kinematics
from contact_bc_ground_guard import BucketGroundGuard
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
source=ROOT/'runs/contact-next-scoop-20260925-v1/candidate/final.npz'
control=json.loads((source.parent/'control-state.json').read_text())
saved=json.loads((ROOT/'runs/contact-second-finish-unload-20260925-v1/physical/control-state.json').read_text())
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
with np.load(source) as f:q=f['mechanism_qpos'].copy()
g=BucketGroundGuard(a,q);prev=np.array(control['last_action']);p=ContactCommandProjector(c)
fb=CarryFeedback(a,c,g,q,prev,frame_end=control['frame'],initial_soil_top_m=saved['carry']['initial_soil_top_m'])
result=dict(physics_steps_added=0,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),previous_command=prev.tolist())
original=p.project
def trace(q,requested):
 applied=original(q,requested)
 if 'applied' not in result:result.update(requested=requested.tolist(),applied=applied.tolist(),projection_reason=p.reason)
 return applied
p.project=trace
try:fb.command(frame=control['frame']+1,qpos=q,projector=p)
except ValueError as e:result['reproduced_failure']=str(e)
jac,gg,bg=fb.linearize(q,include_body=True);applied=np.array(result['applied']);rates=bg@(applied/LIMITS)
result.update(corner_vertical_velocity_m_s=rates.tolist(),minimum_corner_vertical_velocity_m_s=float(min(rates)),
              opening_command_rejected=bool(applied[3]<-1e-10),boom_down_command_rejected=bool(applied[1]<-1e-10),corner_down_rejected=bool(min(rates)<-1e-7),
              pitch_velocity_rad_s=float(jac[3]@(applied/LIMITS)),closure_fraction=float((q[15]-.37)/(.852-.37)))
c.q=q.copy();ref=c.residual(q);rot=yaw_preview.rotation(c,q);start=fb.body_heights(q)
for i in range(9):pred,_=yaw_preview.preview(p,applied,1/60,ref,rot)
result.update(preview_150ms_corner_delta_m=(fb.body_heights(pred)-start).tolist(),preview_cylinder_delta_m=float(pred[15]-q[15]),preview_pitch_delta_rad=float(fb.coordinates(pred)[3]-fb.coordinates(q)[3]))
last=json.loads((source.parent/'actions.ndjson').read_text().splitlines()[-1]);result['previous_frame_minimum_velocity_m_s']=last['feedback']['no_down_minimum_velocity_m_s']
# Diagnostic only: test normal rate-limited computation instead of exact repeat.
fb.first_command=False;c.q=q.copy()
try:
 cmd,detail=fb.command(frame=control['frame']+1,qpos=q,projector=p)
 result['rate_limited_recompute']=dict(passed=True,command=cmd.tolist(),normalized_change=float(np.max(abs((cmd-prev)/LIMITS))),limit=2/60)
except ValueError as e:result['rate_limited_recompute']=dict(passed=False,failure=str(e))
(W/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
