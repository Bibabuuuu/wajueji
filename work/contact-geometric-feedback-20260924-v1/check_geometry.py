"""Kinematic checks only: tangent against independent closed-chain continuation."""
import json
from pathlib import Path
import numpy as np
from layered_excavation_approach_planner import _load_kinematics
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_bc_ground_guard import BucketGroundGuard
from contact_bc_inference import LIMITS,sha
from contact_phase_live import mouth_angle_deg
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
W=Path(__file__).resolve().parent;root=W.parents[1]
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
source=root/'runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz'
meta=json.loads(Path(str(source)+'.json').read_text());q=np.load(source)['mechanism_qpos'];initial=q.copy()
face=meta['selected_cut'].get('contact',meta['selected_cut']);goal={'entry_xy_m':face.get('contact_xy_m',face.get('entry_xy_m')),'surface_z_m':face.get('contact_surface_z_m',face.get('surface_z_m'))}
f=ContactGeometryFeedback(a,c,BucketGroundGuard(a,q),goal,q);proj=ContactCommandProjector(c)
j,g=f.linearize(q);errors=[]
for i in range(4):
 velocity=LIMITS*np.eye(4)[i]*.1;dt=.001
 positions=[]
 for sign in (-1,1):
  c.q=q.copy();test,_=proj._advance_preview(sign*velocity,dt);positions.append(f.coordinates(test))
 measured=(positions[1]-positions[0])/(2*dt*.1);errors.append(float(np.max(abs(measured-j[:,i]))))
assert max(errors)<1e-4,errors
# A synthetic semantic switch tests geometry only, never claims soil contact.
phase='approach_contact';trace=[];contact_target=None
for frame in range(1,901):
 bc=np.array([.2,-.8,.8,.5])
 req,cmd,detail=f.command(frame=frame,qpos=q,phase=phase,bc_action=bc,projector=proj)
 c.q=q.copy();scale=max(1.,float(np.max(abs(cmd)/np.array([.12,.0225,.0225,.0225]))));q,_=proj._advance_preview(cmd/scale,scale/60)
 trace.append(dict(frame=frame,phase=phase,detail=detail,qpos=q.tolist()))
 if phase=='approach_contact' and f.coordinates(q)[2]<goal['surface_z_m']+.15:phase='penetrate'
 if phase=='penetrate' and f.coordinates(q)[3]>.325:
  phase='curl_load';contact_target=f.cut_target.copy()
 if phase=='curl_load' and sum(t['phase']=='curl_load' for t in trace)>=30:break
assert phase=='curl_load',(phase,trace[-1])
assert np.max(abs(np.array([t['detail']['coordinates'][1] for t in trace])))<.02
assert min(t['detail']['preview_minimum_clearance_m'] for t in trace)>=.03-1e-6
assert np.array_equal(contact_target,f.cut_target)
report=dict(passed=True,scope='kinematics only; synthetic contact transition; no physics or soil evidence',frames=len(trace),tangent_maximum_abs_error=max(errors),minimum_preview_clearance_m=min(t['detail']['preview_minimum_clearance_m'] for t in trace),maximum_lateral_error_m=max(abs(t['detail']['coordinates'][1]) for t in trace),final_phase=phase,source_sha256=sha(source),final_coordinates=f.coordinates(q).tolist(),code_sha256=sha(root/'src/v4/contact_geometric_feedback.py'))
(W/'geometry-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
