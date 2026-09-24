"""Counterfactual control feasibility at v2 failure; zero physics steps."""
import json
from pathlib import Path
import numpy as np
from contact_geometric_feedback import ContactGeometryFeedback
from contact_bc_ground_guard import BucketGroundGuard
from contact_bc_inference import sha
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
w=Path(__file__).resolve().parent;root=w.parents[1];run=root/'runs/contact-geometric-probe-20260924-v2/contact900'
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
r=json.loads((run/'report.json').read_text());last=json.loads((run/'actions.ndjson').read_text().splitlines()[-1]);q=np.load(run/'physics/failure.npz')['mechanism_qpos']
f=ContactGeometryFeedback(a,c,BucketGroundGuard(a,q),r['selected_goal'],q);f.last_frame=518;f.last_phase='penetrate';f.contact=np.array(last['feedback']['contact']);f.cut_target=np.array(last['feedback']['cut_target']);req,cmd,detail=f.command(frame=519,qpos=q,phase='penetrate',bc_action=last['action_normalized'],projector=ContactCommandProjector(c))
assert detail['preview_scale']==1.,detail
assert detail['predicted_task_velocity'][2]>=detail['depth_velocity_lower']-1e-7
j,_=f.linearize(q)
report=dict(passed=True,scope='v2 failed state control only; BC irrelevant during penetrate; no physical advance',v2_requested_pitch_rate=detail['desired_task_velocity'][3],axis_box_maximum_pitch_rate=float(abs(j[3]).sum()),v3_control=detail,applied_command_si=cmd.tolist(),source_failure_sha256=sha(run/'physics/failure.npz'),code_sha256=sha(root/'src/v4/contact_geometric_feedback.py'))
(w/'failure-state-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
