"""Counterfactual command check on frozen v6 actual poses; no physics."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha,LIMITS
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
run=ROOT/'runs/contact-geometric-probe-20260924-v7/scoop1500'
read=lambda p:json.loads(p.read_text())
r=read(run/'report.json');events=[json.loads(s) for s in (run/'actions.ndjson').read_text().splitlines()]
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
data=np.load(r['source']['path']);q=data['mechanism_qpos']
ref=GroundSoilCutReference(data['particle_q'],read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'))
f=ContactGeometryFeedback(a,c,BucketGroundGuard(a,q),r['selected_goal'],q,FeedbackConfig(**{**r['feedback_parameters'],'late_curl_no_down':True,'command_rate_per_s':2.}),ref)
p=ContactCommandProjector(c);records=[];error=None
for e in events:
 try:
  req,cmd,d=f.command(frame=e['frame'],qpos=e['before_state']['mechanism_qpos'],phase=e['phase'],bc_action=e['action_normalized'],projector=p)
  records.append(dict(frame=e['frame'],old_command=e['applied_command_si'],command=cmd.tolist(),**d))
 except Exception as exc:
  error=dict(frame=e['frame'],error=repr(exc));break
active=[x for x in records if x['no_down_latched']]
checks=dict(all_frames=len(records)==len(events),trigger_exists=bool(active),no_boom_down=all(e['command'][1]>=-1e-10 for e in active),no_body_down=all(e['no_down_minimum_velocity_m_s']>=-1e-7 and e['no_down_preview_minimum_delta_m']>=-1e-6 for e in active),keeps_curl=any(e['command'][3]>.001 for e in active))
result=dict(passed=all(checks.values()),checks=checks,error=error,frames=len(records),trigger=active[0] if active else None,active_frames=len(active),zeroed_frames=sum(not np.any(e['command']) for e in active),preview_scale_counts={str(v):sum(e['preview_scale']==v for e in active) for v in (1.,.5,.25,.125,0.)},minimum_velocity=min([e['no_down_minimum_velocity_m_s'] for e in active],default=None),minimum_preview_delta=min([e['no_down_preview_minimum_delta_m'] for e in active],default=None),last=records[-1] if records else None,source_actions_sha256=sha(run/'actions.ndjson'),code_sha256=sha(ROOT/'src/v4/contact_geometric_feedback.py'),scope='counterfactual commands on old actual states, zero physics; does not qualify new trajectory, braking, lift or fill')
(W/'recorded-state-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed']
