"""CPU-only regression of the unchanged early command law on both old sources."""
from pathlib import Path
import json
import numpy as np
from candidate_multisource import W,ROOT,LiftFeedback,sha
from contact_geometric_feedback import FeedbackConfig
from contact_bc_ground_guard import BucketGroundGuard
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector

def main():
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,chain=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh');results=[]
 for kind in ('first','second'):
  run=ROOT/f'runs/contact-{kind}-source-20260924-v1/startup120';report=json.loads((run/'report.json').read_text());contract=json.loads((W/(kind+'-protocol.json')).read_text());source=report['source']['path']
  with np.load(source) as f:q=f['mechanism_qpos'].copy();soil=f['particle_q'].copy()
  guard=BucketGroundGuard(assembly,q);proxy=json.loads((ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text());fb=LiftFeedback(assembly,chain,guard,report['selected_goal'],q,FeedbackConfig(**contract['feedback_parameters']),GroundSoilCutReference(soil,proxy));projector=ContactCommandProjector(chain)
  n=0;changed_keys=set();maximum_clearance_difference=0.
  for line in (run/'actions.ndjson').read_text().splitlines():
   e=json.loads(line);req,cmd,detail=fb.command(frame=e['frame'],qpos=e['before_state']['mechanism_qpos'],phase=e['phase'],bc_action=e['action_normalized'],projector=projector)
   assert req.tolist()==e['requested_command_si'] and cmd.tolist()==e['applied_command_si']
   # New lift diagnostics are additive; no late preview occurs in these traces.
   old_detail={k:v for k,v in detail.items() if k not in ('lift_intent','lift_start_origin_z_m')};changed_keys.update(k for k,v in old_detail.items() if v!=e['feedback'][k]);maximum_clearance_difference=max(maximum_clearance_difference,abs(old_detail['preview_minimum_clearance_m']-e['feedback']['preview_minimum_clearance_m']));n+=1
  if changed_keys-{'preview_minimum_clearance_m'}:raise ValueError('unexpected_early_diagnostic_change:'+str(changed_keys))
  results.append(dict(scoop=kind,frames=n,commands_exact=True,changed_diagnostic_fields=sorted(changed_keys),maximum_preview_clearance_difference_m=maximum_clearance_difference,source_sha256=sha(source),physics_steps_added=0))
 (W/'early-command-review.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
