"""Independent control replay and half-step bookkeeping audit, no physics."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from gates import behavior_gate
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
results={}
for name in ('baseline60','candidate120'):
 run=ROOT/'runs/contact-coupling-120hz-20260924-v1'/name
 if not (run/'report.json').exists():continue
 report=read(run/'report.json');actions=rows(run/'actions.ndjson');physical=rows(run/'physics/frames.ndjson');trace=rows(run/'phase-trace.ndjson')
 initial=np.load(report['source']['path']);q=initial['mechanism_qpos']
 ref=GroundSoilCutReference(initial['particle_q'],read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'))
 guard=BucketGroundGuard(a,q);feedback=ContactGeometryFeedback(a,c,guard,report['selected_goal'],q,FeedbackConfig(**report['feedback_parameters']),ref);p=ContactCommandProjector(c)
 error=0.;detail_equal=True
 for event in actions:
  req,cmd,detail=feedback.command(frame=event['frame'],qpos=event['before_state']['mechanism_qpos'],phase=event['phase'],bc_action=event['action_normalized'],projector=p)
  error=max(error,float(np.max(abs(req-event['requested_command_si']))),float(np.max(abs(cmd-event['applied_command_si']))));detail_equal &= detail==event['feedback']
 endpoint=run/('final.npz' if report['status']=='bounded_probe_completed' else 'physics/failure.npz');final=np.load(endpoint);meta=read(str(endpoint)+'.json');source_meta=read(report['source']['path']+'.json')
 checks=dict(exact_restore=all(report['zero_step_arrays_exact'].values()),control_recomputed=error<1e-12 and detail_equal,
  frames=len(physical)==report['frames_completed'],source=sha(report['source']['path'])==report['source']['sha256'],
  policy=sha(report['policy']['path'])==report['policy']['sha256'],
  lineage=all(meta[key]==source_meta[key] for key in ('lineage_root','training_reset')),
  particles=initial['particle_q'].shape==final['particle_q'].shape,rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
  finite=all(np.isfinite(final[key]).all() for key in final.files if final[key].dtype.kind in 'fciu'),
  phase_action=all(e['phase']==t['phase']['phase_before_observation'] for e,t in zip(actions,trace)),
  phase_next=all(actions[i]['phase']==trace[i-1]['phase']['phase_for_next_decision'] for i in range(1,min(len(actions),len(trace)+1))),
  protocol=sha(report['phase_contract']['path'])==report['phase_contract']['sha256'],
  candidate_code=all(sha(path)==digest for path,digest in report['candidate_code'].items()))
 for file,digest in report['runtime_bindings'].items():checks['runtime_'+file]=sha(ROOT/'src/v4'/file)==digest
 for file,digest in report['code_sha256'].items():checks['code_'+file]=sha(ROOT/'src/v4'/file)==digest
 contract=read(report['phase_contract']['path']);checks['inputs']=all(sha(v['path'])==v['sha256'] for v in contract['inputs'])
 if name=='candidate120':
  exchanges=rows(run/'physics/exchanges.ndjson')
  checks.update(half_count=len(exchanges)>=2*len(physical),half_order=all(e['exchange_in_frame']==i%2+1 and e['frame']==i//2+1 for i,e in enumerate(exchanges)),
   all_returned_half_guards=all(e['hard_guard'] and e['ground_check']['safe'] for e in exchanges[:2*len(physical)]),
   isolated_checkpoint_schema=meta['schema']=='layered_47_coupling_candidate_checkpoint_v1',
   native_elapsed_matches_checkpoint=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-(meta['controller_state']['native_time_s']-source_meta['controller_state']['native_time_s']))<1e-10)
  last=exchanges[-1]['after_state'];expected_elapsed=len(exchanges)/120
 else:last=physical[-1]['after_state'];expected_elapsed=len(physical)/60
 checks['endpoint_matches_last_exchange']=np.array_equal(final['mechanism_qpos'],last['mechanism_qpos'])
 checks['simulation_clock']=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-expected_elapsed)<1e-8
 behavior=behavior_gate(actions[:len(physical)],physical,feedback,require_curl=name=='candidate120')
 results[name]=dict(audit_passed=all(checks.values()),checks=checks,maximum_recomputed_command_error=error,behavior=behavior,status=report['status'],failure=report.get('failure'),physics_steps_added=0)
(W/'audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
assert all(v['audit_passed'] for v in results.values())
