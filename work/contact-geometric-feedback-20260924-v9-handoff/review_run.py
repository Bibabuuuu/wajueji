"""Recompute control decisions from recorded actual states, without physics."""
import json,sys
from pathlib import Path
import numpy as np
from contact_bc_inference import sha,LIMITS
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
run=ROOT/sys.argv[1];report=read(run/'report.json');actions=rows(run/'actions.ndjson');physical=rows(run/'physics/frames.ndjson')
n=len(physical);k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
source=Path(report['source']['path']);initial=np.load(source);q=initial['mechanism_qpos']
ref=GroundSoilCutReference(initial['particle_q'],read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'))
guard=BucketGroundGuard(a,q);f=ContactGeometryFeedback(a,c,guard,report['selected_goal'],q,FeedbackConfig(**report['feedback_parameters']),ref);p=ContactCommandProjector(c)
command_error=0.;feedback_equal=True
for event in actions:
 req,cmd,detail=f.command(frame=event['frame'],qpos=event['before_state']['mechanism_qpos'],phase=event['phase'],bc_action=event['action_normalized'],projector=p)
 command_error=max(command_error,float(np.max(abs(req-event['requested_command_si']))),float(np.max(abs(cmd-event['applied_command_si']))))
 feedback_equal &= detail==event['feedback']
endpoint=run/('final.npz' if report['status']=='bounded_probe_completed' else 'physics/failure.npz');final=np.load(endpoint)
ground=[guard.measure(row['after_state']['mechanism_qpos']) for row in physical]
coordinates=np.array([f.coordinates(row['after_state']['mechanism_qpos']) for row in physical])
trace=rows(run/'phase-trace.ndjson')
checks=dict(frames=n==report['frames_completed'] and len(actions)==n+(1 if report['status']=='failed' else 0),exact_restore=all(report['zero_step_arrays_exact'].values()),
 control_recomputed=command_error<1e-12 and feedback_equal,
 commands=all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(actions,physical)),
 actual_previous=all(actions[i]['before_state']==physical[i-1]['after_state'] for i in range(1,len(actions))),
 previous_actions=all(actions[i]['previous_applied_action_si']==actions[i-1]['applied_command_si'] for i in range(1,len(actions))),
 phase_timing=all(x['latest_allowed_phase_evidence_frame']==x['frame']-1 for x in actions),
 phase_action=all(e['phase']==t['phase']['phase_before_observation'] for e,t in zip(actions,trace)),
 phase_next=all(actions[i]['phase']==trace[i-1]['phase']['phase_for_next_decision'] for i in range(1,min(len(actions),len(trace)+1))),
 finite=all(np.isfinite(final[key]).all() for key in final.files if final[key].dtype.kind in 'fciu'),
 final_physical=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),
 particles=initial['particle_q'].shape==final['particle_q'].shape,rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
 source=sha(source)==report['source']['sha256'],policy=sha(report['policy']['path'])==report['policy']['sha256'])
contract=read(report['phase_contract']['path'])
checks['protocol_hash']=sha(report['phase_contract']['path'])==report['phase_contract']['sha256']
checks['protocol_inputs']=all(sha(x['path'])==x['sha256'] for x in contract['inputs'])
checks['config_bound']=report['feedback_parameters']==contract['feedback_parameters']
checks['lineage_inventory']=all(read(str(endpoint)+'.json')[key]==read(str(source)+'.json')[key] for key in ('lineage_root','training_reset'))
if report['contact_window'] or report.get('scoop_window'):
 checks['prefix_exact']=report['prefix_frames_verified']==(report.get('handoff_review_frames',900) if report.get('scoop_window') else 120) and all(report['prefix_final_arrays_exact'].values())
for name,digest in report['runtime_bindings'].items():checks['runtime_'+name]=sha(ROOT/'src/v4'/name)==digest
for name,digest in report['code_sha256'].items():checks['code_'+name]=sha(ROOT/'src/v4'/name)==digest
behavior=dict(no_failure=report['status']=='bounded_probe_completed',all_hard=all(x['hard_guard'] for x in physical),all_material=all(x['hencky_material_guard']['safe'] for x in physical),all_sweep=all(x['moving_bucket_sweep']['safe'] for x in physical),all_measured_ground=all(x['safe'] for x in ground),lateral_corridor=float(np.max(abs(coordinates[:,1])))<=.10,entered_curl=any(e['phase']=='curl_load' for e in actions),bc_controlled_after_curl=any(e['feedback']['mode']=='bc_with_lateral_depth_feedback' for e in actions))
depth_undershoot=max([max(0.,e['feedback']['cut_target'][2]-coord[2]) for e,coord in zip(actions,coordinates) if e['feedback']['cut_target'] is not None],default=0.)
behavior['cut_depth_respected']=bool(depth_undershoot<=.03)
behavior['root_reached_soil_before_bc']=all(e['feedback']['root_ready'] for e in actions if e['phase']=='curl_load')
vertical=[e for e in actions if e['feedback']['soil_plan'] is not None and e['feedback']['coordinates'][3]>=e['feedback']['soil_plan']['vertical_pitch_rad']]
behavior['root_at_vertical']=bool(vertical and vertical[0]['feedback']['root_above_surface_m']<=.01)
approach=[abs(e['feedback']['coordinates'][3]-e['feedback']['target'][3]) for e in actions if e['phase']=='approach_contact']
behavior['approach_pitch_tracking']=max(approach,default=0.)<.03
result=dict(maximum_cut_depth_undershoot_m=depth_undershoot,maximum_approach_pitch_error_rad=max(approach,default=0.),audit_passed=all(checks.values()),checks=checks,behavior_gate_passed=all(behavior.values()),behavior=behavior,frames=n,status=report['status'],failure=report.get('failure'),events=report['events'],maximum_lateral_error_m=float(np.max(abs(coordinates[:,1]))),minimum_measured_bucket_z_m=min(x['minimum_bucket_z_m'] for x in ground),final_coordinates=coordinates[-1].tolist(),maximum_command_recompute_error=command_error,phase_counts={phase:sum(e['phase']==phase for e in actions) for phase in ('approach_contact','penetrate','curl_load','lift','off_soil')},bc_raw_vs_applied_mean_abs_normalized=np.mean([abs(np.asarray(e['action_normalized'])-np.asarray(e['applied_command_si'])/LIMITS) for e in actions],axis=0).tolist(),source=report['source'],policy=report['policy'],run=str(run),physics_steps_added_by_review=0,qualification='bounded geometry/BC handoff only; no full scoop or fill qualification',outputs=[{'path':str(run/name),'sha256':sha(run/name)} for name in ('report.json','actions.ndjson','phase-trace.ndjson','physics/frames.ndjson')])
if report['status']=='bounded_probe_completed':result['outputs'] += [{'path':str(endpoint),'sha256':sha(endpoint)}]
result['soil_plan']=f.soil_plan
result['first_vertical']=None if not vertical else dict(frame=vertical[0]['frame'],root_above_surface_m=vertical[0]['feedback']['root_above_surface_m'])
result['complete_scoop']=report.get('final_phase')=='off_soil'
result['pending_failed_step_commands']=len(actions)-n
if report['status']=='failed':
 result['primary_failure']=read(run/'physics/failure.json')['error']
 result['failure_state_scope']='Failure checkpoint is diagnostic only. Failed step may have partly advanced coupled physics; last valid returned rigid state is frame '+str(n)+'. No continuation qualification.'
contact_events=[e for e in actions if e['feedback']['cut_target'] is not None]
result['tip_vertical_motion_m']=dict(initial_z=float(f.coordinates(initial['mechanism_qpos'])[2]),minimum_z=float(coordinates[:,2].min()),initial_to_minimum=float(f.coordinates(initial['mechanism_qpos'])[2]-coordinates[:,2].min()))
if contact_events:
 contact_z=contact_events[0]['feedback']['coordinates'][2]
 result['tip_vertical_motion_m'].update(contact_confirmation_z=contact_z,contact_to_minimum=float(contact_z-coordinates[:,2].min()),derived_contact_to_target=float(contact_z-f.cut_target[2]))
(W/(run.name+'-review.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['audit_passed']
