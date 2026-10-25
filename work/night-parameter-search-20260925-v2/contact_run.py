"""D154 acquisition with actual prior command/load and causal feature saving.

Targets, BC, stages, limits and physical guards are unchanged.
"""
import search_adapter
import argparse,json,time
from pathlib import Path
import numpy as np
from contact_bc_inference import structured_observation,LIMITS,sha
from contact_phase_bc_inference import PhaseInferenceWorker as InferenceWorker,with_phase
from features import CausalFeatures as OnlineFeatures
from contact_phase_live import LivePhaseObserver
from candidate_multisource import LiftFeedback as ContactGeometryFeedback,save_control
from contact_geometric_feedback import FeedbackConfig
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig
from run_layered_47_sparse_contact_handoff import wrench_summary


def probe_budget(frames,contact_window,scoop_window=False,handoff_review_frames=900):
    if type(handoff_review_frames) is not int or handoff_review_frames not in (900,1200):raise ValueError('invalid_handoff_review_frames')
    if not 0<=frames<=(2400 if scoop_window else handoff_review_frames if contact_window else 120):raise ValueError('bounded_bc_probe_frames')
    return 900 if scoop_window else 360 if contact_window else 180


def execution_summary(commands,latencies,loads):
    return dict(all_executed_frames_safe=all(v['hard_guard'] for v in loads),contact_summary=loads,
                projection_frames=sum(e['command_projection_reason'] is not None for e in commands),
                command_nonzero_frames=sum(bool(np.any(np.abs(e['applied_command_si'])>1e-8)) for e in commands),
                inference_roundtrip_seconds_quantiles=[] if not latencies else np.quantile(latencies,[0,.5,.95,1]).tolist())


def main():
    from candidate_multisource import MultisourceRuntime as CycleRuntime
    from gates import behavior_gate
    import run_layered_47_sparse_manual_contact_episode as manual
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--policy',required=True);p.add_argument('--policy-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=0)
    p.add_argument('--training-python',required=True);p.add_argument('--training-reset-checkpoint',action='store_true')
    p.add_argument('--coupling-hz',type=int,choices=(60,),required=True)
    p.add_argument('--contact-window',action='store_true',help='Bounded handoff window: 900 frames, or explicit protocol 1200')
    p.add_argument('--prefix-review',type=Path,help='Audited startup prefix required for contact extension')
    p.add_argument('--scoop-window',action='store_true',help='Audited contact extension, at most 2400 frames / 900 seconds')
    p.add_argument('--contact-review',type=Path)
    p.add_argument('--capture-display',action='store_true')
    p.add_argument('--phase-contract',type=Path,required=True)
    args=p.parse_args()
    contract=json.loads(args.phase_contract.read_text())
    feedback_config=FeedbackConfig(**contract['feedback_parameters'])
    if contract['schema']!='contact_multisource_lift_protocol_v1':raise ValueError('feedback_contract_schema')
    for item in contract['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('phase_contract_input_changed:'+item['path'])
    interface=json.loads(Path(contract['interface_review']['path']).read_text())
    if sha(contract['interface_review']['path'])!=contract['interface_review']['sha256'] or not interface['passed']:raise ValueError('phase_interface_not_verified')
    if sha(args.source)!=contract['source_sha256'] or args.policy_sha256!=contract['policy_sha256']:raise ValueError('phase_contract_source_or_policy')
    if interface['policy_sha256']!=args.policy_sha256:raise ValueError('phase_interface_policy')
    phase_config=json.loads(Path(contract['contact_config']).read_text())['parameters']
    completion_config=json.loads(Path(contract['completion_config']).read_text())['parameters']
    polygon=json.loads(Path(contract['cavity_config']).read_text())['bucket']['polygon_xz_m']
    handoff_frames=1200
    if args.frames not in (0,120,3000):
        raise ValueError('D154_bounded_frames')
    if args.prefix_review or args.contact_review or args.contact_window or args.scoop_window:
        raise ValueError('D154_same_instance_gates_only')
    wall_budget=180 if args.frames<=120 else 900
    prefix_actions=[];prefix_rows=[];prefix_run=None
    source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='contact_geometric_feedback_probe_v1',status='initializing',frames_requested=args.frames,frames_completed=0,
                source=dict(path=str(source),sha256=sha(source)),policy=dict(path=str(Path(args.policy).resolve()),sha256=args.policy_sha256),
                no_scripted_excavation=False, geometric_feedback=True,manual_default_changed=False,policy_resume_qualified=False,formal_training_admission=False)
    report.update(contact_window=args.contact_window,scoop_window=args.scoop_window,wall_budget_seconds=wall_budget,prefix_frames_verified=0,events=[],
                  handoff_review_frames=handoff_frames,
                  phase_contract=dict(path=str(args.phase_contract.resolve()),sha256=sha(args.phase_contract)),phase_inputs=contract['inputs'])
    if args.prefix_review is not None:
        report['prefix_review']=dict(path=str(args.prefix_review.resolve()),sha256=sha(args.prefix_review))
    started=time.monotonic();commands=[];latencies=[];positions=[];loads=[];physical=[];frame_timings=[];r=None;cache=None
    try:
        with InferenceWorker(args.training_python,args.policy,args.policy_sha256,stderr_path=out/'inference-stderr.log') as worker:
            report['worker_ready']=worker.ready
            report.update(coupling_hz=args.coupling_hz,control_hz=60,rigid_hz=4800,same_instance_gates={},candidate_code={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
            with CycleRuntime(source,out/'physics',allow_training_reset=args.training_reset_checkpoint) as r:
                if r.metadata['stage']!='prepoint':raise ValueError('bc_probe_requires_prepoint')
                initial=r.capture();exact={k:bool(np.array_equal(initial[k],v)) for k,v in r.arrays.items()}
                report['zero_step_arrays_exact']=exact;report['restore']=dict(exact=r.restored.get('exact'),cycle_errors=r.restored['cycle_errors'])
                if not all(exact.values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('bc_prepoint_restore_inexact')
                from contact_bc_ground_guard import BucketGroundGuard
                ground_guard=BucketGroundGuard(r.assembly,r.bridge.state['mechanism_qpos'])
                report['bucket_ground_guard']=dict(geometry_sha256=ground_guard.geometry_sha256,ground_z_m=0.,margin_m=.005,
                    scope='sampled stop only; not continuous collision proof')
                def check_ground(frame,stage):
                    check=dict(frame=frame,stage=stage,**ground_guard.measure(r.bridge.state['mechanism_qpos']))
                    manual.append_json(out/'ground-guard.ndjson',check);report['last_ground_check']=check
                    if not check['safe']:raise RuntimeError('bc_bucket_ground_clearance:'+str(check))
                check_ground(0,'restored')
                selected=r.metadata.get('selected_cut');face=selected.get('contact',selected) if selected else None
                if face is None:raise ValueError('bc_missing_saved_target')
                goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
                report.update(selected_goal=goal,cycle_index=r.metadata['cycle_index'],lineage_root=r.metadata['lineage_root'],training_reset=r.metadata.get('training_reset'))
                vertices=manual._cache_rock_vertices(r.root,r.records);features=OnlineFeatures();terrain=None;anchor=0;previous_action=list(r.metadata['transport_controller_state']['previous_command']);previous_load=list(r.bridge.state['equivalent_drive_load'])
                pose=r.assembly.assemble(r.bridge.state['mechanism_qpos'],include_meshes=False)[1]['bucket']
                phases=LivePhaseObserver(r.proxy,polygon,ContactPhaseConfig(**phase_config),CompletionConfig(**completion_config),pose)
                cut_reference=None
                if feedback_config.terrain_relative_cut:
                    from contact_cut_reference import GroundSoilCutReference
                    cut_reference=GroundSoilCutReference(r.arrays['particle_q'],r.proxy)
                feedback=ContactGeometryFeedback(r.assembly,r.chain,ground_guard,goal,r.bridge.state['mechanism_qpos'],feedback_config,cut_reference)
                report['feedback_parameters']=contract['feedback_parameters']
                report['runtime_bindings']=r.bindings
                initial_position=np.asarray(manual.compact_state(r.bridge.state)['bucket_origin_world_m']);report['initial_bucket_origin_world_m']=initial_position.tolist()
                if args.capture_display:
                    from view_contact_bc_probe import ProbeDisplayCache
                    cache=ProbeDisplayCache(out/'display',args.frames,r.arrays['particle_q'].shape,
                        dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                report['startup_wall_seconds']=time.monotonic()-started
                for frame in range(1,max(1,args.frames)+1):
                    frame_started=time.monotonic()
                    if time.monotonic()-started>wall_budget:
                        report['early_stop_reason']='wall_budget_exhausted';break
                    before=manual.compact_state(r.bridge.state)
                    check_ground(frame,'before_step')
                    if terrain is None or (frame-1)%3==0:
                        terrain=manual.terrain_observation(r.fixture,r.bridge.state,vertices);anchor=frame-1
                        manual.append_json(out/'terrain.ndjson',dict(frame=anchor,terrain=terrain))
                    structured=structured_observation(frame=frame,state=before,terrain=terrain,terrain_frame=anchor,goal=goal,previous_action=previous_action,previous_load=previous_load)
                    phase_before=phases.router.phase_for_decision(frame)
                    structured=with_phase(structured,phase_before,frame-1)
                    x,digest=features.update(structured);action,timing=worker.predict(x,digest)
                    try:requested,applied,control=feedback.command(frame=frame,qpos=before['mechanism_qpos'],phase=phase_before,bc_action=action,projector=r.projector)
                    except ValueError as error:
                        if str(error)=='D153_lift_height_budget_reached':report['early_stop_reason']=str(error);break
                        raise
                    event=dict(frame=frame,before_state=before,terrain_anchor_frame=anchor,feature_spec_sha256=digest,
                               action_normalized=action.tolist(),requested_command_si=requested.tolist(),applied_command_si=applied.tolist(),
                               command_projection_reason=r.projector.reason,timing=timing,source='geometry_feedback_then_guarded_phase_bc',feedback=control,phase=phase_before,latest_allowed_phase_evidence_frame=frame-1,
                               previous_applied_action_si=previous_action,previous_drive_load=previous_load)
                    if args.frames==0:
                        report['zero_step_prediction']=event;break
                    if frame<=len(prefix_actions):
                        expected=prefix_actions[frame-1]
                        for field in ('before_state','action_normalized','requested_command_si','applied_command_si','feedback','phase','latest_allowed_phase_evidence_frame'):
                            if event[field]!=expected[field]:raise ValueError(f'startup_prefix_mismatch:{frame}:{field}')
                    manual.append_json(out/'actions.ndjson',event)
                    try:
                        row=r.step(applied)
                    finally:
                        if cache is not None and r.frames==frame:
                            try:cache.capture(frame,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                            except Exception as error:report['display_capture_error']=repr(error);raise
                    physical.append(row)
                    check_ground(frame,'after_step')
                    pose=r.assembly.assemble(row['after_state']['mechanism_qpos'],include_meshes=False)[1]['bucket']
                    phase_started=time.monotonic()
                    force=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n']
                    stage=phases.observe_end(frame=frame,state=row['after_state'],pose=pose,soil_force_n=force,
                        rock_contact_substeps=row['rock_bucket_contact_substeps'],particles=lambda:r.fixture.state.particle_q.numpy())
                    stage['observation_seconds']=time.monotonic()-phase_started
                    manual.append_json(out/'phase-trace.ndjson',stage)
                    if stage['phase']['curl_boundary_unresolved']:raise RuntimeError('phase_curl_boundary_unresolved')
                    if stage['phase']['event']:
                        report['events'].append(stage['phase']['event']);print('PHASE_EVENT',frame,stage['phase']['event']['transition'],flush=True)
                    if frame<=len(prefix_rows):
                        if row['after_state']!=prefix_rows[frame-1]['after_state']:raise ValueError(f'startup_prefix_after_state:{frame}')
                        report['prefix_frames_verified']=frame
                        if frame==len(prefix_rows):
                            with np.load(prefix_run/'final.npz') as expected:
                                exact={k:bool(np.array_equal(v,expected[k])) for k,v in r.capture().items()}
                            report['prefix_final_arrays_exact']=exact
                            if not all(exact.values()):raise ValueError('startup_prefix_final_physics')
                    previous_action=applied.tolist();previous_load=list(r.bridge.state['equivalent_drive_load'])
                    commands.append(event);latencies.append(timing['roundtrip_seconds']);positions.append(row['after_state']['bucket_origin_world_m'])
                    loads.append(dict(frame=frame,soil_samples=row['bucket_mpm_contact_samples'],rock_substeps=row['rock_bucket_contact_substeps'],hard_guard=row['hard_guard']))
                    report['frames_completed']=r.frames
                    report['final_phase']=phases.router.phase
                    elapsed=time.monotonic()-frame_started
                    frame_timings.append(dict(frame=frame,wall_seconds=elapsed))
                    manual.append_json(out/'frame-timing.ndjson',frame_timings[-1])
                    if frame%120==0:
                        r.save(out/f'periodic-{frame:05d}.npz',stage='hold')
                        save_control(out,r,feedback,phases,features,previous_action,previous_load)
                    if frame in (120,1200):
                        gate=behavior_gate(commands,physical,feedback,require_curl=frame==1200)
                        gate['checkpoint']=r.save(out/f'gate-{frame}.npz',stage='hold')
                        report['same_instance_gates'][str(frame)]=gate
                        (out/f'gate-{frame}.json').write_text(json.dumps(gate,indent=2)+'\n')
                        print('CONTINUOUS_GATE',frame,gate['passed'],flush=True)
                        if not gate['passed']:raise RuntimeError('D154_behavior_gate:'+str(gate['checks']))
                    if frame%60==0:print('PHASE_BC_FRAME',frame,phase_before,'ground',report['last_ground_check']['minimum_bucket_z_m'],flush=True)
                    if phases.router.phase=='off_soil':
                        final_gate=behavior_gate(commands,physical,feedback,require_curl=True)
                        (out/'off-soil-gate.json').write_text(json.dumps(final_gate,indent=2)+'\n')
                        if not final_gate['passed']:raise RuntimeError('D154_off_soil_behavior_gate')
                        report['early_stop_reason']='first_causal_off_soil_confirmation_not_transport_qualification';break
                report['final_phase']=phases.router.phase
                report['final_checkpoint']=r.save(out/'final.npz',stage='hold')
                save_control(out,r,feedback,phases,features,previous_action,previous_load)
                report['final_bucket_origin_world_m']=np.asarray(manual.compact_state(r.bridge.state)['bucket_origin_world_m']).tolist()
                report['bucket_origin_displacement_m']=(np.asarray(report['final_bucket_origin_world_m'])-initial_position).tolist()
                report['status']='zero_step_passed' if args.frames==0 else 'bounded_probe_completed'
        report.update(execution_summary(commands,latencies,loads))
    except BaseException as error:
        report.update(status='failed',failure=repr(error),frames_completed=0 if r is None else r.frames)
        raise
    finally:
        if cache is not None:
            try:report['display_cache']=cache.finish(report['status'])
            except Exception as error:report['display_finish_error']=repr(error)
        report['timing']=dict(completed_frame_count=len(frame_timings),loop_wall_seconds=sum(x['wall_seconds'] for x in frame_timings),runtime_step_wall_seconds=sum(x['runtime_step_wall_seconds'] for x in getattr(r,'timings',[])),simulated_seconds=len(frame_timings)/60)
        report['source_unchanged']=sha(source)==report['source']['sha256'];report['wall_seconds']=time.monotonic()-started
        report['code_sha256']={name:sha(Path(__file__).resolve().parents[2]/'src/v4'/name) for name in ('run_contact_geometric_probe.py','contact_geometric_feedback.py','contact_cut_reference.py','contact_phase_bc_inference.py','contact_phase_live.py','contact_phase_bc_policy.py','contact_bc_inference.py','contact_bc_policy.py','contact_bc_data.py','view_contact_bc_probe.py','contact_bc_ground_guard.py','contact_phase_candidate.py','contact_phase_completion.py','contact_phase_geometry.py')}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_reset','contact_summary')},indent=2))

if __name__=='__main__':main()
