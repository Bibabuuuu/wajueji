"""Explicit bounded BC candidate at a saved prepoint; never a manual default."""
import argparse,json,time
from pathlib import Path
import numpy as np
from contact_bc_inference import InferenceWorker,OnlineFeatures,structured_observation,LIMITS,sha


def probe_budget(frames,contact_window):
    if not 0<=frames<=(600 if contact_window else 120):raise ValueError('bounded_bc_probe_frames')
    return 240 if contact_window else 180


def execution_summary(commands,latencies,loads):
    return dict(all_executed_frames_safe=all(v['hard_guard'] for v in loads),contact_summary=loads,
                projection_frames=sum(e['command_projection_reason'] is not None for e in commands),
                command_nonzero_frames=sum(bool(np.any(np.abs(e['applied_command_si'])>1e-8)) for e in commands),
                inference_roundtrip_seconds_quantiles=[] if not latencies else np.quantile(latencies,[0,.5,.95,1]).tolist())


def main():
    from layered_47_cycle_runtime import CycleRuntime
    import run_layered_47_sparse_manual_contact_episode as manual
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--policy',required=True);p.add_argument('--policy-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=0)
    p.add_argument('--training-python',required=True);p.add_argument('--training-reset-checkpoint',action='store_true')
    p.add_argument('--contact-window',action='store_true',help='Explicit bounded extension to at most 600 frames')
    p.add_argument('--prefix-review',type=Path,help='Audited startup prefix required for contact extension')
    p.add_argument('--capture-display',action='store_true')
    args=p.parse_args()
    wall_budget=probe_budget(args.frames,args.contact_window)
    if args.contact_window and args.prefix_review is None:p.error('contact window requires --prefix-review')
    prefix_actions=[];prefix_rows=[];prefix_run=None
    if args.prefix_review is not None:
        from view_contact_bc_probe import read_rows
        prefix=json.loads(args.prefix_review.read_text());prefix_run=Path(prefix['run'])
        if not prefix['passed'] or prefix['frames']!=120:raise ValueError('startup_prefix_not_qualified')
        for item in prefix['inputs']+prefix['outputs']:
            if sha(item['path'])!=item['sha256']:raise ValueError('startup_prefix_changed')
        if sha(args.source)!=prefix['inputs'][0]['sha256'] or args.policy_sha256!=prefix['inputs'][1]['sha256']:
            raise ValueError('startup_prefix_source_or_policy')
        prefix_actions=read_rows(prefix_run/'actions.ndjson');prefix_rows=read_rows(prefix_run/'physics/frames.ndjson')
    source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='contact_bc_closed_loop_probe_v1',status='initializing',frames_requested=args.frames,frames_completed=0,
                source=dict(path=str(source),sha256=sha(source)),policy=dict(path=str(Path(args.policy).resolve()),sha256=args.policy_sha256),
                no_scripted_excavation=True,manual_default_changed=False,policy_resume_qualified=False,formal_training_admission=False)
    report.update(contact_window=args.contact_window,wall_budget_seconds=wall_budget,prefix_frames_verified=0)
    if args.prefix_review is not None:
        report['prefix_review']=dict(path=str(args.prefix_review.resolve()),sha256=sha(args.prefix_review))
    started=time.monotonic();commands=[];latencies=[];positions=[];loads=[];r=None;cache=None
    try:
        with InferenceWorker(args.training_python,args.policy,args.policy_sha256,stderr_path=out/'inference-stderr.log') as worker:
            report['worker_ready']=worker.ready
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
                vertices=manual._cache_rock_vertices(r.root,r.records);features=OnlineFeatures();terrain=None;anchor=0;previous_action=[0.]*4;previous_load=None
                initial_position=np.asarray(manual.compact_state(r.bridge.state)['bucket_origin_world_m']);report['initial_bucket_origin_world_m']=initial_position.tolist()
                if args.capture_display:
                    from view_contact_bc_probe import ProbeDisplayCache
                    cache=ProbeDisplayCache(out/'display',args.frames,r.arrays['particle_q'].shape,
                        dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                for frame in range(1,max(1,args.frames)+1):
                    if time.monotonic()-started>wall_budget:raise TimeoutError('bc_probe_wall_budget')
                    before=manual.compact_state(r.bridge.state)
                    check_ground(frame,'before_step')
                    if terrain is None or (frame-1)%3==0:
                        terrain=manual.terrain_observation(r.fixture,r.bridge.state,vertices);anchor=frame-1
                        manual.append_json(out/'terrain.ndjson',dict(frame=anchor,terrain=terrain))
                    structured=structured_observation(frame=frame,state=before,terrain=terrain,terrain_frame=anchor,goal=goal,previous_action=previous_action,previous_load=previous_load)
                    x,digest=features.update(structured);action,timing=worker.predict(x,digest)
                    requested=action*LIMITS;applied=r.projector.project(before['mechanism_qpos'],requested)
                    event=dict(frame=frame,before_state=before,terrain_anchor_frame=anchor,feature_spec_sha256=digest,
                               action_normalized=action.tolist(),requested_command_si=requested.tolist(),applied_command_si=applied.tolist(),
                               command_projection_reason=r.projector.reason,timing=timing,source='native60_bc_candidate',
                               previous_applied_action_si=previous_action,previous_drive_load=previous_load)
                    if args.frames==0:
                        report['zero_step_prediction']=event;break
                    if frame<=len(prefix_actions):
                        expected=prefix_actions[frame-1]
                        for field in ('before_state','action_normalized','requested_command_si','applied_command_si'):
                            if event[field]!=expected[field]:raise ValueError(f'startup_prefix_mismatch:{frame}:{field}')
                    manual.append_json(out/'actions.ndjson',event)
                    try:
                        row=r.step(applied)
                    finally:
                        if cache is not None and r.frames==frame:
                            try:cache.capture(frame,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                            except Exception as error:report['display_capture_error']=repr(error);raise
                    check_ground(frame,'after_step')
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
                report['final_checkpoint']=r.save(out/'final.npz',stage='hold')
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
        report['source_unchanged']=sha(source)==report['source']['sha256'];report['wall_seconds']=time.monotonic()-started
        report['code_sha256']={name:sha(Path(__file__).with_name(name)) for name in ('run_contact_bc_probe.py','contact_bc_inference.py','contact_bc_policy.py','contact_bc_data.py','view_contact_bc_probe.py','contact_bc_ground_guard.py')}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_reset','contact_summary')},indent=2))

if __name__=='__main__':main()
