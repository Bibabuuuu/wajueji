"""D153: one bounded continuation; no prefix simulation or automatic retry."""
import json,time
from pathlib import Path
import numpy as np
from continuation_runtime import ContinuationRuntime,SOURCE,SOURCE_SHA,W,ROOT,sha
from control_history import construct,restore,encode,DYNAMIC,OUT,read
from lift_feedback import LiftFeedback
from contact_phase_bc_inference import PhaseInferenceWorker,with_phase,encode_episode
from contact_bc_inference import structured_observation
from run_layered_47_sparse_contact_handoff import wrench_summary
import run_layered_47_sparse_manual_contact_episode as manual
from view_contact_bc_probe import ProbeDisplayCache


def main():
    protocol=read(W/'protocol.json')
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('D153_protocol_input_changed:'+item['path'])
    preflight=read(W/'preflight-review.json')
    if preflight['status']!='D153_zero_restore_and_preview_passed' or preflight['policy_history_sha256']!=sha(OUT/'control-history.json'):raise ValueError('D153_preflight_not_admitted')
    out=OUT/'continuation';out.mkdir(exist_ok=False)
    report=dict(schema='D153_bounded_lift_v1',status='initializing',source=str(SOURCE),source_sha256=SOURCE_SHA,frames_completed=0,global_start_frame=2400,wall_budget_seconds=360,frame_budget=1200,events=[],policy_resume_qualified=False,formal_training_admission=False,protocol_sha256=sha(W/'protocol.json'))
    started=time.monotonic();r=None;cache=None;timings=[]
    try:
        old=read(ROOT/'runs/contact-full-response-20260924-v1/candidate60/report.json')
        with PhaseInferenceWorker('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',old['policy']['path'],old['policy']['sha256'],stderr_path=out/'inference-stderr.log') as worker:
            with ContinuationRuntime(SOURCE,out/'physics',allow_training_reset=True) as r:
                initial=r.capture();report['zero_step_arrays_exact']={k:bool(np.array_equal(v,r.arrays[k])) for k,v in initial.items()}
                if not all(report['zero_step_arrays_exact'].values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('D153_live_restore_inexact')
                feedback,phases,features,old,contract=construct(LiftFeedback,r.assembly,r.chain)
                history=restore(feedback,phases,features,OUT/'control-history.json')
                # Re-encoded serialized history must reproduce the last frozen BC action.
                arrays={k:np.stack([row[k] for row in features.rows]) for k in features.rows[-1]}
                x,_,digest=encode_episode(arrays,goal_valid=True);action,_=worker.predict(x[-1],digest)
                last=json.loads((ROOT/'runs/contact-full-response-20260924-v1/candidate60/actions.ndjson').read_text().splitlines()[-1])
                if action.tolist()!=last['action_normalized']:raise ValueError('D153_serialized_feature_prediction')
                report['serialized_history_bc_exact']=True
                previous_action=history['last_action'];previous_load=history['last_load'];goal=old['selected_goal'];terrain=None;anchor=2400
                vertices=manual._cache_rock_vertices(r.root,r.records)
                cache=ProbeDisplayCache(out/'display',1200,r.arrays['particle_q'].shape,dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                report['initial_bucket_origin_world_m']=manual.compact_state(r.bridge.state)['bucket_origin_world_m']
                def ground(frame,stage):
                    check=dict(frame=frame,stage=stage,**feedback.ground.measure(r.bridge.state['mechanism_qpos']))
                    manual.append_json(out/'ground-guard.ndjson',check)
                    if not check['safe']:raise RuntimeError('D153_bucket_ground_clearance')
                def save_control(checkpoint):
                    data=dict(schema='D153_saved_control_state_v1',checkpoint=checkpoint,last_frame=2400+r.frames,feedback=encode({k:getattr(feedback,k) for k in (*DYNAMIC,'lift_intent','lift_start_origin_z')}),phase=encode({k:v for k,v in vars(phases.router).items() if k not in ('config','contact')}),contact=encode({k:v for k,v in vars(phases.router.contact).items() if k!='config'}),features=encode(features.rows),feature_last_frame=features.last_frame,last_action=previous_action,last_load=previous_load,continuation_qualified=False)
                    (out/'control-state.json').write_text(json.dumps(data,indent=2)+'\n')
                report['startup_wall_seconds']=time.monotonic()-started
                try:
                    for local in range(1,1201):
                        if time.monotonic()-started>=359:report['stop_reason']='wall_budget';break
                        tick=time.monotonic();frame=2400+local;before=manual.compact_state(r.bridge.state);ground(frame,'before')
                        if terrain is None or (frame-1)%3==0:
                            terrain=manual.terrain_observation(r.fixture,r.bridge.state,vertices);anchor=frame-1
                            manual.append_json(out/'terrain.ndjson',dict(frame=anchor,terrain=terrain))
                        phase=phases.router.phase_for_decision(frame)
                        obs=structured_observation(frame=frame,state=before,terrain=terrain,terrain_frame=anchor,goal=goal,previous_action=previous_action,previous_load=previous_load)
                        x,digest=features.update(with_phase(obs,phase,frame-1));action,timing=worker.predict(x,digest)
                        try:requested,applied,detail=feedback.command(frame=frame,qpos=before['mechanism_qpos'],phase=phase,bc_action=action,projector=r.projector)
                        except ValueError as error:
                            if str(error)=='D153_lift_height_budget_reached':report['stop_reason']=str(error);break
                            raise
                        event=dict(frame=frame,local_frame=local,before_state=before,terrain_anchor_frame=anchor,feature_spec_sha256=digest,action_normalized=action.tolist(),requested_command_si=requested.tolist(),applied_command_si=applied.tolist(),feedback=detail,phase=phase,previous_applied_action_si=previous_action,previous_drive_load=previous_load,timing=timing)
                        manual.append_json(out/'actions.ndjson',event)
                        try:row=r.step(applied)
                        finally:
                            if r.frames==local:cache.capture(local,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                        ground(frame,'after');pose=r.assembly.assemble(row['after_state']['mechanism_qpos'],include_meshes=False)[1]['bucket']
                        force=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n']
                        stage=phases.observe_end(frame=frame,state=row['after_state'],pose=pose,soil_force_n=force,rock_contact_substeps=row['rock_bucket_contact_substeps'],particles=lambda:r.fixture.state.particle_q.numpy())
                        manual.append_json(out/'phase-trace.ndjson',stage)
                        previous_action=applied.tolist();previous_load=list(r.bridge.state['equivalent_drive_load'])
                        report.update(frames_completed=r.frames,final_phase=phases.router.phase)
                        if stage['phase']['curl_boundary_unresolved']:raise RuntimeError('phase_curl_boundary_unresolved')
                        if stage['phase']['event']:report['events'].append(stage['phase']['event']);print('EVENT',frame,stage['phase']['event'],flush=True)
                        timings.append(dict(frame=frame,wall_seconds=time.monotonic()-tick));manual.append_json(out/'frame-timing.ndjson',timings[-1])
                        if local%60==0:print('LIFT_FRAME',local,frame,phases.router.phase,'origin',row['after_state']['bucket_origin_world_m'],'seconds',time.monotonic()-started,flush=True)
                        if phases.router.phase=='off_soil':report['stop_reason']='first_causal_off_soil_confirmation';break
                    report.setdefault('stop_reason','frame_budget')
                    report['final_checkpoint']=r.save(out/'final.npz',stage='hold');save_control(report['final_checkpoint'])
                    report['final_bucket_origin_world_m']=manual.compact_state(r.bridge.state)['bucket_origin_world_m'];report['final_phase']=phases.router.phase
                    report['status']='bounded_continuation_completed'
                except BaseException:
                    recovery=r.save(out/'recovery.npz',stage='hold');save_control(recovery);report['recovery_checkpoint']=recovery
                    raise
    except BaseException as error:
        report.update(status='failed',failure=repr(error),frames_completed=0 if r is None else r.frames)
        raise
    finally:
        if cache is not None:report['display_cache']=cache.finish(report['status'])
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA,timing=dict(loop_wall_seconds=sum(x['wall_seconds'] for x in timings),simulated_seconds=len(timings)/60))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
