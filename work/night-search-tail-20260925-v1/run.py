"""Finish curl, lift, swing and dump without independent inter-stage audits."""
import json,time,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from runtime import ROOT,W,SOURCE,FinishRuntime,read,sha,PROTOCOL,TRIAL_ROOT
from controller import CarryFeedback,TransportController
from observation import stationary_terrain
from plan import build_plan
from contact_transport_handoff import CarryHandoffGate,CarryHandoffConfig
from contact_bc_ground_guard import BucketGroundGuard
from layered_47_cycle_geometry import TransportClearance,bucket_mask
from terrain.truck_receiver import TruckReceiver,CAVITY
from view_contact_bc_probe import ProbeDisplayCache
import run_layered_47_sparse_manual_contact_episode as manual

def main():
    evaluation_mode=os.environ.get('NIGHT_EVALUATION_MODE','dig_only')
    if evaluation_mode not in ('dig_only','unload_validation'):raise ValueError('invalid_night_evaluation_mode')
    protocol=read(PROTOCOL)
    if protocol.get('evaluation_mode')!=evaluation_mode:raise ValueError('night_evaluation_mode_contract')
    if protocol['status']!='authorized_continuous_run' or protocol['maximum_frames']!=6000 or protocol['maximum_wall_seconds']!=2000:
        raise ValueError('second_tail_budget_contract')
    for p,h in protocol['inputs'].items():
        if sha(p)!=h:raise ValueError('second_tail_input_changed:'+p)
    out=Path(os.environ['NIGHT_TAIL_OUTPUT']).resolve()
    if not out.is_relative_to(TRIAL_ROOT):raise ValueError('night_tail_output_outside_trial_root')
    out.mkdir(parents=True,exist_ok=False)
    source_report=read(SOURCE.parent/'report.json');source_control=read(SOURCE.parent/'control-state.json')
    source_n=source_report['frames_completed'];started=time.monotonic();cache=None;carry=None;transport=None;gate=None;phase='INITIALIZE';stage_step=0;peak_center=None
    report=dict(status='initializing',source_sha256=sha(SOURCE),source_frame_end=source_n,frames=0,stages=[],
                independent_review_deferred=True,evaluation_mode=evaluation_mode,formal_three_scoop_qualified=False,formal_fill_qualified=False)
    def write():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        with FinishRuntime(SOURCE,out/'physics') as r:
            def state():
                d=dict(phase=phase,stage_step=stage_step,peak_bucket_center_z_m=peak_center,frame=source_n+r.frames,gate=None if gate is None else gate.snapshot())
                if carry is not None:d['carry']={k:manual._clean(getattr(carry,k)) for k in ('previous_command','last_frame','first_command','carry_pitch','carry_cylinder_m','stage','handoff_frame','initial_soil_top_m','hold_latched')}
                if transport is not None:d['transport']=dict(previous_command=transport.previous.tolist(),load_bias=transport.tracking.bias.tolist(),first=transport.first,
                    minimum_cylinder=transport.minimum_cylinder,source_pitch=transport.pitch,source_cylinder=transport.cylinder)
                r.highlevel_state=d;return d
            def save(name,stage='hold'):
                state();return r.save(out/name,stage=stage)
            def budget():
                if r.frames>=6000 or time.monotonic()-started>=2000:raise TimeoutError('second_tail_budget_exhausted')
            def capture(row):
                nonlocal peak_center
                cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy());report['frames']=r.frames
                center=float(np.mean(carry.body_heights(np.array(row['after_state']['mechanism_qpos']))))
                if phase=='FINISH_CURL':
                    peak_center=max(peak_center,center)
                    if peak_center-center>.010+1e-9:raise ValueError('loaded_center_cumulative_drawdown_exceeds_10mm')
            def periodic():
                if r.frames%120==0:report['periodic_checkpoint']=save('periodic-%05d.npz'%r.frames);write()
            try:
                exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()};report['zero_arrays_exact']=exact
                if not all(exact.values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('second_tail_restore_changed')
                if source_report['final_phase'] not in ('curl_load','lift','off_soil'):raise ValueError('second_contact_not_completed')
                q=np.array(r.bridge.state['mechanism_qpos']);ground=BucketGroundGuard(r.assembly,q)
                top=float(r.metadata['selected_cut']['contact']['crest_surface_z_m'])
                gate=CarryHandoffGate(initial_soil_top_m=top,source_sha256=sha(SOURCE),config=CarryHandoffConfig(trend_frames=30,maximum_drawdown_m=.010))
                history=[json.loads(s) for s in (SOURCE.parent/'physics/frames.ndjson').read_text().splitlines()]
                if len(history)!=source_n or not np.array_equal(history[-1]['after_state']['mechanism_qpos'],q):raise ValueError('second_source_history_endpoint')
                for entry in history[-181:]:
                    frame=entry['frame'];s=entry['after_state'];pq=np.array(s['mechanism_qpos'])
                    readiness=gate.observe_end(frame=frame,closure_fraction=(pq[15]-.37)/(.852-.37),origin_z_m=s['bucket_origin_world_m'][2],
                        minimum_bucket_z_m=ground.measure(pq)['minimum_bucket_z_m'],contact_established=True)
                previous=np.array(source_control['last_action'])
                carry=CarryFeedback(r.assembly,r.chain,ground,q,previous,frame_end=source_n,initial_soil_top_m=top)
                peak_center=float(np.mean(carry.body_heights(q)))
                cache=ProbeDisplayCache(out/'display',6000,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                phase='FINISH_CURL';report['status']='running';write()
                for stage_step in range(1,601):
                    if readiness['ready']:break
                    budget();q=np.array(r.bridge.state['mechanism_qpos'])
                    if not ground.measure(q)['safe']:raise ValueError('carry_before_ground')
                    carry.loaded_center_peak=peak_center
                    cmd,detail=carry.command(frame=source_n+r.frames+1,qpos=q,projector=r.projector,native_target=r.bridge.state['controller_target'],native_velocity=r.bridge.state['controller_target_velocity'])
                    manual.append_json(out/'actions.ndjson',dict(frame=source_n+r.frames+1,stage=phase,stage_step=stage_step,
                        before_state=manual.compact_state(r.bridge.state),command=cmd.tolist(),detail=detail))
                    row=r.step(cmd);capture(row);q=np.array(row['after_state']['mechanism_qpos']);measured=ground.measure(q)
                    if not measured['safe']:raise ValueError('carry_after_ground')
                    readiness=gate.observe_end(frame=source_n+r.frames,closure_fraction=(q[15]-.37)/(.852-.37),
                        origin_z_m=row['after_state']['bucket_origin_world_m'][2],minimum_bucket_z_m=measured['minimum_bucket_z_m'],contact_established=True)
                    manual.append_json(out/'handoff.ndjson',readiness)
                    periodic()
                if not readiness['ready']:raise ValueError('second_handoff_not_ready_within_finish_window')
                carry.accept_handoff(readiness=readiness,qpos=q);report['handoff']=readiness
                report['handoff_checkpoint']=save('handoff.npz');print('SECOND_HANDOFF',source_n+r.frames,flush=True)
                # Current live arrays feed the next route; native state is never assigned.
                view=SimpleNamespace(**vars(r));view.arrays=r.capture();view.metadata={**r.metadata,'controller_state':dict(controller_target=list(r.bridge.state['controller_target']))}
                phase='PLAN_TRANSPORT';stage_step=0;peak_center=None
                if hasattr(carry,'loaded_center_peak'):del carry.loaded_center_peak
                segments,checker,plan_report=build_plan(view,out/'plan');report['plan']=plan_report
                transport=TransportController(view,checker,carry.previous_command)
                pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket'];payload=bucket_mask(view.arrays['particle_q'],pose)
                receiver=TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))
                report['source_payload_particles']=int(payload.sum())
                def advance(desired,velocity):
                    budget();q=np.array(r.bridge.state['mechanism_qpos']);transport.check(q,phase,physical=True)
                    cmd,detail=transport.command(phase,q,r.bridge.state['controller_target'],desired,velocity)
                    manual.append_json(out/'actions.ndjson',dict(frame=source_n+r.frames+1,stage=phase,stage_step=stage_step,
                        before_state=manual.compact_state(r.bridge.state),controller_target=r.bridge.state['controller_target'],desired_qpos=desired.tolist(),
                        reference_velocity=velocity.tolist(),command=cmd.tolist(),detail=detail))
                    row=r.step(cmd);capture(row);q=np.array(row['after_state']['mechanism_qpos']);measured=transport.check(q,phase,physical=True)
                    error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
                    manual.append_json(out/'clearance.ndjson',dict(frame=source_n+r.frames,stage=phase,axis_error=error,**measured))
                    if error>.025:raise ValueError('second_transport_tracking:'+str(error))
                    periodic()
                for segment in segments:
                    phase=segment['name'];path=segment['path'];goal=segment['goal'];print('SECOND_STAGE',phase,flush=True)
                    for stage_step,(desired,velocity) in enumerate(zip(path['qpos'][1:],path['velocity'],strict=True),1):advance(desired,velocity)
                    for stage_step in range(len(path['velocity'])+1,len(path['velocity'])+31):advance(goal,np.zeros(4))
                    if phase=='CLEAR_PILE':
                        terrain,observation=stationary_terrain(r,r.capture());transport.checker=TransportClearance(view,terrain)
                        transport.check(np.array(r.bridge.state['mechanism_qpos']),'ALIGN',physical=True)
                        report['transport_entry_observation']=observation
                    if phase=='DUMP':
                        for stage_step in range(len(path['velocity'])+31,len(path['velocity'])+211):advance(goal,np.zeros(4))
                        particles=r.fixture.state.particle_q.numpy();q=np.array(r.bridge.state['mechanism_qpos']);pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
                        remaining=payload&bucket_mask(particles,pose);received=payload&(receiver.classify(particles)==CAVITY)
                        report['unload_inventory']=dict(source_particles=int(payload.sum()),received_source_particles=int(received.sum()),remaining_source_particles=int(remaining.sum()),scope='development soil centers; stable mixed volume not qualified')
                        if remaining.sum()>max(50,.1*payload.sum()) or received.sum()<.5*payload.sum():raise ValueError('second_development_unload_inventory')
                    error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-goal[r.chain.axes])))
                    if error>.015:raise ValueError('second_stage_endpoint:'+phase)
                    report['stages'].append(dict(name=phase,frame=r.frames,checkpoint=save(phase.lower()+'.npz','unloaded' if phase=='DUMP' else 'transport')));write()
                report['status']='dig_lift_completed' if evaluation_mode=='dig_only' else 'second_scoop_unload_completed'
            except TimeoutError as error:report.update(status='budget_stopped',failure=str(error))
            except BaseException as error:report.update(status='failed',failure=repr(error));raise
            finally:
                report['frames']=r.frames
                try:
                    report['final_checkpoint']=save('recovery.npz' if report['status']=='failed' else 'final.npz','failure' if report['status']=='failed' else 'unloaded' if report['status']=='second_scoop_unload_completed' else 'transport' if report['status']=='dig_lift_completed' else 'hold')
                    (out/'control-state.json').write_text(json.dumps(state(),indent=2)+'\n')
                except Exception as error:report['save_error']=repr(error)
    finally:
        if cache is not None:
            try:report['display']=cache.finish(report['status'])
            except Exception as error:report['display_error']=repr(error)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==protocol['source_sha256']);write()
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
