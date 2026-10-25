"""One fixed-route continuation, preserving every state until dump or protection."""
import json,time
import numpy as np
from runtime import ROOT,W,RUN,SOURCE,SOURCE_SHA,SOURCE_FRAME,MAX_FRAMES,MAX_SECONDS,TransportRuntime,read,sha
from controller import TransportController
from clearance import FixedRouteClearance
from layered_47_cycle_geometry import bucket_mask
from view_contact_bc_probe import ProbeDisplayCache
from terrain.truck_receiver import TruckReceiver,CAVITY
import run_layered_47_sparse_manual_contact_episode as manual

def main():
    protocol=read(W/'protocol.json');preflight=read(W/'preflight-review.json')
    if protocol['status']!='authorized_fixed_route_continuation' or protocol['maximum_frames']!=MAX_FRAMES or protocol['maximum_wall_seconds']!=MAX_SECONDS:raise ValueError('fixed_route_budget_contract')
    for p,h in protocol['inputs'].items():
        if sha(p)!=h:raise ValueError('fixed_route_input_changed:'+p)
    if preflight['status']!='fixed_route_preflight_passed' or sha(RUN/'plan/plan.npz')!=preflight['plan_sha256']:raise ValueError('fixed_route_not_preflighted')
    out=RUN/'physical';out.mkdir(parents=True,exist_ok=False);paths=np.load(RUN/'plan/plan.npz');plan=read(RUN/'plan/report.json')
    started=time.monotonic();cache=None;control=None;phase='INITIALIZE';stage_step=0
    report=dict(status='initializing',source=str(SOURCE),source_sha256=SOURCE_SHA,source_frame_end=SOURCE_FRAME,
        frames=0,stages=[],protocol_sha256=sha(W/'protocol.json'),formal_three_scoop_qualified=False,formal_fill_qualified=False,
        particle_terrain_check_enabled=False,physical_particles_modified=False)
    def write():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        with TransportRuntime(SOURCE,out/'physics',authorize_steps=True) as r:
            def state():
                d=dict(frame=SOURCE_FRAME+r.frames,phase=phase,stage_step=stage_step)
                if control is not None:d['transport']=dict(previous_command=control.previous.tolist(),load_bias=control.tracking.bias.tolist(),
                    first=control.first,minimum_cylinder=control.minimum_cylinder,source_pitch=control.pitch,source_cylinder=control.cylinder)
                r.highlevel_state=d;return d
            def save(name,stage='transport'):state();return r.save(out/name,stage=stage)
            try:
                report['zero_arrays_exact']={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()}
                if not all(report['zero_arrays_exact'].values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('fixed_route_restore_changed')
                checker=FixedRouteClearance(r);previous=np.array(r.metadata['highlevel_state']['carry']['previous_command'])
                control=TransportController(r,checker,previous)
                cache=ProbeDisplayCache(out/'display',MAX_FRAMES,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                pose=r.assembly.assemble(r.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket'];payload=bucket_mask(r.arrays['particle_q'],pose)
                receiver=TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))
                report.update(status='running',source_payload_particles=int(payload.sum()),handoff=read(SOURCE.parent/'report.json')['handoff'])
                report['entry_checkpoint']=save('entry.npz');write()
                def advance(desired,velocity):
                    if r.frames>=MAX_FRAMES or time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('fixed_route_budget_exhausted')
                    q=np.array(r.bridge.state['mechanism_qpos']);control.check(q,phase,physical=True)
                    cmd,detail=control.command(phase,q,r.bridge.state['controller_target'],desired,velocity)
                    manual.append_json(out/'actions.ndjson',dict(frame=SOURCE_FRAME+r.frames+1,stage=phase,stage_step=stage_step,
                        before_state=manual.compact_state(r.bridge.state),controller_target=r.bridge.state['controller_target'],
                        desired_qpos=desired.tolist(),reference_velocity=velocity.tolist(),command=cmd.tolist(),detail=detail))
                    row=r.step(cmd);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy())
                    report.update(frames=r.frames,stage=phase,stage_step=stage_step)
                    q=np.array(row['after_state']['mechanism_qpos']);measured=control.check(q,phase,physical=True)
                    error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
                    manual.append_json(out/'clearance.ndjson',dict(frame=SOURCE_FRAME+r.frames,stage=phase,axis_error=error,**measured))
                    if error>.025:raise ValueError('fixed_route_tracking_error:'+str(error))
                    if r.frames%120==0:report['periodic_checkpoint']=save('periodic-%05d.npz'%r.frames);write()
                for segment in plan['stages']:
                    phase=segment['name'];qs=paths[phase+'_qpos'];vs=paths[phase+'_velocity'];goal=qs[-1]
                    print('FIXED_ROUTE_STAGE',phase,flush=True)
                    for stage_step,(desired,velocity) in enumerate(zip(qs[1:],vs,strict=True),1):advance(desired,velocity)
                    for stage_step in range(len(vs)+1,len(vs)+31):advance(goal,np.zeros(4))
                    if phase=='CLEAR_PILE':control.check(np.array(r.bridge.state['mechanism_qpos']),'ALIGN',physical=True)
                    if phase=='DUMP':
                        for stage_step in range(len(vs)+31,len(vs)+211):advance(goal,np.zeros(4))
                        particles=r.fixture.state.particle_q.numpy();pose=r.assembly.assemble(np.array(r.bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
                        remaining=payload&bucket_mask(particles,pose);received=payload&(receiver.classify(particles)==CAVITY)
                        report['unload_inventory']=dict(source_particles=int(payload.sum()),received_source_particles=int(received.sum()),
                            remaining_source_particles=int(remaining.sum()),scope='development soil centers; not stable mixed volume')
                        if remaining.sum()>max(50,.1*payload.sum()) or received.sum()<.5*payload.sum():raise ValueError('fixed_route_development_unload_inventory')
                    error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-goal[r.chain.axes])))
                    if error>.015:raise ValueError('fixed_route_stage_endpoint:'+phase)
                    report['stages'].append(dict(name=phase,frame=r.frames,checkpoint=save(phase.lower()+'.npz','unloaded' if phase=='DUMP' else 'transport')));write()
                report['status']='second_scoop_unload_completed'
            except TimeoutError as error:report.update(status='budget_stopped',failure=str(error))
            except BaseException as error:report.update(status='failed',failure=repr(error));raise
            finally:
                report['frames']=r.frames
                try:
                    report['final_checkpoint']=save('recovery.npz' if report['status']=='failed' else 'final.npz','failure' if report['status']=='failed' else 'unloaded' if report['status']=='second_scoop_unload_completed' else 'transport')
                    (out/'control-state.json').write_text(json.dumps(state(),indent=2)+'\n')
                except Exception as error:report['save_error']=repr(error)
    finally:
        if cache is not None:
            try:report['display']=cache.finish(report['status'])
            except Exception as error:report['display_error']=repr(error)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA);write()
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
