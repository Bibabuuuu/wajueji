"""One authorized saved-endpoint transport, with every displayed state cached."""
import json,time
import numpy as np
from runtime import ROOT,W,SOURCE,SOURCE_SHA,TransportRuntime,read,sha
from controller import TransportController
from layered_47_cycle_geometry import TransportClearance,bucket_mask
from observation import stationary_terrain
from planning.cycle_geometry import MOUTH,tool_pose
from view_contact_bc_probe import ProbeDisplayCache
from terrain.truck_receiver import TruckReceiver,CAVITY
import run_layered_47_sparse_manual_contact_episode as manual

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    protocol=read(W/'protocol.json')
    if protocol['status']!='source_exception_authorized' or protocol['maximum_frames']!=4914 or protocol['maximum_wall_seconds']!=1395:
        raise ValueError('transport_budget_contract')
    for p,h in protocol['inputs'].items():
        if sha(p)!=h:raise ValueError('transport_input_changed:'+p)
    if read(W/'preflight-review.json')['status']!='transport_zero_step_passed':raise ValueError('transport_not_preflighted')
    if read(W/'control-review.json')['status']!='reference_commands_passed':raise ValueError('transport_commands_not_preflighted')
    if read(W/'geometry-review.json')['status']!='updated_scene_route_geometry_passed':raise ValueError('updated_scene_not_preflighted')
    if not read(W/'yaw-review.json')['passed']:raise ValueError('yaw_regression_not_passed')
    plan=read(ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/report.json')
    paths=np.load(ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/plan.npz')
    out=ROOT/'runs/contact-fixed-truck-continuation-20260925-v2/physical';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cache=None;controller=None;phase='INITIALIZE';local_index=0
    report=dict(status='initializing',source=str(SOURCE),source_sha256=SOURCE_SHA,source_frame_end=3981,
        protocol_sha256=sha(W/'protocol.json'),frames=0,stages=[],actual_handoff_executed=False,
        full_three_scoop_qualified=False,formal_fill_qualified=False)
    def update():write(out/'report.json',report)
    try:
        with TransportRuntime(SOURCE,out/'physics',freeze_passive_jp=True,authorize_steps=True) as r:
            try:
                report['zero_arrays_exact']={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()}
                if not all(report['zero_arrays_exact'].values()):raise ValueError('transport_restore_changed')
                terrain,observation=stationary_terrain(r);checker=TransportClearance(r,terrain)
                previous=np.array(read(SOURCE.parent/'control-state.json')['previous_command'])
                controller=TransportController(r,checker,previous);r.transport_controller=controller
                cache=ProbeDisplayCache(out/'display',4914,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                q=r.arrays['mechanism_qpos'];pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
                payload=bucket_mask(r.arrays['particle_q'],pose)
                receiver=TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))
                report.update(handoff_accepted=True,handoff_frame_end=3981,source_payload_particles=int(payload.sum()),
                    observed_terrain=observation,truck_enabled=True,material_history_policy=r.metadata['material_history_policy'])
                r.save(out/'handoff.npz',stage='transport');update()
                def advance(desired,velocity):
                    if r.frames>=4914 or time.monotonic()-started>=1395:raise TimeoutError('transport_budget_exhausted')
                    q=np.asarray(r.bridge.state['mechanism_qpos']);controller.check(q,phase,physical=True)
                    cmd,detail=controller.command(phase,q,r.bridge.state['controller_target'],desired,velocity)
                    manual.append_json(out/'actions.ndjson',dict(frame=3982+r.frames,local_frame=r.frames+1,stage=phase,
                        stage_step=local_index,before_state=manual.compact_state(r.bridge.state),controller_target=r.bridge.state['controller_target'],
                        desired_qpos=np.asarray(desired).tolist(),reference_velocity=np.asarray(velocity).tolist(),command=cmd.tolist(),detail=detail))
                    row=r.step(cmd);report['actual_handoff_executed']=True
                    cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy())
                    q=np.asarray(row['after_state']['mechanism_qpos']);measured=controller.check(q,phase,physical=True)
                    error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
                    manual.append_json(out/'clearance.ndjson',dict(frame=3981+r.frames,stage=phase,axis_error=error,**measured))
                    if error>.025:raise ValueError('transport_tracking_error:'+str(error))
                    report.update(frames=r.frames,stage=phase,stage_step=local_index)
                    if r.frames%120==0:print('TRANSPORT',phase,r.frames,'axis_error',error,flush=True);update()
                for segment in plan['stages'][1:]:
                    phase=segment['name'];qpath=paths[phase+'_qpos'];velocity=paths[phase+'_velocity'];goal=qpath[-1]
                    print('STAGE',phase,flush=True)
                    for local_index,(desired,v) in enumerate(zip(qpath[1:],velocity,strict=True),1):advance(desired,v)
                    for local_index in range(len(velocity)+1,len(velocity)+31):advance(goal,np.zeros(4))
                    if phase=='CLEAR_PILE':
                        arrays=r.capture();terrain,observation=stationary_terrain(r,arrays)
                        controller.checker=TransportClearance(r,terrain);controller.checker.check(np.array(r.bridge.state['mechanism_qpos']),physical=True)
                        report['transport_entry_observation']=observation
                    if phase=='DUMP':
                        for local_index in range(len(velocity)+31,len(velocity)+211):advance(goal,np.zeros(4))
                        particles=r.fixture.state.particle_q.numpy();actual=np.array(r.bridge.state['mechanism_qpos'])
                        pose=r.assembly.assemble(actual,include_meshes=False)[1]['bucket'];remaining=payload&bucket_mask(particles,pose)
                        received=payload&(receiver.classify(particles)==CAVITY)
                        report['unload_inventory']=dict(source_particles=int(payload.sum()),received_source_particles=int(received.sum()),
                            remaining_source_particles=int(remaining.sum()),scope='soil center occupancy, not stable mixed delivered volume')
                        if remaining.sum()>max(50,.1*payload.sum()) or received.sum()<.5*payload.sum():raise ValueError('transport_development_unload_inventory')
                    error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-goal[r.chain.axes])))
                    if error>.015:raise ValueError('transport_stage_endpoint:'+phase)
                    saved=r.save(out/(phase.lower()+'.npz'),stage='returned' if phase=='RETURN_HIGH' else 'unloaded' if phase=='DUMP' else 'transport')
                    report['stages'].append(dict(name=phase,frame=r.frames,axis_error=error,checkpoint=saved));update()
                report['status']='transport_unload_return_completed'
            except TimeoutError as error:report.update(status='budget_stopped',failure=str(error))
            except BaseException as error:report.update(status='failed',failure=repr(error));raise
            finally:
                report['frames']=r.frames
                try:
                    report['final_checkpoint']=r.save(out/('recovery.npz' if report['status']=='failed' else 'final.npz'),stage='failure' if report['status']=='failed' else 'returned' if report['status']=='transport_unload_return_completed' else 'transport')
                    write(out/'control-state.json',dict(global_frame_end=3981+r.frames,phase=phase,stage_step=local_index,
                        previous_command=None if controller is None else controller.previous.tolist(),
                        load_bias=None if controller is None else controller.tracking.bias.tolist(),continuation_qualified=False))
                except Exception as error:report['save_error']=repr(error)
    finally:
        if cache is not None:
            try:report['display']=cache.finish(report['status'])
            except Exception as error:report['display_error']=repr(error)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA);update()
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
