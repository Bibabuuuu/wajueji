"""One bounded movement from the real first return to the selected second cut."""
import json,time
import numpy as np
from runtime import ROOT,W,SOURCE,PrepointRuntime,read,sha
from controller import PrepointController,PrepointClearance
from observation import stationary_terrain
from view_contact_bc_probe import ProbeDisplayCache
import run_layered_47_sparse_manual_contact_episode as manual

def main():
    protocol=read(W/'protocol.json');preparation=read(W/'preparation-review.json');control=read(W/'control-review.json')
    if protocol['status']!='authorized_after_preflight' or protocol['maximum_frames']!=4830 or protocol['maximum_wall_seconds']!=1800:
        raise ValueError('next_prepoint_budget_contract')
    if preparation['status']!='zero_and_geometry_passed' or control['status']!='reference_commands_passed':raise ValueError('preflight_required')
    for p,h in protocol['inputs'].items():
        if sha(p)!=h:raise ValueError('prepoint_input_changed:'+p)
    paths=np.load(ROOT/'runs/contact-next-prepoint-20260925-v1/prepare/plan.npz')
    out=ROOT/'runs/contact-next-prepoint-20260925-v1/physical';out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cache=None;controller=None;index=0
    report=dict(status='initializing',source_sha256=sha(SOURCE),protocol_sha256=sha(W/'protocol.json'),frames=0,
                selected=preparation['selected'],cycle_index=2,full_three_scoop_qualified=False)
    def update():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        with PrepointRuntime(SOURCE,out/'physics',freeze_passive_jp=True,authorize_steps=True) as r:
            try:
                report['zero_arrays_exact']={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()}
                if not all(report['zero_arrays_exact'].values()):raise ValueError('prepoint_restore_changed')
                terrain,observation=stationary_terrain(r);checker=PrepointClearance(r,terrain)
                previous=np.array(r.metadata['transport_controller_state']['previous_command'])
                controller=PrepointController(r,checker,previous);r.transport_controller=controller
                cache=ProbeDisplayCache(out/'display',4830,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=preparation['selected'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                report.update(status='moving',observation=observation)
                sequence=[*zip(paths['qpos'][1:],paths['velocity'],strict=True),*[(paths['qpos'][-1],np.zeros(4))]*30]
                for index,(desired,velocity) in enumerate(sequence,1):
                    if r.frames>=4830 or time.monotonic()-started>=1800:raise TimeoutError('next_prepoint_budget')
                    q=np.array(r.bridge.state['mechanism_qpos']);controller.check(q,'PREPOINT',physical=True)
                    command,detail=controller.command('PREPOINT',q,r.bridge.state['controller_target'],desired,velocity)
                    manual.append_json(out/'actions.ndjson',dict(local_frame=index,before_state=manual.compact_state(r.bridge.state),
                        controller_target=r.bridge.state['controller_target'],desired_qpos=desired.tolist(),reference_velocity=velocity.tolist(),command=command.tolist(),detail=detail))
                    row=r.step(command);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy())
                    q=np.array(row['after_state']['mechanism_qpos']);measured=controller.check(q,'PREPOINT',physical=True)
                    error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
                    manual.append_json(out/'clearance.ndjson',dict(frame=r.frames,axis_error=error,**measured))
                    if error>.025:raise ValueError('prepoint_tracking:'+str(error))
                    if r.frames%120==0:report['frames']=r.frames;update();print('PREPOINT',r.frames,'axis_error',error,flush=True)
                if error>.015:raise ValueError('prepoint_endpoint_tracking')
                fresh,observation=stationary_terrain(r,r.capture());controller.checker=PrepointClearance(r,fresh)
                report['fresh_endpoint_clearance']=controller.check(q,'PREPOINT',physical=True)
                report.update(status='prepoint_saved_requires_independent_audit',fresh_observation=observation)
            except TimeoutError as error:report.update(status='budget_stopped',failure=str(error))
            except BaseException as error:report.update(status='failed',failure=repr(error));raise
            finally:
                report['frames']=r.frames
                try:
                    passed=report['status']=='prepoint_saved_requires_independent_audit'
                    report['checkpoint']=r.save(out/('prepoint.npz' if passed else 'recovery.npz'),stage='prepoint' if passed else 'failure' if report['status']=='failed' else 'hold',
                        cycle_index=2 if passed else 1,selected_cut=preparation['selected'] if passed else r.metadata['selected_cut'])
                    (out/'control-state.json').write_text(json.dumps(dict(stage_step=index,previous_command=None if controller is None else controller.previous.tolist(),
                        load_bias=None if controller is None else controller.tracking.bias.tolist(),continuation_qualified=False),indent=2)+'\n')
                except Exception as error:report['save_error']=repr(error)
    finally:
        if cache is not None:
            try:report['display']=cache.finish(report['status'])
            except Exception as error:report['display_error']=repr(error)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==protocol['source_sha256']);update()
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
