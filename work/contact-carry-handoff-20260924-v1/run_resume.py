"""One bounded first-endpoint continuation, only with an approved protocol."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from manual_key4_review import ROOT,W,sha
from restore_runtime import HandoffRuntime,SOURCE,SOURCE_SHA
from carry_feedback import CarryFeedback
from contact_transport_handoff import CarryHandoffGate
from contact_bc_ground_guard import BucketGroundGuard
from control_history import encode
from contact_phase_geometry import SolidBucketGap
from view_contact_bc_probe import ProbeDisplayCache
import run_layered_47_sparse_manual_contact_episode as manual

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();protocol=read(args.protocol)
    if (protocol['schema']!='D155_handoff_resume_protocol_v1'
            or protocol['status']!='authorized_single_continuation'
            or protocol['maximum_frames']!=600 or protocol['maximum_wall_seconds']!=240
            or protocol['source_sha256']!=SOURCE_SHA):
        raise ValueError('D155_physics_budget_not_authorized')
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('D155_protocol_input_changed:'+item['path'])
    preflight=read(W/'preflight-review.json')
    if preflight['status']!='D155_preflight_passed':raise ValueError('D155_preflight_not_passed')
    cached=read(W/'cache-review.json')['episodes'][0]
    gate=CarryHandoffGate.restore(cached['gate_snapshot'])
    if gate.last_frame!=2327:raise ValueError('D155_gate_history_not_causal')
    source_meta=read(str(SOURCE)+'.json');previous=np.asarray(source_meta['controller_state']['controller_target_velocity'])
    out=Path(protocol['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();cache=None;report=dict(status='initializing',source_sha256=SOURCE_SHA,
        source_frame_end=2327,frames_completed=0,handoff=None,physics_steps_added=0,
        protocol_path=str(args.protocol.resolve()),protocol_sha256=sha(args.protocol),transport_unload_tested=False)
    try:
        with HandoffRuntime(SOURCE,out/'physics') as runtime:
            feedback=None
            try:
                initial=runtime.capture();exact={key:bool(np.array_equal(value,runtime.arrays[key])) for key,value in initial.items()}
                report['zero_step_arrays_exact']=exact
                if not all(exact.values()) or max(runtime.restored['cycle_errors'].values())>1e-12:
                    raise ValueError('D155_resume_restore_not_exact')
                q=np.asarray(runtime.bridge.state['mechanism_qpos']);ground=BucketGroundGuard(runtime.assembly,q)
                feedback=CarryFeedback(runtime.assembly,runtime.chain,ground,q,previous,
                    frame_end=2327,initial_soil_top_m=gate.initial_soil_top_m)
                geometry=SolidBucketGap(runtime.proxy)
                cache=ProbeDisplayCache(out/'display',600,runtime.arrays['particle_q'].shape,
                    dict(records=runtime.records,root=str(runtime.root),selected_cut=runtime.metadata['selected_cut'],
                         particle_spacing_m=runtime.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(runtime.bridge.state),runtime.fixture.state.particle_q.numpy())
                for local_frame in range(1,601):
                    if time.monotonic()-started>=240:
                        report['stop_reason']='wall_budget';break
                    frame=2327+local_frame;q=np.asarray(runtime.bridge.state['mechanism_qpos'])
                    if not ground.measure(q)['safe']:raise ValueError('D155_before_ground_guard')
                    command,detail=feedback.command(frame=frame,qpos=q,projector=runtime.projector)
                    manual.append_json(out/'actions.ndjson',dict(frame=frame,source='D155_finish_curl_then_carry',
                        before_state=manual.compact_state(runtime.bridge.state),applied_command_si=command.tolist(),control=detail))
                    row=runtime.step(command);state=row['after_state'];q=np.asarray(state['mechanism_qpos'])
                    cache.capture(local_frame,state,runtime.fixture.state.particle_q.numpy())
                    measured=ground.measure(q)
                    if not measured['safe']:raise ValueError('D155_after_ground_guard')
                    result=gate.observe_end(frame=frame,closure_fraction=(q[15]-.37)/(.852-.37),
                        origin_z_m=state['bucket_origin_world_m'][2],minimum_bucket_z_m=measured['minimum_bucket_z_m'],contact_established=True)
                    manual.append_json(out/'handoff.ndjson',result)
                    report.update(frames_completed=local_frame,physics_steps_added=local_frame,last_gate=result)
                    if result['ready'] and feedback.handoff_frame is None:
                        feedback.accept_handoff(readiness=result,qpos=q);report['handoff']=result
                        # Expensive soil gap is a diagnostic, never a gate.
                        pose=runtime.assembly.assemble(q,include_meshes=False)[1]['bucket']
                        try:report['handoff_gap_diagnostic']=geometry.observe(runtime.fixture.state.particle_q.numpy(),pose)
                        except ValueError as error:report['handoff_gap_diagnostic_error']=str(error)
                        report['handoff_checkpoint']=runtime.save(out/'handoff.npz',stage='hold')
                        print('HANDOFF',frame,flush=True)
                    if local_frame%60==0:print('RESUME',local_frame,'closure',result['closure_held'],'rise',result['sustained_actual_rise'],flush=True)
                    if feedback.handoff_frame is not None and frame>=feedback.handoff_frame+60:
                        report['stop_reason']='handoff_plus_60_frames';break
                report.setdefault('stop_reason','frame_budget')
                report['status']='bounded_continuation_completed'
            except BaseException as error:
                report.update(status='failed',failure=repr(error));raise
            finally:
                # Save before report assembly and before closing the runtime.
                report['frames_completed']=runtime.frames;report['physics_steps_added']=runtime.frames
                try:
                    report['final_checkpoint']=runtime.save(out/('final.npz' if report['status']!='failed' else 'recovery.npz'),stage='hold' if report['status']!='failed' else 'failure')
                    write(out/'control-state.json',dict(schema='D155_carry_controller_state_v1',
                        global_physical_frame=2327+runtime.frames,gate=gate.snapshot(),
                        controller=None if feedback is None else encode({key:getattr(feedback,key) for key in
                            ('previous_command','last_frame','first_command','carry_pitch','carry_cylinder_m','stage','handoff_frame','initial_soil_top_m')}),
                        continuation_qualified=False))
                except Exception as error:report['recovery_save_error']=repr(error)
    finally:
        if cache is not None:
            try:report['display']=cache.finish(report['status'])
            except Exception as error:report['display_save_error']=repr(error)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA)
        write(out/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
