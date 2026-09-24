"""Read-only D155 continuation audit; never advances MPM or native physics."""
import json
from pathlib import Path
import numpy as np
from preflight import read, rows, ROOT, W, SOURCE, SOURCE_SHA, sha, encode
from carry_feedback import CarryFeedback, LIMITS
from contact_transport_handoff import CarryHandoffGate
from contact_bc_ground_guard import BucketGroundGuard
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from audit import matrix_power_response

RUN = ROOT/'runs/contact-carry-handoff-20260924-v1/first-resume'

def main():
    report = read(RUN/'report.json')
    events = rows(RUN/'actions.ndjson')
    physical = rows(RUN/'physics/frames.ndjson')
    stages = rows(RUN/'handoff.ndjson')
    halves = rows(RUN/'physics/native-halves.ndjson')
    responses = rows(RUN/'physics/response.ndjson')
    timing = rows(RUN/'physics/timing.ndjson')
    n = len(physical)
    endpoint = RUN/('recovery.npz' if report['status']=='failed' else 'final.npz')
    meta, source_meta = read(str(endpoint)+'.json'), read(str(SOURCE)+'.json')
    initial, final = np.load(SOURCE), np.load(endpoint)
    gate = CarryHandoffGate.restore(read(W/'cache-review.json')['episodes'][0]['gate_snapshot'])
    history = list(gate.rows)
    k = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
    assembly, chain = _load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
    ground = BucketGroundGuard(assembly, initial['mechanism_qpos'])
    previous = np.array(source_meta['controller_state']['controller_target_velocity'])
    feedback = CarryFeedback(assembly,chain,ground,initial['mechanism_qpos'],previous,
                            frame_end=2327,initial_soil_top_m=gate.initial_soil_top_m)
    projector = ContactCommandProjector(chain)
    error = 0.; detail_exact = True; gate_exact = True; ground_safe = True
    first_ready = None; actual_retained = True; before_exact = True
    for i, (event,row,stage) in enumerate(zip(events[:n],physical,stages,strict=True)):
        q_before = initial['mechanism_qpos'] if i==0 else physical[i-1]['after_state']['mechanism_qpos']
        before_exact &= np.array_equal(q_before,event['before_state']['mechanism_qpos'])
        command,detail = feedback.command(frame=event['frame'],qpos=q_before,projector=projector)
        error = max(error,float(np.max(abs(command-event['applied_command_si']))))
        detail_exact &= detail==event['control']
        state = row['after_state'];q=np.array(state['mechanism_qpos'])
        measured = ground.measure(q);ground_safe &= measured['safe']
        actual_retained &= q[15]>=feedback.carry_cylinder_m-.001 and feedback.coordinates(q)[3]>=feedback.carry_pitch-.01
        result = gate.observe_end(frame=event['frame'],closure_fraction=(q[15]-.37)/.482,
            origin_z_m=state['bucket_origin_world_m'][2],minimum_bucket_z_m=measured['minimum_bucket_z_m'],contact_established=True)
        gate_exact &= result==stage
        history.append(dict(frame=event['frame'],closure_fraction=(q[15]-.37)/.482,
            origin_z_m=state['bucket_origin_world_m'][2],minimum_bucket_z_m=measured['minimum_bucket_z_m']))
        if result['ready'] and first_ready is None:
            first_ready=result;feedback.accept_handoff(readiness=result,qpos=q)
    commands = np.vstack([previous,*[e['applied_command_si'] for e in events[:n]]])
    slew = float(np.max(abs(np.diff(commands,axis=0)/LIMITS)))
    matrix_error = max(float(np.max(abs(matrix_power_response(r['coefficients'])-r['response_world']))) for r in responses)
    saved = read(RUN/'control-state.json')
    attrs = ('previous_command','last_frame','first_command','carry_pitch','carry_cylinder_m','stage','handoff_frame','initial_soil_top_m')
    window = [] if first_ready is None else [r for r in history if first_ready['frame_end']-180<=r['frame']<=first_ready['frame_end']]
    independent_gate = False
    if window:
        z=np.array([r['origin_z_m'] for r in window]);b=np.array([r['minimum_bucket_z_m'] for r in window])
        independent_gate = (len(window)==181 and all(r['closure_fraction']>=.99 for r in window)
            and np.all(z[12:]-z[:-12]>.0001) and np.max(np.maximum.accumulate(z)-z)<=.001
            and np.max(np.maximum.accumulate(b)-b)<=.001)
    checks = dict(exact_restore=all(report['zero_step_arrays_exact'].values()),
        frames=n==report['frames_completed']==len(stages),action_frames=all(e['frame']==2328+i for i,e in enumerate(events[:n])),
        before_state_contiguous=before_exact,first_command_exact=np.array_equal(commands[0],commands[1]),
        commands_recomputed=error<1e-12 and detail_exact,commands_match_physics=all(e['applied_command_si']==r['applied_command_si'] for e,r in zip(events[:n],physical)),
        slew=slew<=2/60+1e-9,no_open_command=bool(np.all(commands[1:,3]>=-1e-10)),
        actual_retention=actual_retained,gate_recomputed=gate_exact,gate_saved=gate.snapshot()==saved['gate'],
        saved_controller=encode({k:getattr(feedback,k) for k in attrs})==saved['controller'],
        handoff_report=first_ready==report['handoff'],ground=ground_safe,physical=all(r['hard_guard'] for r in physical),
        halves=len(halves)==2*n,half_order=all(h['frame']==i//2+1 and h['half']==i%2+1 for i,h in enumerate(halves)),
        native_clock=np.max(abs(np.array([h['time_s'] for h in halves])-source_meta['controller_state']['native_time_s']-np.arange(1,2*n+1)/120))<1e-8,
        native_guards=all(h['closure_m']<=.001 and h['forbidden_contact_count']==0 for h in halves),
        dispatch=all(h['dispatch_exact'] for h in halves),one_real_dispatch=len(timing)==n and all(t['real_step_calls_this_frame']==1 for t in timing),
        responses=len(responses)==n,operator_once=all(r['mpm_operator_calls']==1 for r in responses),
        shadow_unchanged=all(r['live_state_unchanged'] and r['pending_unchanged_before_real_step'] for r in responses),
        response_matrix=matrix_error<1e-10,source_unchanged=sha(SOURCE)==SOURCE_SHA,
        inputs=all(sha(x['path'])==x['sha256'] for x in read(W/'resume-protocol.json')['inputs']),
        bindings=all(sha(p)==v for field in ('handoff_bindings','handoff_gate_binding','candidate_bindings') for p,v in meta[field].items()),
        runtime=all(sha(ROOT/'src/v4'/p)==v for p,v in meta['runtime_bindings'].items()),
        protocol_hash=sha(W/'resume-protocol.json')==report['protocol_sha256'],
        lineage=all(meta[k]==source_meta[k] for k in ('lineage_root','training_reset','cycle_index')),
        schema=meta['schema']=='layered_47_full_response_carry_handoff_candidate_v1',
        simulation_clock=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-n/60)<1e-8,
        final_state=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),
        finite=all(np.isfinite(final[k]).all() for k in final.files if final[k].dtype.kind in 'fciu'),
        particles=initial['particle_q'].shape==final['particle_q'].shape,rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
        frame_budget=n<=600)
    from layered_47_cycle_checkpoint import load
    try:load(endpoint);checks['ordinary_loader_rejects']=False
    except ValueError as e:checks['ordinary_loader_rejects']=str(e)=='cycle_checkpoint_schema'
    v=np.array([h['mechanism_qvel'] for h in halves])[:,[6,8,11,14]]
    result=dict(audit_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        status=report['status'],failure=report.get('failure'),stop_reason=report.get('stop_reason'),
        frames=n,wall_seconds=report['wall_seconds'],command_max_error=error,matrix_max_error=matrix_error,
        max_normalized_command_step=slew,first_ready=first_ready,independent_handoff_window_passed=bool(independent_gate),
        post_handoff_frames=0 if first_ready is None else 2327+n-first_ready['frame_end'],
        closure_initial=(initial['mechanism_qpos'][15]-.37)/.482,closure_final=(final['mechanism_qpos'][15]-.37)/.482,
        origin_rise_m=physical[-1]['after_state']['bucket_origin_world_m'][2]-events[0]['before_state']['bucket_origin_world_m'][2],
        axis_peak_speed=np.max(abs(v),axis=0).tolist(),axis_max_half_jump=np.max(abs(np.diff(v,axis=0)),axis=0).tolist(),
        endpoint_sha256=sha(endpoint),physics_steps_added=0,transport_unload_tested=False)
    (W/'resume-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['audit_passed']

if __name__=='__main__':main()
