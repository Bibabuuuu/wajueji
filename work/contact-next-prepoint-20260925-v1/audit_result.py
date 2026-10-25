"""Independent CPU control replay, clocks, identities and checkpoint audit."""
import json
import numpy as np
from runtime import ROOT,W,SOURCE,PrepointRuntime,read,sha
from controller import PrepointController,PrepointClearance,LIMITS
from layered_47_cycle_geometry import TransportClearance
from observation import stationary_terrain
from audit import matrix_power_response

RUN=ROOT/'runs/contact-next-prepoint-20260925-v1/physical'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
    report=read(RUN/'report.json');physical=rows(RUN/'physics/frames.ndjson');n=len(physical)
    actions=rows(RUN/'actions.ndjson');halves=rows(RUN/'physics/native-halves.ndjson')
    response=rows(RUN/'physics/response.ndjson');timings=rows(RUN/'physics/timing.ndjson')
    endpoint=RUN/('prepoint.npz' if report['status']=='prepoint_saved_requires_independent_audit' else 'recovery.npz')
    meta=read(str(endpoint)+'.json');source_meta=read(str(SOURCE)+'.json')
    initial=np.load(SOURCE);final=np.load(endpoint)
    r=PrepointRuntime(SOURCE,ROOT/'runs/contact-next-prepoint-20260925-v1/audit-runtime')
    terrain,_=stationary_terrain(r);checker=PrepointClearance(r,terrain)
    previous=np.array(r.metadata['transport_controller_state']['previous_command'])
    controller=PrepointController(r,checker,previous);error=0.;details=True
    for e in actions[:n]:
        command,detail=controller.command('PREPOINT',np.array(e['before_state']['mechanism_qpos']),e['controller_target'],np.array(e['desired_qpos']),np.array(e['reference_velocity']))
        error=max(error,float(np.max(abs(command-e['command']))));details &= detail==e['detail']
    matrix_error=max(float(np.max(abs(matrix_power_response(x['coefficients'])-x['response_world']))) for x in response)
    commands=np.array([previous,*[e['command'] for e in actions[:n]]]);slew=float(np.max(abs(np.diff(commands,axis=0)/LIMITS)))
    checks=dict(restore=all(report['zero_arrays_exact'].values()),frames=n==report['frames'],
        command_recomputed=error<1e-12 and details,command_physics=all(e['command']==p['applied_command_si'] for e,p in zip(actions[:n],physical,strict=True)),
        first_command=np.array_equal(commands[0],commands[1]),slew=slew<=2/60+1e-9,
        clearance=all(e['bucket_terrain_m']>=.30 and e['mechanism_terrain_clearance_m']>=.02 and e['static_truck_aabb_separation_m']>=.02 for e in rows(RUN/'clearance.ndjson')),
        frame_order=all(e['local_frame']==i+1 for i,e in enumerate(actions[:n])),
        physical=all(p['hard_guard'] for p in physical),halves=len(halves)==2*n,
        half_order=all(h['frame']==i//2+1 and h['half']==i%2+1 for i,h in enumerate(halves)),
        clock=np.max(abs(np.array([h['time_s'] for h in halves])-source_meta['controller_state']['native_time_s']-np.arange(1,2*n+1)/120))<1e-8,
        native_guard=all(h['forbidden_contact_count']==0 and h['closure_m']<=.001 for h in halves),
        dispatch=all(h['dispatch_exact'] for h in halves),responses=len(response)==n,
        operator_once=all(x['mpm_operator_calls']==1 for x in response),shadow_unchanged=all(x['live_state_unchanged'] and x['pending_unchanged_before_real_step'] for x in response),
        real_step_once=len(timings)==n and all(x['real_step_calls_this_frame']==1 for x in timings),matrix=matrix_error<1e-10,
        simulation_clock=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-n/60)<1e-8,
        final_state=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),
        finite=all(np.isfinite(final[k]).all() for k in final.files if final[k].dtype.kind in 'fciu'),
        particles=initial['particle_q'].shape==final['particle_q'].shape,rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
        source=sha(SOURCE)==read(W/'protocol.json')['source_sha256'],lineage=all(meta.get(k)==source_meta.get(k) for k in ('lineage_root','training_reset')),
        jp_preserved=np.array_equal(initial['particle_Jp'],final['particle_Jp']),jp_policy=meta['material_history_policy']=='zero_hardening_frozen_jp_v1',
        truck=meta['truck_enabled'],schema=meta['schema']=='layered_47_full_response_next_prepoint_v1',cycle=meta['cycle_index']==(2 if report['status']=='prepoint_saved_requires_independent_audit' else 1),
        bindings=all(sha(p)==h for field in ('candidate_bindings','prepoint_bindings','prepoint_dependency_bindings') for p,h in meta[field].items()),
        runtime=all(sha(ROOT/'src/v4'/p)==h for p,h in meta['runtime_bindings'].items()),
        load_bias_saved=np.array_equal(meta['transport_controller_state']['load_bias'],controller.tracking.bias),
        previous_command_saved=np.array_equal(meta['transport_controller_state']['previous_command'],controller.previous),
        inputs=all(sha(p)==h for p,h in read(W/'protocol.json')['inputs'].items()),
        protocol=sha(W/'protocol.json')==report['protocol_sha256'],budget=n<=4830)
    v=np.array([h['mechanism_qvel'] for h in halves])[:,[6,8,11,14]]
    result=dict(audit_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        status=report['status'],failure=report.get('failure'),frames=n,command_max_error=error,
        maximum_normalized_command_step=slew,matrix_max_error=matrix_error,wall_seconds=report['wall_seconds'],
        selected=report['selected'],endpoint_sha256=sha(endpoint),axis_peak_speed=np.max(abs(v),axis=0).tolist(),
        physics_steps_added=0,formal_mixed_delivered_volume_qualified=False)
    (W/'audit-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['audit_passed']
if __name__=='__main__':main()
