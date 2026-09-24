"""Post-run CPU-only checks; no simulator stepping or outcome rewriting."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/contact-second-fixed-route-20260925-v1'))
from runtime import RUN,SOURCE,SOURCE_SHA,SOURCE_FRAME,MAX_FRAMES,TransportRuntime,read,sha
from controller import TransportController,LIMITS
from clearance import FixedRouteClearance
from layered_47_cycle_geometry import bucket_mask
from terrain.truck_receiver import TruckReceiver,CAVITY
PHYSICAL=RUN/'physical'
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
def independent_response(co):
    M=np.array(co['mass']).reshape(4,4);D=np.diag(co['drive_damping']);K=np.diag(co['stiffness']);P=np.array(co['passive_damping']).reshape(4,4);h=1/4800
    inv=np.linalg.inv(M+h*P);R=inv@(M-h*D);S=-h*inv@K;B=h*inv
    A=np.block([[np.eye(4)+h*S,h*R,h*B],[S,R,B],[np.zeros((4,8)),np.eye(4)]])
    J=np.array(co['jacobian']).reshape(6,4)
    return 60*J@np.linalg.matrix_power(A,80)[4:8,8:]@J.T

def main():
    report=read(PHYSICAL/'report.json');physical=rows(PHYSICAL/'physics/frames.ndjson');n=len(physical)
    actions=rows(PHYSICAL/'actions.ndjson');halves=rows(PHYSICAL/'physics/native-halves.ndjson');response=rows(PHYSICAL/'physics/response.ndjson');timings=rows(PHYSICAL/'physics/timing.ndjson')
    endpoint=Path(report['final_checkpoint']['path']);meta=read(str(endpoint)+'.json');source_meta=read(str(SOURCE)+'.json')
    initial=np.load(SOURCE);final=np.load(endpoint)
    r=TransportRuntime(SOURCE,RUN/'audit-runtime');control=TransportController(r,FixedRouteClearance(r),np.array(source_meta['highlevel_state']['carry']['previous_command']))
    previous=control.previous.copy();error=0.;details=True
    for e in actions[:n]:
        command,detail=control.command(e['stage'],np.array(e['before_state']['mechanism_qpos']),e['controller_target'],np.array(e['desired_qpos']),np.array(e['reference_velocity']))
        error=max(error,float(np.max(abs(command-e['command']))));details &= detail==e['detail']
    matrix_error=max((float(np.max(abs(independent_response(x['coefficients'])-x['response_world']))) for x in response),default=0.)
    commands=np.array([previous,*[e['command'] for e in actions[:n]]]);slew=float(np.max(abs(np.diff(commands,axis=0)/LIMITS))) if n else 0.
    saved=meta['highlevel_state'].get('transport',{})
    checks=dict(restore=all(report['zero_arrays_exact'].values()),frames=n==report['frames'],
        command_recomputed=error<1e-12 and details,command_physics=len(actions)>=n and all(e['command']==p['applied_command_si'] for e,p in zip(actions[:n],physical,strict=True)),
        slew=slew<=2/60+1e-9,loaded_no_open=all(e['command'][3]>=-1e-10 for e in actions[:n] if e['stage'] in ('CLEAR_PILE','ALIGN')),
        frame_order=all(e['frame']==SOURCE_FRAME+i+1 for i,e in enumerate(actions[:n])),physical=all(p['hard_guard'] for p in physical),
        halves=len(halves)==2*n,half_order=all(h['frame']==i//2+1 and h['half']==i%2+1 for i,h in enumerate(halves)),
        clock=(not halves or np.max(abs(np.array([h['time_s'] for h in halves])-source_meta['controller_state']['native_time_s']-np.arange(1,len(halves)+1)/120))<1e-8),
        native_guard=all(h['forbidden_contact_count']==0 and h['closure_m']<=.001 for h in halves),dispatch=all(h['dispatch_exact'] for h in halves),responses=len(response)==n,
        operator_once=all(x['mpm_operator_calls']==1 for x in response),shadow_unchanged=all(x['live_state_unchanged'] and x['pending_unchanged_before_real_step'] for x in response),
        real_step_once=len(timings)==n and all(x['real_step_calls_this_frame']==1 for x in timings),matrix=matrix_error<1e-10,
        simulation_clock=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-n/60)<1e-8,
        final_state=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos'] if n else initial['mechanism_qpos']),
        finite=all(np.isfinite(final[k]).all() for k in final.files if final[k].dtype.kind in 'fciu'),particles=initial['particle_q'].shape==final['particle_q'].shape,
        rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),source=sha(SOURCE)==SOURCE_SHA,
        lineage=all(meta.get(k)==source_meta.get(k) for k in ('lineage_root','training_reset','cycle_index')),
        jp_preserved=np.array_equal(initial['particle_Jp'],final['particle_Jp']),jp_policy=meta['material_history_policy']=='zero_hardening_frozen_jp_v1',
        truck=meta['truck_enabled'],schema=meta['schema']=='layered_47_full_response_second_fixed_route_v1',
        bindings=all(sha(p)==h for field in ('candidate_bindings','fixed_route_bindings') for p,h in meta[field].items()),
        runtime=all(sha(ROOT/'src/v4'/p)==h for p,h in meta['runtime_bindings'].items()),
        previous_command_saved=np.array_equal(saved.get('previous_command'),control.previous),load_bias_saved=np.array_equal(saved.get('load_bias'),control.tracking.bias),
        inputs=all(sha(p)==h for p,h in read(ROOT/'work/contact-second-fixed-route-20260925-v1/protocol.json')['inputs'].items()),
        budget=n<=MAX_FRAMES,terrain_explicitly_unmeasured=all(x['bucket_terrain_m'] is None for x in rows(PHYSICAL/'clearance.ndjson')))
    inventory=None
    if report['status']=='second_scoop_unload_completed':
        pose=r.assembly.assemble(initial['mechanism_qpos'],include_meshes=False)[1]['bucket'];payload=bucket_mask(initial['particle_q'],pose)
        pose=r.assembly.assemble(final['mechanism_qpos'],include_meshes=False)[1]['bucket'];receiver=TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))
        inventory=dict(source_particles=int(payload.sum()),received_source_particles=int((payload&(receiver.classify(final['particle_q'])==CAVITY)).sum()),remaining_source_particles=int((payload&bucket_mask(final['particle_q'],pose)).sum()))
        checks['inventory_recomputed']=all(report['unload_inventory'][k]==v for k,v in inventory.items())
        checks['completed_stages']=[x['name'] for x in report['stages']]==['CLEAR_PILE','ALIGN','DUMP']
    result=dict(audit_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},status=report['status'],failure=report.get('failure'),
        frames=n,command_max_error=error,matrix_max_error=matrix_error,maximum_normalized_command_step=slew,development_inventory=inventory,
        endpoint_sha256=sha(endpoint),physics_steps_added=0,formal_three_scoop_qualified=False,formal_fill_qualified=False)
    (W/'audit-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['audit_passed']
if __name__=='__main__':main()
