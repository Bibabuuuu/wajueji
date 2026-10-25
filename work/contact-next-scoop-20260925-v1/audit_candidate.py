"""Post-run audit: independent command replay, discrete matrix power, clocks/hashes."""
import json,hashlib,argparse
from candidate_multisource import LiftFeedback as ContactGeometryFeedback
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import FeedbackConfig
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from gates import behavior_gate
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
RUN=ROOT/'runs/contact-next-scoop-20260925-v1/candidate'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]

def matrix_power_response(co):
    M=np.array(co['mass']).reshape(4,4);D=np.diag(co['drive_damping']);K=np.diag(co['stiffness']);P=np.array(co['passive_damping']).reshape(4,4);h=1/4800
    inv=np.linalg.inv(M+h*P);R=inv@(M-h*D);S=-h*inv@K;B=h*inv
    # State [displacement,velocity,constant force], one symplectic Euler step.
    A=np.block([[np.eye(4)+h*S,h*R,h*B],[S,R,B],[np.zeros((4,8)),np.eye(4)]])
    v=np.linalg.matrix_power(A,80)[4:8,8:]
    J=np.array(co['jacobian']).reshape(6,4)
    return 60*J@v@J.T

def main():
    report=read(RUN/'report.json');actions=rows(RUN/'actions.ndjson');physical=rows(RUN/'physics/frames.ndjson');trace=rows(RUN/'phase-trace.ndjson');responses=rows(RUN/'physics/response.ndjson');halves=rows(RUN/'physics/native-halves.ndjson');timings=rows(RUN/'physics/timing.ndjson')
    endpoint=RUN/('final.npz' if report['status']=='bounded_probe_completed' else 'physics/failure.npz');final=np.load(endpoint);initial=np.load(report['source']['path']);meta=read(str(endpoint)+'.json');source_meta=read(report['source']['path']+'.json');n=len(physical)
    k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh');q=initial['mechanism_qpos']
    ref=GroundSoilCutReference(initial['particle_q'],read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'));guard=BucketGroundGuard(a,q);feedback=ContactGeometryFeedback(a,c,guard,report['selected_goal'],q,FeedbackConfig(**report['feedback_parameters']),ref);projector=ContactCommandProjector(c)
    error=0;detail_equal=True
    for event in actions:
        req,cmd,detail=feedback.command(frame=event['frame'],qpos=event['before_state']['mechanism_qpos'],phase=event['phase'],bc_action=event['action_normalized'],projector=projector)
        error=max(error,float(np.max(abs(req-event['requested_command_si']))),float(np.max(abs(cmd-event['applied_command_si']))));detail_equal &= detail==event['feedback']
    matrix_error=max(float(np.max(np.abs(matrix_power_response(x['coefficients'])-x['response_world']))) for x in responses)
    half_times=np.array([x['time_s'] for x in halves]);t0=source_meta['controller_state']['native_time_s']
    checks=dict(exact_restore=all(report['zero_step_arrays_exact'].values()),control_recomputed=error<1e-12 and detail_equal,
        frames=n==report['frames_completed'],one_response_per_frame=len(responses)==n,one_real_step_per_frame=len(timings)==n and all(t['real_step_calls_this_frame']==1 for t in timings),two_native_halves_per_frame=len(halves)==2*n,
        half_order=all(e['half']==i%2+1 and e['frame']==i//2+1 for i,e in enumerate(halves)),half_clock=np.max(abs(half_times-t0-np.arange(1,len(halves)+1)/120))<1e-8,
        native_dispatch_exact=all(h['dispatch_exact'] for h in halves),native_guards=all(h['closure_m']<=.001 and h['forbidden_contact_count']==0 for h in halves),
        operator_once_per_frame=all(x['mpm_operator_calls']==1 for x in responses),shadow_did_not_mutate_live=all(x['live_state_unchanged'] and x['pending_unchanged_before_real_step'] for x in responses),matrix_independent_power=matrix_error<1e-10,
        source=sha(report['source']['path'])==report['source']['sha256'],policy=sha(report['policy']['path'])==report['policy']['sha256'],lineage=all(meta.get(key)==source_meta.get(key) for key in ('lineage_root','training_reset')),
        particle_count=initial['particle_q'].shape==final['particle_q'].shape,rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),finite=all(np.isfinite(final[k]).all() for k in final.files if final[k].dtype.kind in 'fciu'),
        phase_action=all(e['phase']==t['phase']['phase_before_observation'] for e,t in zip(actions,trace)),phase_next=all(actions[i]['phase']==trace[i-1]['phase']['phase_for_next_decision'] for i in range(1,min(len(actions),len(trace)+1))),
        protocol=sha(report['phase_contract']['path'])==report['phase_contract']['sha256'],candidate_code=all(sha(p)==d for p,d in report['candidate_code'].items()),
        candidate_schema=meta['schema']=='layered_47_full_response_real_second_contact_v1',final_state_matches=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),simulation_clock=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-n/60)<1e-8)
    for file,digest in {**report['runtime_bindings'],**report['code_sha256']}.items():checks['runtime_'+file]=sha(ROOT/'src/v4'/file)==digest
    checks['control_bindings']=all(sha(p)==d for p,d in meta['control_bindings'].items())
    checks['multisource_bindings']=all(sha(p)==d for p,d in meta['multisource_bindings'].items())
    checks['inputs']=all(sha(v['path'])==v['sha256'] for v in read(report['phase_contract']['path'])['inputs'])
    checks['new_contact_bindings']=all(sha(p)==d for p,d in meta['contact_bindings'].items())
    checks['preview_binding']=all(sha(p)==d for p,d in meta['preview_binding'].items())
    checks['actual_prior_command']=actions[0]['previous_applied_action_si']==source_meta['transport_controller_state']['previous_command']
    if report['status']=='bounded_probe_completed':
        saved=read(RUN/'control-state.json')
        checks['saved_causal_history']=(saved['frame']==n and saved['feature_last_frame']==n and saved['feedback']['last_frame']==n and saved['last_action']==actions[n-1]['applied_command_si'])
    checks['constant_wrench_over_both_halves']=all(halves[2*i]['applied_bucket_mpm_force_world_n']==halves[2*i+1]['applied_bucket_mpm_force_world_n'] and halves[2*i]['applied_bucket_mpm_torque_world_nm']==halves[2*i+1]['applied_bucket_mpm_torque_world_nm'] for i in range(n))
    from layered_47_cycle_checkpoint import load
    try:load(endpoint,allow_training_reset=True);checks['ordinary_resume_rejected']=False
    except ValueError as e:checks['ordinary_resume_rejected']=str(e)=='cycle_checkpoint_schema'
    behavior=behavior_gate(actions[:n],physical,feedback,require_curl=n>=1200)
    result=dict(audit_passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},maximum_recomputed_command_error=error,maximum_independent_matrix_error=matrix_error,behavior=behavior,status=report['status'],failure=report.get('failure'),physics_steps_added=0)
    (W/'audit-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['audit_passed']
if __name__=='__main__':main()
