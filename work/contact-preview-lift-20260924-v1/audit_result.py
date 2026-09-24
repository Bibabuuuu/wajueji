"""Read-only D153 audit, no physics steps."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/contact-preview-lift-20260924-v1';sys.path.insert(0,str(W))
from continuation_runtime import SOURCE,SOURCE_SHA,sha
from control_history import construct,restore,OUT,read,rows,encode,DYNAMIC
from lift_feedback import LiftFeedback
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from audit import matrix_power_response
RUN=OUT/'continuation'
def main():
 report=read(RUN/'report.json');events=rows(RUN/'actions.ndjson');phys=rows(RUN/'physics/frames.ndjson');stages=rows(RUN/'phase-trace.ndjson');halves=rows(RUN/'physics/native-halves.ndjson');responses=rows(RUN/'physics/response.ndjson');times=rows(RUN/'physics/timing.ndjson');n=len(phys)
 fb,ph,features,_,_=construct(LiftFeedback);restore(fb,ph,features,OUT/'control-history.json');projector=ContactCommandProjector(fb.chain)
 error=0;details=True;phase_exact=True
 for e,row,s in zip(events[:n],phys,stages,strict=True):
  req,cmd,detail=fb.command(frame=e['frame'],qpos=e['before_state']['mechanism_qpos'],phase=e['phase'],bc_action=e['action_normalized'],projector=projector)
  error=max(error,float(max(abs(req-e['requested_command_si']))),float(max(abs(cmd-e['applied_command_si']))));details &= detail==e['feedback']
  observation=s['observation'];q=row['after_state']['mechanism_qpos'];pose=fb.assembly.assemble(q,include_meshes=False)[1]['bucket']
  from contact_phase_live import mouth_angle_deg
  phase=ph.router.observe_end(frame_end=e['frame'],soil_force_n=s['bucket_mpm_force_norm_n'],rock_contact_substeps=row['rock_bucket_contact_substeps'],mouth_angle_deg=mouth_angle_deg(pose,ph.polygon),actual_boom_cylinder_m=q[9],bucket_origin_z_m=pose[2,3],remaining_soil_gap_m=None if observation is None else observation['remaining_soil_gap_m'],gap_frame_end=None if observation is None else e['frame'])
  phase_exact &= phase==s['phase']
 endpoint=RUN/('final.npz' if report['status']=='bounded_continuation_completed' else 'recovery.npz');meta=read(str(endpoint)+'.json');source_meta=read(str(SOURCE)+'.json');final=np.load(endpoint);initial=np.load(SOURCE)
 matrix_error=max(float(np.max(abs(matrix_power_response(x['coefficients'])-x['response_world']))) for x in responses)
 saved=read(RUN/'control-state.json');control_equal=encode({k:getattr(fb,k) for k in (*DYNAMIC,'lift_intent','lift_start_origin_z')})==saved['feedback']
 checks=dict(exact_restore=all(report['zero_step_arrays_exact'].values()),serialized_history_bc_exact=report['serialized_history_bc_exact'],control_recomputed=error<1e-12 and details,phase_recomputed=phase_exact,saved_control_recomputed=control_equal,frames=n==report['frames_completed'],one_response_per_frame=len(responses)==n,two_halves_per_frame=len(halves)==2*n,one_native_dispatch_per_frame=len(times)==n and all(x['real_step_calls_this_frame']==1 for x in times),half_order=all(x['frame']==i//2+1 and x['half']==i%2+1 for i,x in enumerate(halves)),native_clock=max(abs(np.array([h['time_s'] for h in halves])-source_meta['controller_state']['native_time_s']-np.arange(1,2*n+1)/120))<1e-8,dispatch_exact=all(h['dispatch_exact'] for h in halves),native_guard=all(h['closure_m']<=.001 and h['forbidden_contact_count']==0 for h in halves),physical_guards=all(x['hard_guard'] for x in phys),ground_guards=all(x['safe'] for x in rows(RUN/'ground-guard.ndjson')),operator_once=all(x['mpm_operator_calls']==1 for x in responses),shadow_no_live_mutation=all(x['live_state_unchanged'] and x['pending_unchanged_before_real_step'] for x in responses),independent_matrix=matrix_error<1e-10,source=sha(SOURCE)==SOURCE_SHA,inputs=all(sha(x['path'])==x['sha256'] for x in read(W/'protocol.json')['inputs']),candidate_bindings=all(sha(p)==v for p,v in meta['control_candidate_bindings'].items()),runtime_bindings=all(sha(ROOT/'src/v4'/p)==v for p,v in meta['runtime_bindings'].items()),schema=meta['schema']=='layered_47_full_response_lift_candidate_checkpoint_v1',simulation_clock=abs(float(final['simulation_time_s'][0]-initial['simulation_time_s'][0])-n/60)<1e-8,final_state=np.array_equal(final['mechanism_qpos'],phys[-1]['after_state']['mechanism_qpos']),finite=all(np.isfinite(final[k]).all() for k in final.files if final[k].dtype.kind in 'fciu'),lineage=all(meta[k]==source_meta[k] for k in ('lineage_root','training_reset')),particle_count=initial['particle_q'].shape==final['particle_q'].shape,rock_ids=np.array_equal(initial['rigid_ids'],final['rigid_ids']),frame_budget=n<=1200)
 from layered_47_cycle_checkpoint import load
 try:load(endpoint,allow_training_reset=True);checks['ordinary_loader_rejects']=False
 except ValueError as e:checks['ordinary_loader_rejects']=str(e)=='cycle_checkpoint_schema'
 v=np.array([h['mechanism_qvel'] for h in halves])[:,[6,8,11,14]]
 result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},command_max_error=error,matrix_max_error=matrix_error,frames=n,final_phase=report.get('final_phase'),stop_reason=report.get('stop_reason'),axis_peak_speed=np.max(abs(v),axis=0).tolist(),axis_max_half_jump=np.max(abs(np.diff(v,axis=0)),axis=0).tolist(),endpoint_sha256=sha(endpoint),physics_steps_added=0,wall_seconds=report['wall_seconds'],timing=report['timing'])
 (W/'audit-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
