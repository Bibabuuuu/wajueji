"""Run once after the fixed-route run ends: review old contact and saved curl prefix."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
CONTACT=ROOT/'runs/contact-next-scoop-20260925-v1/candidate'
CARRY=ROOT/'runs/contact-second-finish-unload-20260925-v4/physical'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def bindings(directory,endpoint):
 paths=[directory/name for name in ('report.json','actions.ndjson','physics/frames.ndjson','physics/native-halves.ndjson','physics/response.ndjson',endpoint,endpoint+'.json')]
 return {str(p):sha(p) for p in paths}
def contact_review():
 sys.path.insert(0,str(ROOT/'work/contact-next-scoop-20260925-v1'))
 p=ROOT/'work/contact-next-scoop-20260925-v1/audit_candidate.py'
 spec=importlib.util.spec_from_file_location('_night_baseline_contact_audit',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 out=ROOT/'runs/contact-second-fixed-route-20260925-v1/contact-prefix-review';out.mkdir(exist_ok=False)
 module.W=out
 try:module.main()
 finally:
  if (out/'audit-review.json').exists():
   result=read(out/'audit-review.json');result['reviewed_inputs']=bindings(CONTACT,'final.npz')
   (W/'contact-audit.json').write_text(json.dumps(result,indent=2)+'\n')
def response_power(co):
 M=np.array(co['mass']).reshape(4,4);D=np.diag(co['drive_damping']);K=np.diag(co['stiffness']);P=np.array(co['passive_damping']).reshape(4,4);h=1/4800
 inv=np.linalg.inv(M+h*P);R=inv@(M-h*D);S=-h*inv@K;B=h*inv
 A=np.block([[np.eye(4)+h*S,h*R,h*B],[S,R,B],[np.zeros((4,8)),np.eye(4)]])
 J=np.array(co['jacobian']).reshape(6,4)
 return 60*J@np.linalg.matrix_power(A,80)[4:8,8:]@J.T

def carry_review():
 sys.path.insert(0,str(ROOT/'work/contact-second-finish-unload-20260925-v4'))
 from controller import CarryFeedback,LIMITS
 from layered_excavation_approach_planner import _load_kinematics
 from contact_bc_ground_guard import BucketGroundGuard
 from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
 from contact_transport_handoff import CarryHandoffGate,CarryHandoffConfig
 start=np.load(CONTACT/'final.npz');end=np.load(CARRY/'handoff.npz');meta=read(str(CARRY/'handoff.npz')+'.json');source_meta=read(str(CONTACT/'final.npz')+'.json')
 state=meta['highlevel_state'];report=read(CARRY/'report.json');actions=rows(CARRY/'actions.ndjson');physical=rows(CARRY/'physics/frames.ndjson');halves=rows(CARRY/'physics/native-halves.ndjson');responses=rows(CARRY/'physics/response.ndjson');timings=rows(CARRY/'physics/timing.ndjson');observations=rows(CARRY/'handoff.ndjson')
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh');q=start['mechanism_qpos'];guard=BucketGroundGuard(a,q)
 previous=np.array(read(CONTACT/'control-state.json')['last_action']);fb=CarryFeedback(a,c,guard,q,previous,frame_end=2725,initial_soil_top_m=state['carry']['initial_soil_top_m']);projector=ContactCommandProjector(c)
 gate=CarryHandoffGate(initial_soil_top_m=state['carry']['initial_soil_top_m'],source_sha256=sha(CONTACT/'final.npz'),config=CarryHandoffConfig(**state['gate']['config']))
 for entry in rows(CONTACT/'physics/frames.ndjson')[-181:]:
  s=entry['after_state'];pq=np.array(s['mechanism_qpos']);gate.observe_end(frame=entry['frame'],closure_fraction=(pq[15]-.37)/(.852-.37),origin_z_m=s['bucket_origin_world_m'][2],minimum_bucket_z_m=guard.measure(pq)['minimum_bucket_z_m'],contact_established=True)
 error=0.;details=True;gate_equal=True;peak=float(np.mean(fb.body_heights(q)));drawdown=0.
 for e,p,ob in zip(actions,physical,observations,strict=True):
  s=e['before_state'];fb.loaded_center_peak=peak
  command,detail=fb.command(frame=e['frame'],qpos=np.array(s['mechanism_qpos']),projector=projector,native_target=s['controller_target'],native_velocity=s['controller_target_velocity'])
  error=max(error,float(np.max(abs(command-e['command']))));details &= detail==e['detail']
  q=np.array(p['after_state']['mechanism_qpos']);center=float(np.mean(fb.body_heights(q)));peak=max(peak,center);drawdown=max(drawdown,peak-center)
  observed=gate.observe_end(frame=e['frame'],closure_fraction=(q[15]-.37)/(.852-.37),origin_z_m=p['after_state']['bucket_origin_world_m'][2],minimum_bucket_z_m=guard.measure(q)['minimum_bucket_z_m'],contact_established=True);gate_equal &= observed==ob
 fb.accept_handoff(readiness=observed,qpos=q)
 matrix_error=max(float(np.max(abs(response_power(x['coefficients'])-x['response_world']))) for x in responses)
 n=len(physical);commands=np.array([previous,*[e['command'] for e in actions]])
 checks=dict(frames=n==439==report['frames']==len(actions)==len(observations),command_recomputed=error<1e-12 and details,
  applied_commands=all(e['command']==p['applied_command_si'] for e,p in zip(actions,physical,strict=True)),
  handoff_recomputed=gate_equal and observed==report['handoff'] and observed['ready'],gate_history_saved=gate.snapshot()==state['gate'],
  carry_saved=all(np.array_equal(getattr(fb,key),value) for key,value in state['carry'].items()),
  restore=all(report['zero_arrays_exact'].values()),physical_guards=all(p['hard_guard'] for p in physical),drawdown=drawdown<=.010+1e-9,
  slew=float(np.max(abs(np.diff(commands,axis=0)/LIMITS)))<=2/60+1e-9,
  native_count=len(halves)==2*n,native_order=all(h['frame']==i//2+1 and h['half']==i%2+1 for i,h in enumerate(halves)),
  native_clock=np.max(abs(np.array([h['time_s'] for h in halves])-source_meta['controller_state']['native_time_s']-np.arange(1,len(halves)+1)/120))<1e-8,
  native_guards=all(h['dispatch_exact'] and h['forbidden_contact_count']==0 and h['closure_m']<=.001 for h in halves),
  response_count=len(responses)==n,matrix=matrix_error<1e-10,operator_once=all(x['mpm_operator_calls']==1 for x in responses),
  shadow_unchanged=all(x['live_state_unchanged'] and x['pending_unchanged_before_real_step'] for x in responses),
  real_step_once=len(timings)==n and all(x['real_step_calls_this_frame']==1 for x in timings),
  source=sha(CONTACT/'final.npz')==report['source_sha256'],endpoint=sha(CARRY/'handoff.npz')==report['handoff_checkpoint']['sha256'],
  endpoint_q=np.array_equal(end['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),
  finite=all(np.isfinite(end[k]).all() for k in end.files if end[k].dtype.kind in 'fciu'),particles=start['particle_q'].shape==end['particle_q'].shape,
  rock_ids=np.array_equal(start['rigid_ids'],end['rigid_ids']),jp_preserved=np.array_equal(start['particle_Jp'],end['particle_Jp']),
  sim_clock=abs(float(end['simulation_time_s'][0]-start['simulation_time_s'][0])-n/60)<1e-8,
  saved_normal_handoff=meta['stage']=='hold' and not report['stages'],
  bindings=all(sha(p)==h for field in ('candidate_bindings','tail_bindings') for p,h in meta[field].items()),
  runtime=all(sha(ROOT/'src/v4'/p)==h for p,h in meta['runtime_bindings'].items()))
 result=dict(audit_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},frames=n,maximum_command_error=error,matrix_max_error=matrix_error,
  endpoint_sha256=sha(CARRY/'handoff.npz'),original_run_status=report['status'],original_failure=report['failure'],physical_steps_added=0,
  scope='Normal handoff prefix before zero-step planner failure; original failure preserved',reviewed_inputs=bindings(CARRY,'handoff.npz'))
 (W/'carry-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['audit_passed']
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('segment',choices=['contact','carry']);args=parser.parse_args()
 status=read(ROOT/'runs/contact-second-fixed-route-20260925-v1/physical/report.json')['status']
 if status in ('initializing','running'):raise ValueError('wait_until_run_ends_for_unified_review')
 if (W/(args.segment+'-audit.json')).exists():raise ValueError('review_already_exists')
 if args.segment=='contact':contact_review()
 else:carry_review()
