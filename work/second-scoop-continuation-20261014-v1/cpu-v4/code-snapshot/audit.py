"""Read-only second-scoop CPU artifact audit; no native initialization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,time,subprocess,numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sys.path.insert(0,str(ROOT/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import CpuRuntime,sha,read,write,FixedRouteClearance
from planning.rear_lower_prepoint import ground_completed_terrain,sampled_surface_clearance
def main():
 start=time.monotonic();checks={}
 def check(k,v):checks[k]=bool(v)
 p=read(W/'cpu-v4/report.json');rot=read(W/'rotation-probe-v2/report.json')
 check('CPU approach and reference probe complete',p['status']=='cpu_geometry_passed_reference_adaptation_required' and rot['status']=='cpu_rotation_geometry_passed')
 source=Path(p['source']);r=CpuRuntime(source,W/'audit-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 check('source NPZ bound unchanged',sha(source)==p['source_sha256'])
 check('source metadata unchanged',sha(str(source)+'.json')==p['source_metadata_sha256'])
 check('source report unchanged',sha(source.parent/'report.json')==p['source_report_sha256'])
 check('source original audit unchanged',sha(source.parent/'read-only-audit.json')==p['source_audit_sha256'])
 protocol=read(ROOT/'work/continuous-single-scoop-20261014-v1/protocol.json')
 check('all frozen single-scoop code unchanged',all(sha(Path(n))==h for n,h in protocol['code_sha256'].items()))
 check('frozen model unchanged',sha(ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip')==p['model_sha256'])
 check('no training reset or physical source relabel',not r.metadata.get('training_reset') and r.metadata['cycle_index']==1 and r.metadata['stage']=='returned')
 check('original physical particle and rock count',len(r.arrays['particle_q'])==256441 and len(r.records)==47)
 check('new candidate uses new observed stock',p['next_prepoint']['cycle_index']==2 and p['candidate_count']==8)
 check('old reference explicitly rejected',not p['unmodified_reference_entry']['trusted'] and p['unmodified_reference_entry']['position_error_m']>.15)
 check('CPU only zero physics inference training',all(p[k]==0 for k in ['physics_steps_added','native_initializations','policy_inference_calls','training_updates']))
 check('rotation probe CPU only not admitted',not rot['execution_approved'] and all(rot[k]==0 for k in ['physics_steps_added','policy_inference_calls','training_updates']))
 check('rotation geometry and entry gates',all(rot['checks'].values()))
 with np.load(W/'cpu-v4/next-prepoint-plan.npz') as f:q=f['qpos'].copy();velocity=f['velocity'].copy()
 check('planned state command shapes',q.shape==(p['next_prepoint']['total_frames']+1,19) and velocity.shape==(len(q)-1,4))
 check('all path states commands finite',np.isfinite(q).all() and np.isfinite(velocity).all())
 check('actuator displacement matches 60Hz SI commands',np.max(abs(np.diff(q[:,r.chain.axes],axis=0)-velocity/60))<=1e-8)
 check('initial physical actuator and base pose preserved',np.max(abs(q[0,r.chain.axes]-r.arrays['mechanism_qpos'][r.chain.axes]))<=1e-12 and np.array_equal(q[0,:7],r.arrays['mechanism_qpos'][:7]))
 check('original absolute joint hard bounds',all(np.all((q[:,i]>=lo-1e-9)&(q[:,i]<=hi+1e-9)) for i,(lo,hi) in r.chain.limits.items()))
 check('path max velocities original SI bounds',np.all(np.max(abs(velocity),axis=0)<=np.array([.12,.0225,.0225,.0225])+1e-9))
 worst=max(float(np.max(abs(r.chain.residual(x)))) for x in q)
 check('all recorded path closures <=0.1mm',worst<=.0001)
 check('original arrays unmodified',p['original_arrays_unchanged'] and p['source_unchanged'])
 # Independently re-evaluate every segment boundary and one state/30 frames.
 with np.load(W/'cpu-v4/terrain.npz') as f:t=ground_completed_terrain({k:f[k] for k in f.files})
 static=FixedRouteClearance(r);bounds=np.cumsum([s['frames'] for s in p['next_prepoint']['segments']]);rise_end=int(bounds[0])
 indices=sorted(set(range(0,len(q),30))|set(bounds.tolist())|{0,len(q)-1})
 terrain=[];vertices=[];static_ok=True
 for i in indices:
  try:static.check(q[i])
  except ValueError:static_ok=False
  pose=r.assembly.assemble(q[i],include_meshes=False)[1]['bucket'];points=static.samples@pose[:3,:3].T+pose[:3,3]
  terrain.append((i,sampled_surface_clearance(points,t)['clearance_m']));vertices.append((i,float(points[:,2].min())))
 check('independent sampled fixed-route checks',static_ok)
 check('return extension does not reduce inherited gap',all(g>=p['next_prepoint']['return_rise']['inherited_bucket_stock_gap_m']-1e-8 for i,g in terrain if i<=rise_end))
 check('approach begins only after original32cm gate',all(g>=.32 for i,g in terrain if i>=rise_end))
 check('sampled bucket remains above original5mm floor',all(z>=.005 for i,z in vertices))
 check('17 existing regressions passed','Ran 17 tests' in (W/'existing-cpu-tests-v2.log').read_text() and (W/'existing-cpu-tests-v2.log').read_text().strip().endswith('OK'))
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 check('GPU compute idle after CPU work',not gpu)
 snapshots=W/'cpu-v4/code-snapshot';snapshots.mkdir(exist_ok=False)
 import shutil
 for name in ['preflight.py','next_prepoint_candidate.py','rotation_probe.py','audit.py']:shutil.copyfile(W/name,snapshots/name)
 result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),independent_geometry_states=len(indices),
  maximum_recorded_closure_m=worst,original_source_clock_s=p['simulation_time_s'],source_sha256=p['source_sha256'],
  source_final_bucket_mpm_contact_samples=p['source_final_physics_diagnostics']['bucket_mpm_contact_samples'],
  contact_classification_boundary='Original final row has bucket contact samples; adhesion versus grounded stock not established by aggregate counts. No strict contact-free continuation claim.',
  original_old_failures_retained=[str(d) for d in W.glob('cpu-v[123]')],
  code_sha256={n:sha(snapshots/n) for n in ['preflight.py','next_prepoint_candidate.py','rotation_probe.py','audit.py']},
  policy_input_adaptation_implemented=False,physical_runner_admitted=False,new_physics_steps=0,training_updates=0,
  scope='CPU sampled geometry and reference feasibility only; no second-scoop excavation, continuous three-scoop or policy-generalization result',wall_seconds=time.monotonic()-start)
 write(W/'independent-cpu-audit.json',result);print(result['status'],result['check_count'],[k for k,v in checks.items() if not v],flush=True)
 assert all(checks.values())
if __name__=='__main__':main()
