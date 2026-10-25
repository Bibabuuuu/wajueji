"""CPU-only investigation: actual first-return stock -> next candidate/prepoint.

No native context, physical steps, policy inference, training or state mutation.
Existing single-scoop code/model/physical data remain read-only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,hashlib,signal,importlib.util,traceback
import numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sys.path.insert(0,str(ROOT/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import CpuRuntime,sha,read,write,ProgressAlignment,MODEL,MODEL_SHA
from manual_residual_sac_env import ManualResidualEnv
import layered_47_cycle_selection as selection
SOURCE=ROOT/'runs/continuous-single-scoop-20261014-v1/final.npz'
RUN=SOURCE.parent
def arrays_sha(a):return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()}
def main():
 out=W/'cpu-v4';out.mkdir(exist_ok=False);start=time.monotonic()
 report=dict(status='checking',physics_steps_added=0,native_initializations=0,policy_inference_calls=0,training_updates=0,
             source=str(SOURCE),source_sha256=sha(SOURCE),source_metadata_sha256=sha(str(SOURCE)+'.json'),
             scope='CPU geometry and frozen-reference compatibility only; not physical continuation or two/three-scoop evidence')
 def timeout(*_):raise TimeoutError('CPU 300 second ceiling')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
 try:
  source_report=read(RUN/'report.json');audit=read(RUN/'read-only-audit.json')
  assert source_report['status']=='continuous_single_scoop_completed' and audit['status']=='passed'
  assert source_report['final_checkpoint']['sha256']==report['source_sha256']
  assert sha(MODEL)==MODEL_SHA
  report.update(source_report_sha256=sha(RUN/'report.json'),source_audit_sha256=sha(RUN/'read-only-audit.json'),model_sha256=MODEL_SHA)
  with (RUN/'physics/frames.ndjson').open() as f:
   for line in f:last=line
  last=json.loads(last)
  report['source_final_physics_row_keys']=list(last)
  report['source_final_physics_diagnostics']={k:v for k,v in last.items() if 'contact' in k or 'coupl' in k or 'guard' in k}
  r=CpuRuntime(SOURCE,out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  identity=arrays_sha(r.arrays);report.update(source_stage=r.metadata['stage'],source_cycle_index=r.metadata['cycle_index'],
    simulation_time_s=float(r.arrays['simulation_time_s'][0]),particle_count=len(r.arrays['particle_q']),rock_count=len(r.records))
  assert r.metadata['stage']=='returned' and r.metadata['cycle_index']==1 and not r.metadata.get('training_reset')
  for method in [lambda:r.__enter__(),lambda:r.step(np.zeros(4))]:
   try:method()
   except RuntimeError:pass
   else:raise AssertionError('CPU runtime allowed native entry/step')
  # Reuse the historical supported-rock observer as-is, with the new arrays.
  folder=ROOT/'work/contact-fixed-truck-continuation-20260925-v2'
  sys.path.insert(0,str(folder))
  spec=importlib.util.spec_from_file_location('second_scoop_stock_observation',folder/'observation.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  report['observer_sha256']=sha(folder/'observation.py')
  terrain,observation=module.stationary_terrain(r,r.arrays)
  candidates=selection.rank_workspace_candidates(terrain,2)
  report.update(observation=observation,candidate_count=len(candidates),candidates=candidates)
  np.savez(out/'terrain.npz',**{k:np.asarray(v) for k,v in terrain.items() if isinstance(v,(list,np.ndarray,int,float))})
  write(out/'candidates.json',dict(source_sha256=report['source_sha256'],source_simulation_time_s=report['simulation_time_s'],observation=observation,candidates=candidates))
  print('CPU candidates',len(candidates),'source clock',report['simulation_time_s'],flush=True)
  from next_prepoint_candidate import plan_next
  plan=plan_next(r,terrain,observation)
  path=plan.pop('path');plan.pop('terrain')
  np.savez(out/'next-prepoint-plan.npz',qpos=path['qpos'],velocity=path['velocity'])
  report['next_prepoint']=dict(**plan,total_frames=path['total_frames'],segments=path['segments'],
                              prepoint_tip_world_m=path['prepoint_tip_world_m'],plan_sha256=sha(out/'next-prepoint-plan.npz'))
  # Show the precise current-policy failure rather than silently rebasing data.
  from types import SimpleNamespace
  env=ManualResidualEnv(out/'reference-read-only',SimpleNamespace(),collect=False,waiting_action_mode='error_strength_v1')
  poses=np.asarray([r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in env.reference_q])
  q=np.array(path['qpos'][-1]);pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
  alignment=ProgressAlignment(env.reference_q,poses).observe(q,pose,0.)
  report['unmodified_reference_entry']=alignment
  target=plan['selected']['contact'];report['current_policy_fixed_world_anchor_m']=[5.85,-.05,.65]
  report['selected_contact']=target
  report.update(status='cpu_geometry_passed' if alignment['trusted'] else 'cpu_geometry_passed_reference_adaptation_required',
                original_arrays_unchanged=arrays_sha(r.arrays)==identity,
                source_unchanged=sha(SOURCE)==report['source_sha256'] and sha(str(SOURCE)+'.json')==report['source_metadata_sha256'])
  assert report['original_arrays_unchanged'] and report['source_unchanged']
 except BaseException as e:
  report.update(status='cpu_failed',failure=repr(e),traceback=traceback.format_exc())
  raise
 finally:
  signal.alarm(0);report['wall_seconds']=time.monotonic()-start
  report['source_unchanged']=sha(SOURCE)==report['source_sha256'] and sha(str(SOURCE)+'.json')==report['source_metadata_sha256']
  write(out/'report.json',report)
  print(json.dumps({k:report[k] for k in ['status','wall_seconds','source_unchanged']},indent=2),flush=True)
if __name__=='__main__':main()
