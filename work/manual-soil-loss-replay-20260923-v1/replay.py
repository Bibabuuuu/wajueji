"""Read-only replay of original manual commands; inspect physics and display lag."""
import argparse,json,os,time,sys,traceback
from pathlib import Path
import numpy as np
import warp as wp
import newton
import run_layered_47_sparse_manual_contact_episode as manual
import layered_mound_checkpoint_runtime as runtime
import hashlib
root=Path('/home/qunwang/wajueji')
out=Path(__file__).resolve().parent
diagnosis=root/'work/surface-render-diagnosis-20260923'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
repo=bundle/'repo'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279'
trace=[json.loads(x) for x in (run/'physics.ndjson').read_text().splitlines()]
assert len(trace)==1329
arrays=dict(np.load(start/'approach-handoff.npz'))


base=json.loads((repo/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text())
_,identity=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text())
approach=json.loads((start/'report.json').read_text())
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
args=argparse.Namespace(device='cuda:0',cxx='g++',
    mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),
    mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
    controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))

protocol=json.loads((out/'protocol.json').read_text())
for path,digest in protocol['bindings'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
work=out/f'replay-build-{os.getpid()}';work.mkdir(exist_ok=False)
bridge=None;frame=0;first_bad=None;exact=True
with manual.optimized_bridge(args.mechanism_header_dir,out/f'replay-optimization-{os.getpid()}'):
 original_render=runtime.render_capacity_bridge
 frozen_code=(run/'bucket-integral-build/capacity-bridge.cc').read_text()
 def render_frozen(*arguments):
  _,manifest=original_render(*arguments)
  return frozen_code,{**manifest,'diagnostic_replay_frozen_manual_bridge_sha256':hashlib.sha256(frozen_code.encode()).hexdigest()}
 runtime.render_capacity_bridge=render_frozen
 fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],work)
 bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
 original_projection=fixture.solver.project_outside
 projection_record={}
 def observe_projection(state_in,state_out,dt,gap=None):
  before=state_in.particle_q.numpy()
  original_projection(state_in,state_out,dt,gap=gap)
  after=state_out.particle_q.numpy();delta=np.linalg.norm(after-before,axis=1)
  ids=np.argsort(delta)[-20:][::-1]
  projection_record.clear();projection_record.update(ids=ids.tolist(),before=before[ids].tolist(),after=after[ids].tolist(),delta_m=delta[ids].tolist())
 fixture.solver.project_outside=observe_projection
 def save_raw(label):
  state=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+frame/60)
  bad={k:int(np.count_nonzero(~np.isfinite(v))) for k,v in state.items() if v.dtype.kind in 'fc' and not np.isfinite(v).all()}
  p=out/f'{label}-{frame:06}.npz'
  info=manual.motion.probe.snapshots.write_rejected_diagnostic_atomic(p,state,'diagnostic replay only; nonfinite='+str(bad))
  return info
 try:
  with (out/'trace.ndjson').open('w') as journal,(out/'projection-top20.ndjson').open('w') as projections:
   for i,source_row in enumerate(trace,1):
    frame=i;bridge.command=np.asarray(source_row['applied_command_si'])
    row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,i)
    state=manual.compact_state(bridge.state)
    comparisons={'complete_compact_state':state==source_row['after_state']}
    for k in ['particle_max_speed_m_s','particle_p99_speed_m_s','particle_minimum_z_m','postsolve_projection','closure_m','bucket_mpm_contact_samples']:
     comparisons[k]=row[k]==source_row[k]
    if not all(comparisons.values()):
     exact=False
     (out/'mismatch.json').write_text(json.dumps({'frame':i,'checks':comparisons,'current':state,'recorded':source_row['after_state']},indent=2))
     raise RuntimeError('replay mismatch before diagnostic claim')
    record={'frame':i,'exact':comparisons,'particle_max_speed_m_s':row['particle_max_speed_m_s'],'particle_p99_speed_m_s':row['particle_p99_speed_m_s'],'minimum_z_m':row['particle_minimum_z_m'],'wall_seconds':row['wall_seconds'],'postsolve_projection':row['postsolve_projection']}
    if i==1 or i%30==0 or i==len(trace):
     captured=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+i/60)
     bad={k:int(np.count_nonzero(~np.isfinite(v))) for k,v in captured.items() if v.dtype.kind in 'fc' and not np.isfinite(v).all()}
     record['nonfinite_fields']=bad
     if bad and first_bad is None:
      first_bad={'observed_frame':i,'previous_audit_frame':max(1,i-30),'fields':bad}
      save_raw('first-nonfinite-diagnostic')
      print('FIRST_NONFINITE',json.dumps(first_bad),flush=True)
    journal.write(json.dumps(record)+'\n');journal.flush()
    projections.write(json.dumps({'frame':i,**projection_record})+'\n');projections.flush()
    if i in (277,600,800,1000,1193,1329):save_raw('diagnostic')
    if i%60==0:print('REPLAY',i,'exact',all(comparisons.values()),'first_bad',first_bad,flush=True)
   result={'frames':frame,'all_recorded_states_and_metrics_exact':exact,'restore':restored,'first_nonfinite_audit':first_bad,'source_bindings_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['bindings'].items()),'original_run_saved_snapshot':False,'reconstructed_diagnostic_only':True,'training_eligible':False,'resume_qualified':False}
   (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',frame,flush=True)
 except BaseException as error:
  try:save_raw('failed-replay-raw')
  finally:(out/'failure.json').write_text(json.dumps({'frame':frame,'error':repr(error),'traceback':traceback.format_exc(),'exact_so_far':exact},indent=2)+'\n')
  raise
 finally:
  bridge.close()
  runtime.render_capacity_bridge=original_render
