"""Diagnostic intervention after an exact recorded 1000-frame prefix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
import warp as wp
import newton
import run_layered_47_sparse_manual_contact_episode as manual
import layered_mound_checkpoint_runtime as runtime

root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent
baseline=root/'work/manual-soil-loss-replay-20260923-v1'
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1');repo=bundle/'repo'
protocol=json.loads((out/'protocol.json').read_text())
for p,h in protocol['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
trace=[json.loads(l) for l in (run/'physics.ndjson').read_text().splitlines()]
arrays=dict(np.load(start/'approach-handoff.npz'))
base=json.loads((repo/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text())
_,identity=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text())
approach=json.loads((start/'report.json').read_text())
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
args=argparse.Namespace(device='cuda:0',cxx='g++',mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
work=out/f'build-{os.getpid()}';work.mkdir(exist_ok=False)
bridge=None;frame=0;prefix_arrays_exact=None
with manual.optimized_bridge(args.mechanism_header_dir,out/f'optimization-{os.getpid()}'):
 original_render=runtime.render_capacity_bridge
 frozen=(run/'bucket-integral-build/capacity-bridge.cc').read_text()
 def render_frozen(*args):
  _,metadata=original_render(*args)
  return frozen,{**metadata,'diagnostic_frozen_original_bridge':True}
 runtime.render_capacity_bridge=render_frozen
 fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],work)
 bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
 real_projection=fixture.solver.project_outside
 def project_branch(state_in,state_out,dt,gap=None):
  if frame<=1000:return real_projection(state_in,state_out,dt,gap=gap)
  assert state_in is state_out, 'in_place_intervention_only'
 fixture.solver.project_outside=project_branch
 def capture():return manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+frame/60)
 def save(label):return manual.motion.probe.snapshots.write_rejected_diagnostic_atomic(out/f'{label}-{frame:06d}.npz',capture(),'causal diagnostic only; retains existing invalid Jp; never resume/train')
 try:
  with (out/'trace.ndjson').open('w') as log:
   for source in trace:
    frame=source['frame'];bridge.command=np.asarray(source['applied_command_si'])
    row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,frame)
    compact=manual.compact_state(bridge.state)
    if frame<=1000:
     assert compact==source['after_state'],('state_prefix_mismatch',frame)
     for k in ['particle_max_speed_m_s','particle_p99_speed_m_s','particle_minimum_z_m','postsolve_projection','closure_m','bucket_mpm_contact_samples']:
      assert row[k]==source[k],('metric_prefix_mismatch',frame,k)
    else:
     row['postsolve_projection']['applied']=False
     row['postsolve_projection']['diagnostic_intervention']='disabled_after_exact_frame_1000'
    row['known_Jp_invalid_diagnostic_only']=frame>=540
    row['after_state']=compact
    if frame%30==0 or frame in (1000,1193,1329):
     a=capture();row['nonfinite_fields']={k:int((~np.isfinite(v)).sum()) for k,v in a.items() if v.dtype.kind in 'fc' and not np.isfinite(v).all()}
    log.write(json.dumps(row)+'\n');log.flush()
    assert row['finite'] and manual.motion.probe.bench._finite_tree(bridge.state),'position_velocity_nonfinite'
    assert row['joint_bounds_ok'] and row['closure_m']<=.001,'mechanism_guard'
    assert row['forbidden_contact_count']==0,'forbidden_contact'
    if frame==1000:
     expected=np.load(baseline/'diagnostic-001000.npz');a=capture()
     prefix_arrays_exact={k:bool(np.array_equal(a[k],expected[k])) for k in expected.files}
     assert all(prefix_arrays_exact.values()),prefix_arrays_exact
     (out/'prefix-verification.json').write_text(json.dumps(prefix_arrays_exact,indent=2)+'\n')
    if frame in (1193,1329):save('candidate')
    if frame%60==0:print('FRAME',frame,'projection',frame<=1000,flush=True)
  result={'status':'diagnostic_completed','frames':frame,'prefix_frames_exact':1000,'prefix_arrays_exact':prefix_arrays_exact,'restore_exact':restored['exact'],'bindings_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['bindings'].items()),'physics_qualified':False,'training_eligible':False,'resume_qualified':False}
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',frame,flush=True)
 except BaseException as e:
  try:save('failure')
  finally:(out/'failure.json').write_text(json.dumps({'frame':frame,'error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n')
  raise
 finally:
  bridge.close();runtime.render_capacity_bridge=original_render
