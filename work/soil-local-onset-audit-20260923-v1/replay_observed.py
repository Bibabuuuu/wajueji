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
import warp.fem as fem
from observer import observe_update, query
import run_layered_47_sparse_manual_contact_episode as manual
import layered_mound_checkpoint_runtime as runtime

root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent
baseline=root/'work/manual-soil-loss-replay-20260923-v1'
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1');repo=bundle/'repo'
protocol=json.loads((out/'protocol.json').read_text())
for p,h in protocol['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
trace=[json.loads(l) for l in (run/'physics.ndjson').read_text().splitlines()][:1193]
arrays=dict(np.load(start/'approach-handoff.npz'))
base=json.loads((repo/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text())
_,identity=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text())
approach=json.loads((start/'report.json').read_text())
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
args=argparse.Namespace(device='cuda:0',cxx='g++',mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
_,chain=manual._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
projector=manual.ContactCommandProjector(chain)
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
 selected=np.load(out/'common-hole-initial-particle-ids.npy')
 mapping=np.full(len(arrays['particle_q']),-1,dtype=np.int32);mapping[selected]=np.arange(len(selected),dtype=np.int32)
 mapping_gpu=wp.array(mapping,dtype=int);ids_gpu=wp.array(selected.astype(np.int32),dtype=int)
 observed=wp.zeros((len(selected),20),dtype=float)
 sdf_out=wp.zeros((len(selected),7),dtype=float);hit_out=wp.zeros(len(selected),dtype=int)
 material_frames=[];cohort_frames=[];sdf_frames=[];hit_frames=[];observation_frames=[]
 (out/'collider-map.json').write_text(json.dumps({'body_index':fixture.solver._mpm_model.collider.collider_body_index.numpy().tolist(),'bucket_body':fixture.bucket_body_index,'body_names':fixture.body_indices},indent=2)+'\n')
 real_update=fixture.solver._update_particles
 def observed_update(state_in,state_out,dt,pic,scratch):
  if frame>=900:
   observed.zero_()
   fem.interpolate(observe_update,at=pic,values={'mapping':mapping_gpu,'volume':fixture.solver._mpm_model.material_particle_volume,'output':observed},fields={'elastic':scratch.elastic_strain_delta_field,'plastic':scratch.plastic_strain_delta_field,'velocity':scratch.velocity_field},temporary_store=fixture.solver.temporary_store)
   material_frames.append(observed.numpy())
  return real_update(state_in,state_out,dt,pic,scratch)
 fixture.solver._update_particles=observed_update
 real_projection=fixture.solver.project_outside
 def observe_projection(state_in,state_out,dt,gap=None):
  if frame>=900:
   collider=fixture.solver._mpm_model.collider
   previous_gap=collider.query_max_dist
   try:
    if gap is not None:collider.query_max_dist=gap
    mode=fixture.solver.collider_velocity_mode
    wp.launch(query,dim=len(selected),inputs=[state_in.particle_q,ids_gpu,collider,state_in.body_q,state_in.body_qd if mode=='forward' else None,fixture.solver._last_step_data.body_q_prev if mode=='backward' else None,dt,sdf_out,hit_out])
    sdf_frames.append(sdf_out.numpy());hit_frames.append(hit_out.numpy())
   finally:collider.query_max_dist=previous_gap
  return real_projection(state_in,state_out,dt,gap=gap)
 fixture.solver.project_outside=observe_projection
 def capture():return manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+frame/60)
 def save(label):return manual.motion.probe.snapshots.write_rejected_diagnostic_atomic(out/f'{label}-{frame:06d}.npz',capture(),'read-only observer replay; retains invalid Jp; never resume/train')
 try:
  with (out/'trace.ndjson').open('w') as log:
   for source in trace:
    frame=source['frame'];bridge.command=np.asarray(source['applied_command_si'])
    row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,frame)
    compact=manual.compact_state(bridge.state)
    if frame<=1193:
     assert compact==source['after_state'],('state_prefix_mismatch',frame)
     for k in ['particle_max_speed_m_s','particle_p99_speed_m_s','particle_minimum_z_m','postsolve_projection','closure_m','bucket_mpm_contact_samples']:
      assert row[k]==source[k],('metric_prefix_mismatch',frame,k)
    row['known_Jp_invalid_diagnostic_only']=frame>=540
    row['after_state']=compact
    if frame>=900:
     a=capture()
     row['nonfinite_fields']={k:int((~np.isfinite(v)).sum()) for k,v in a.items() if v.dtype.kind in 'fc' and not np.isfinite(v).all()}
     observation_frames.append(frame)
     cohort_frames.append({k:a[k][selected] for k in ['particle_q','particle_qd','particle_elastic_strain','particle_Jp','particle_stress']})
     sv=np.linalg.svd(a['particle_elastic_strain'][selected].astype(float),compute_uv=False)[:,0]
     row['cohort_elastic_singular_max_median_max']=[float(np.median(sv)),float(sv.max())]
     row['cohort_elastic_over10']=int((sv>10).sum())
     row['cohort_elastic_over1000']=int((sv>1000).sum())
    log.write(json.dumps(row)+'\n');log.flush()
    assert row['finite'] and manual.motion.probe.bench._finite_tree(bridge.state),'position_velocity_nonfinite'
    assert projector.physical_bounds_ok(bridge.state['mechanism_qpos']) and row['closure_m']<=.001,'mechanism_guard'
    assert row['forbidden_contact_count']==0,'forbidden_contact'
    if frame==1000:
     expected=np.load(baseline/'diagnostic-001000.npz');a=capture()
     prefix_arrays_exact={k:bool(np.array_equal(a[k],expected[k])) for k in expected.files}
     assert all(prefix_arrays_exact.values()),prefix_arrays_exact
     (out/'prefix-verification.json').write_text(json.dumps(prefix_arrays_exact,indent=2)+'\n')
    if frame==1193:save('observed-original')
    if frame%60==0:print('FRAME',frame,'observing',frame>=900,flush=True)
  np.savez_compressed(out/'cohort-history.npz',frames=np.asarray(observation_frames),ids=selected,material_update=np.asarray(material_frames),sdf=np.asarray(sdf_frames),collider=np.asarray(hit_frames),**{k:np.asarray([r[k] for r in cohort_frames]) for k in cohort_frames[0]})
  expected=np.load(baseline/'diagnostic-001193.npz');final=capture();final_exact={k:bool(np.array_equal(final[k],expected[k])) for k in expected.files};assert all(final_exact.values())
  result={'final1193_arrays_exact':final_exact,'status':'diagnostic_completed','frames':frame,'recorded_frames_exact':1193,'prefix_arrays_exact':prefix_arrays_exact,'restore_exact':restored['exact'],'bindings_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['bindings'].items()),'physics_qualified':False,'training_eligible':False,'resume_qualified':False}
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',frame,flush=True)
 except BaseException as e:
  try:save('failure')
  finally:(out/'failure.json').write_text(json.dumps({'frame':frame,'error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n')
  raise
 finally:
  bridge.close();runtime.render_capacity_bridge=original_render
