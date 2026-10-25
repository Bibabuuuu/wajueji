"""Bounded matched-prefix curl and higher-inventory carry experiments."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import warp as wp
import newton
import run_layered_47_sparse_manual_contact_episode as manual
from layered_47_active_curl_candidate import active_curl_bridge
from moving_bucket_sweep_runtime import MovingBucketSweepRuntime
from soil_hencky_update_guard import installed_material_guard
import soil_hencky_update_candidate as hencky
import layered_47_loaded_handoff_checkpoint as loaded

root=Path('/home/qunwang/wajueji');work=Path(__file__).resolve().parent
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1');repo=bundle/'repo'
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['baseline','candidate']);parser.add_argument('--case',choices=['recorded','loaded'],default='recorded');cli=parser.parse_args()
out=work/(cli.case+'-'+cli.mode);out.mkdir(exist_ok=False)
source=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-235749-1040071'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
original=[json.loads(line) for line in (source/'physics.ndjson').read_text().splitlines()]
source_report=json.loads((source/'report.json').read_text())
checkpoint=start/'approach-handoff.npz';controller=None
if cli.case=='loaded':
 checkpoint=root/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2/loaded-handoff.npz'
 arrays,metadata,identity=loaded.load_checkpoint_readonly(checkpoint,Path(str(checkpoint)+'.json'))
 controller=metadata['controller_state']
 commands=[([0,0,0,.02],'loaded_curl') for _ in range(120)]
 activation=0.0
else:
 arrays=dict(np.load(checkpoint));commands=[(r['requested_command_si'],'recorded') for r in original[:1100]];activation=10.0
commands += [([0,0,0,0],'hold') for _ in range(60)]+[([0,0,0,-.02],'reverse') for _ in range(30)]+[([0,0,0,0],'stop') for _ in range(60)]
base=manual.motion.probe.p47.load_config(repo/'configs/v4/layered-47-excavation-preflight-v1.json',repo)
base['input_checkpoint']=json.loads((root/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json').read_text())['input_checkpoint']
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text());_,profile=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in profile['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text());approach=json.loads((start/'report.json').read_text())
args=argparse.Namespace(device='cuda:0',cxx='g++',mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))
assembly,chain=manual._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh');projector=manual.ContactCommandProjector(chain)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bindings=dict(source_report['input_bindings'])
for p in [checkpoint,Path(str(checkpoint)+'.json'),source/'physics.ndjson',Path(__file__),work/'protocol.json',root/'src/v4/bucket_active_curl_candidate.h',root/'src/v4/layered_47_active_curl_candidate.py']:
 bindings[str(p)]=sha(p)
assert all(sha(p)==v for p,v in bindings.items()),'source binding changed before experiment'
(out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
started=time.perf_counter();completed=0;bridge=None;fixture=None;sweep=None;rows=[]
report={'status':'initializing','case':cli.case,'mode':cli.mode,'training_eligible':False,'prefix_original_exact':True,'source_bindings':bindings}
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
with ExitStack() as ctx:
 try:
  ctx.enter_context(manual.optimized_bridge(args.mechanism_header_dir,out/'optimized-build'))
  ctx.enter_context(manual.bucket_integral_bridge(True,out/'bucket-integral-build'))
  policy=ctx.enter_context(active_curl_bridge(cli.mode=='candidate',out/'active-curl-build',activation_time_s=activation,initialize_from_d120=cli.case=='loaded'))
  ctx.enter_context(hencky.installed_candidate())
  with ExitStack() as restoration:
   if cli.case=='loaded':restoration.enter_context(loaded.restored_sparse_grid_topology(arrays))
   fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],out/'runtime-build',controller)
  bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
  report['restore']=restored;assert restored['exact'];report['active_curl_policy']=policy.copy()
  material=ctx.enter_context(installed_material_guard(fixture.solver))
  sweep=MovingBucketSweepRuntime(fixture,bridge,proxy,base['mpm']['material']['friction'])
  captured={};original_step=fixture.step
  def step_mpm(dt,origins):
   w,p=sweep.step_mpm(original_step,dt,origins);captured['wrench']=w['bucket'];return w,p
  fixture.step=step_mpm
  def save(label):
   data=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+completed/60)
   np.savez_compressed(out/f'{label}-{completed:06d}.npz',**data)
   state={'controller':manual.compact_state(bridge.state),'supplemental_effort':bridge.state.get('bucket_active_curl_effort',[0]*4),'sweep':sweep.diagnostic_state(),'resume_qualified':False}
   (out/f'{label}-{completed:06d}.json').write_text(json.dumps(state,indent=2)+'\n')
  with (out/'trace.ndjson').open('w') as journal:
   for frame,(requested,phase) in enumerate(commands,1):
    if time.perf_counter()-started>600:raise TimeoutError('bounded group budget')
    before=manual.compact_state(bridge.state)
    # Reapply the same operator requests through the unchanged physical projector.
    bridge.command=projector.project(bridge.state['mechanism_qpos'],np.asarray(requested))
    material.prepare_frame();row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,frame);completed=frame
    sweep_row=sweep.finish_frame();q=fixture.state.particle_q.numpy();v=fixture.state.particle_qd.numpy();speed=np.linalg.norm(v,axis=1)
    row.update(finite=bool(row['finite'] and np.isfinite(q).all() and np.isfinite(v).all() and manual.motion.probe.bench._finite_tree(bridge.state)),particle_max_speed_m_s=float(speed.max()),particle_p99_speed_m_s=float(np.quantile(speed,.99)),particle_minimum_z_m=float(q[:,2].min()),moving_bucket_sweep=sweep_row,hencky_material_guard=material.summarize(fixture.state),joint_bounds_ok=projector.physical_bounds_ok(np.asarray(bridge.state['mechanism_qpos'])),applied_command_si=bridge.command.tolist(),rigid_contact_substeps=bridge.state['contact_substeps'],**manual.contact.wrench_summary(captured['wrench']))
    row.update(axis_velocity=np.asarray(bridge.state["mechanism_qvel"])[manual.motion.AXES-1].tolist(),bucket_linear_speed_m_s=float(np.linalg.norm(bridge.state["bucket_linear_velocity_world_m_s"])),bucket_angular_speed_rad_s=float(np.linalg.norm(bridge.state["bucket_angular_velocity_world_rad_s"])))
    guards=manual.hard_guard_checks(row)
    after=manual.compact_state(bridge.state)
    record={'frame':frame,'phase':phase,'requested_command':requested,'command':bridge.command.tolist(),'projection_reason':projector.reason,'before_state':before,'after_state':after,'supplemental_effort_N':bridge.state.get('bucket_active_curl_effort',[0]*4)[3],'external_drive_load':bridge.state['equivalent_drive_load'],'actuator_force':bridge.state['actuator_force'],'hard':guards,'stopped':manual.motion.stopped(row),'particle_max_speed_m_s':row['particle_max_speed_m_s'],'particle_minimum_z_m':row['particle_minimum_z_m'],'closure_m':row['closure_m'],'sweep':sweep_row,'wall_seconds':row['wall_seconds']}
    journal.write(json.dumps(record)+'\n');journal.flush();rows.append(record)
    if cli.case=='recorded' and frame<=600:
     exact=after==original[frame-1]['after_state'] and record['command']==original[frame-1]['applied_command_si']
     report['prefix_original_exact'] &= exact
     if not exact:raise RuntimeError('original compact prefix differs at '+str(frame))
    if not all(guards.values()):raise RuntimeError('hard guard failed '+str(guards))
    if frame in ({600,900,1100,1160,1250} if cli.case=='recorded' else {120,180,270}):save('diagnostic')
    if frame%60==0 or frame==1:print(cli.case,cli.mode,frame,'q',after['mechanism_qpos'][15],'extra',record['supplemental_effort_N'],'wall',round(time.perf_counter()-started,1),flush=True)
  report.update(status='recorded',frames=completed,all_hard=True)
 except BaseException as error:
  report.update(status='failed',frames=completed,error=repr(error),traceback=traceback.format_exc())
  if bridge is not None:
   try:save('recovery')
   except BaseException as save_error:report['recovery_error']=repr(save_error)
  raise
 finally:
  report.update(wall_seconds=time.perf_counter()-started,bindings_unchanged=all(sha(p)==v for p,v in bindings.items()))
  if sweep:report['sweep_state']=sweep.diagnostic_state()
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  if bridge:bridge.close()
print('DONE',report['status'],completed,flush=True)
