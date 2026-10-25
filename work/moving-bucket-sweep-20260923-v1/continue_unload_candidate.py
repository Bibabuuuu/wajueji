"""Bounded diagnostic continuation: 120 opening frames + 60 hold, no qualification promotion."""
import argparse, json, hashlib, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import audit_layered_47_truck_physics_gate as gate

root=Path('/home/qunwang/wajueji')
work=Path(__file__).resolve().parent
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
loaded=root/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2'
cli=SimpleNamespace(name='replay')
args=SimpleNamespace(output=str(work/cli.name),repository_root=str(bundle/'repo'),truck_repository_root=str(root),
 config=str(bundle/'repo/configs/v4/layered-47-excavation-preflight-v1.json'),
 local_source_config=str(root/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json'),
 checkpoint=str(loaded/'loaded-handoff.npz'),metadata=str(loaded/'loaded-handoff.npz.json'),
 loaded_audit=str(loaded/'audit-v4/independent-restore-audit.json'),
 approach_report=str(root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json'),
 truck_config=str(root/'configs/v4/layered-47-truck-physics-v1.json'),device='cuda:0',
 mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),
 mechanism_header_dir=str(bundle/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
 controller_header_dir=str(bundle/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'),
 kinematics_python_dir=str(kin),kinematics_model=str(kin/'scene/input/model/A.xml'),kinematics_mesh_dir=str(kin/'scene/input/mesh'),
 inventory_contract='/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json',
 front_rim='/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz',cxx='g++')
from contextlib import ExitStack
import traceback
from layered_excavation_approach_planner import _load_kinematics
from soil_hencky_update_guard import installed_material_guard
import soil_hencky_update_candidate as hencky
import warp as wp
import newton

parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=36);parser.add_argument('--name',required=True);probe_cli=parser.parse_args()
out=work/probe_cli.name;out.mkdir(exist_ok=False)
source_run=root/'runs/layered-47-loaded-transport-v1/loaded-transport-20260923-215454-962382'
seed_dir=work/'full-887'
seed_report=json.loads((seed_dir/'report.json').read_text())
seed_path=Path(seed_report['terminal']['path'])
assert hashlib.sha256(seed_path.read_bytes()).hexdigest()==seed_report['terminal']['sha256']
seed_last=json.loads((seed_dir/'trace.ndjson').read_text().splitlines()[-1])
with np.load(seed_path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
metadata={'controller_state':{k:seed_last['after_state'][k] for k in ('controller_target','controller_target_velocity','bucket_integral_effort')}}
identity={'path':str(seed_path),'sha256':hashlib.sha256(seed_path.read_bytes()).hexdigest()}
trace=[{'applied_command_si':[0.,0.,0.,-.0675 if i<120 else 0.]} for i in range(180)]
assert probe_cli.frames==180
base=gate.motion.probe.p47.load_config(args.config,bundle/'repo')
base['input_checkpoint']=json.loads(Path(args.local_source_config).read_text())['input_checkpoint']
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
manifest=json.loads((bundle/'repo'/base['physical_geometry']['manifest_path']).read_text())
_,profile=gate.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in profile['layers']]
proxy=json.loads((bundle/'repo'/base['open_bucket_proxy']['path']).read_text())
approach=json.loads(Path(args.approach_report).read_text())
assembly,chain=_load_kinematics(args.kinematics_python_dir,args.kinematics_model,args.kinematics_mesh_dir)
projector=gate.manual.ContactCommandProjector(chain)
contract=json.loads(Path(args.inventory_contract).read_text())
front=np.load(args.front_rim)['front']
polygon=np.asarray(contract['bucket']['polygon_xz_m']); ylimits=np.asarray(contract['bucket']['y_limits_m'])
initial_pose=assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
local=(arrays['particle_q']-initial_pose[:3,3])@initial_pose[:3,:3]
in_cavity=np.zeros(len(local),dtype=bool)
with np.load(seed_dir/'tracking-selection.npz') as cohort:in_cavity[cohort['ids'][cohort['initial_cavity']]]=True
vertices=gate.manual._proxy_local_vertices(proxy)
near=np.all((local >= vertices.min(axis=0)-.3)&(local <= vertices.max(axis=0)+.3),axis=1)
# Include all terminal shell/near-bucket particles even if they started outside the initial halo.
with np.load(source_run/'final-000887.diagnostic.npz') as final:
 pose=assembly.assemble(final['mechanism_qpos'],include_meshes=False)[1]['bucket']
 end_local=(final['particle_q']-pose[:3,3])@pose[:3,:3]
 end_near=np.all((end_local>=vertices.min(axis=0)-.15)&(end_local<=vertices.max(axis=0)+.15),axis=1)
track_ids=np.flatnonzero(near|in_cavity|end_near)
np.savez(out/'tracking-selection.npz',ids=track_ids,initial_cavity=in_cavity[track_ids],initial_local=local[track_ids],initial_world=arrays['particle_q'][track_ids])
positions=np.lib.format.open_memmap(out/'post-projection-world.npy',mode='w+',dtype=np.float32,shape=(len(trace)+1,len(track_ids),3))
solver_positions=np.lib.format.open_memmap(out/'pre-projection-world.npy',mode='w+',dtype=np.float32,shape=(len(trace),len(track_ids),3))
positions[0]=arrays['particle_q'][track_ids]
poses=np.lib.format.open_memmap(out/'actual-bucket-poses.npy',mode='w+',dtype=np.float64,shape=(len(trace)+1,4,4));poses[0]=initial_pose
collider_poses=np.lib.format.open_memmap(out/'projection-bucket-transforms.npy',mode='w+',dtype=np.float32,shape=(len(trace),7))
files=[seed_path,seed_dir/'report.json',seed_dir/'trace.ndjson',Path(args.checkpoint),Path(args.metadata),Path(args.config),Path(args.truck_config),source_run/'physics.ndjson',source_run/'final-000887.diagnostic.npz',Path(__file__),root/'src/v4/run_layered_47_excavation_preflight.py',root/'src/v4/mpm_static_contact_ownership.py',Path(hencky.__file__),root/'src/v4/moving_bucket_sweep_candidate.py']
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
protocol={'frames_limit':len(trace),'wall_limit_seconds':600,'source_run':str(source_run),'diagnostic_continuation_seed':identity,'command_schedule':'120 frames bucket opening -0.0675 m/s through existing projector, then 60 zero-command frames','particle_count':len(arrays['particle_q']),'tracked_particle_count':len(track_ids),'tracked_initial_cavity_count':int(in_cavity[track_ids].sum()),'tracks_include_all_initial_cavity_particles':bool(in_cavity[track_ids].sum()==in_cavity.sum()),'selection':'initial proxy AABB plus 0.30m, union final proxy AABB plus 0.15m, union initial cavity','bindings':bindings,'physics_changes':'opt-in convex moving-bucket sweep; incremental velocity impulse routed current step plus one-step delayed actual-motion correction','resume_qualified':False,'training_eligible':False}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print('PROTOCOL',json.dumps({k:v for k,v in protocol.items() if k!='bindings'}),flush=True)
from moving_bucket_sweep_candidate import MovingBucketSweep
from run_layered_47_excavation_preflight import _bucket_transform
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
bridge=None;frame=0;completed=0;started=time.perf_counter();captured={};report={'status':'initializing','protocol':protocol};pending=seed_report['pending_final_velocity_reaction'];impulse_generated=np.asarray(pending['reaction_impulse_ns']).copy();impulse_routed=np.zeros(3);rows=[]
with ExitStack() as ctx:
 ctx.enter_context(gate.optimized_bridge(args.mechanism_header_dir,out/'optimized-build'))
 ctx.enter_context(gate.bucket_integral_bridge(True,out/'bucket-integral-build'))
 fixtures=ctx.enter_context(gate.truck.installed_candidate(args.truck_config,root))
 ctx.enter_context(hencky.installed_candidate())
 with gate.handoff.restored_sparse_grid_topology(arrays) as grid_restore:
  fixture,held,restored=gate.motion.audit.restore_candidate_runtime(args,base,bundle/'repo',records,arrays,proxy,approach['solver_profile'],out/'runtime-build',metadata['controller_state'],fixtures)
 bridge=gate.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
 guard=ctx.enter_context(installed_material_guard(fixture.solver));report['restore']=restored
 assert restored['exact'] and grid_restore['count']==1
 report['pending_reaction_restored']=pending==seed_report['pending_final_velocity_reaction']
 assert report['pending_reaction_restored']
 mesh=fixture.solver._mpm_model._collider_meshes[int(fixture.bucket_collider_ids[0])]
 sweep=MovingBucketSweep(mesh,fixture.state,fixture.particle_mass,base['mpm']['material']['friction'],proxy=proxy)
 prior_q=wp.empty_like(fixture.state.particle_q);after_mpm=wp.empty_like(fixture.state.particle_q)
 original_projection=fixture.solver.project_outside
 def observe_projection(state_in,state_out,dt,**kwargs):
  solver_positions[frame-1]=state_in.particle_q.numpy()[track_ids]
  collider_poses[frame-1]=state_in.body_q.numpy()[fixture.bucket_body_index]
  return original_projection(state_in,state_out,dt,**kwargs)
 fixture.solver.project_outside=observe_projection
 original_step=fixture.step
 def measured_step(dt,origins):
  global pending,impulse_generated,impulse_routed
  wp.copy(prior_q,fixture.state.particle_q)
  old_pose=_bucket_transform(bridge.state);captured['old_pose']=old_pose
  wrench,particles=original_step(dt,origins)
  started_sweep=time.perf_counter()
  correction=sweep.apply(fixture.state,prior_q,old_pose,old_pose,bridge.state['bucket_linear_velocity_world_m_s'],bridge.state['bucket_angular_velocity_world_rad_s'])
  correction['point_world_m']=list(wp.transform_get_translation(old_pose));captured['mpm_correction']=correction
  impulse_generated+=np.asarray(correction['reaction_impulse_ns'])
  increments=[correction]+([pending] if pending is not None else [])
  for inc in increments:
   impulse=np.asarray(inc['reaction_impulse_ns']);angular=np.asarray(inc['reaction_angular_impulse_nms'])
   shift=np.asarray(inc['point_world_m'])-np.asarray(origins['bucket'])
   wrench['bucket']['force']+=impulse/dt
   wrench['bucket']['torque']+=(angular+np.cross(shift,impulse))/dt
   impulse_routed+=impulse
  pending=None
  wp.copy(after_mpm,fixture.state.particle_q)
  captured['sweep_mpm_wall_seconds']=time.perf_counter()-started_sweep
  captured['wrench']=wrench['bucket']
  return wrench,particles
 fixture.step=measured_step
 def save(label):
  data=gate.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+completed/60)
  return gate.motion.probe.snapshots.write_rejected_diagnostic_atomic(out/f'{label}-{completed:06d}.npz',data,'moving sweep candidate diagnostic only; pending reaction separately in report; not resumable')
 try:
  with (out/'trace.ndjson').open('w') as log:
   for frame,source in enumerate(trace,1):
    if time.perf_counter()-started>600:raise TimeoutError('bounded sweep candidate wall budget')
    frame_start=time.perf_counter();bridge.command=projector.project(np.asarray(bridge.state['mechanism_qpos']),source['applied_command_si']);command_reason=projector.reason;guard.prepare_frame()
    row,_=gate.motion.probe.bench._advance_frame(fixture,bridge,base,frame)
    completed=frame
    new_pose=_bucket_transform(bridge.state)
    pending=sweep.apply(fixture.state,after_mpm,captured['old_pose'],new_pose,bridge.state['bucket_linear_velocity_world_m_s'],bridge.state['bucket_angular_velocity_world_rad_s'])
    pending['point_world_m']=list(wp.transform_get_translation(new_pose));impulse_generated+=np.asarray(pending['reaction_impulse_ns'])
    balance=impulse_generated-impulse_routed-np.asarray(pending['reaction_impulse_ns'])
    q=fixture.state.particle_q.numpy();qd=fixture.state.particle_qd.numpy();speed=np.linalg.norm(qd,axis=1)
    positions[frame]=q[track_ids]
    poses[frame]=assembly.assemble(np.asarray(bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
    material=guard.summarize(fixture.state)
    row.update(finite=bool(row['finite'] and np.isfinite(q).all() and np.isfinite(qd).all()),particle_max_speed_m_s=float(speed.max()),particle_p99_speed_m_s=float(np.quantile(speed,.99)),particle_minimum_z_m=float(q[:,2].min()),applied_command_si=bridge.command.tolist(),hencky_material_guard=material,joint_bounds_ok=projector.physical_bounds_ok(np.asarray(bridge.state['mechanism_qpos'])),bucket_mpm_wrench_finite=all(np.isfinite(v).all() for v in captured['wrench'].values()),rigid_contact_substeps=bridge.state['contact_substeps'])
    hard=gate.manual.hard_guard_checks(row,allow_truck=True)
    local=(q[track_ids]-poses[frame,:3,3])@poses[frame,:3,:3]
    retained=gate.loaded.cavity_mask(local,polygon,ylimits,front)
    record={'frame':frame,'requested_command_si':source['applied_command_si'],'applied_command_si':bridge.command.tolist(),'command_projection_reason':command_reason,'hard':hard,'material_safe':material['safe'],'wall_seconds':time.perf_counter()-frame_start,'bench_wall_seconds':row['wall_seconds'],'mpm_wall_seconds':row['mpm_wall_seconds'],'mujoco_wall_seconds':row['mujoco_wall_seconds'],'truck_mpm_contact_samples':row['truck_mpm_contact_samples'],'bucket_force':captured['wrench']['force'].tolist(),'mpm_sweep':captured['mpm_correction'],'actual_motion_sweep_pending_next_frame':pending,'incremental_impulse_balance_error_ns':balance.tolist(),'original_cavity_particles_still_in_cavity':int(np.count_nonzero(in_cavity[track_ids]&retained)),'particle_max_speed_m_s':row['particle_max_speed_m_s'],'particle_minimum_z_m':row['particle_minimum_z_m'],'after_state':gate.manual.compact_state(bridge.state),'native_projection':row['postsolve_projection']}
    rows.append(record);log.write(json.dumps(record)+'\n');log.flush()
    if not all(hard.values()) or not material['safe'] or np.max(np.abs(balance))>1e-7:raise RuntimeError('candidate guard or incremental impulse balance failed')
    if frame%60==0 or frame==1 or frame==len(trace):
     positions.flush();solver_positions.flush();poses.flush();collider_poses.flush()
     print('CANDIDATE',frame,'wall',round(time.perf_counter()-started,1),'retained',record['original_cavity_particles_still_in_cavity'],'hits',captured['mpm_correction']['hits'],pending['hits'],'force',round(np.linalg.norm(captured['wrench']['force'])),flush=True)
   final_info=save('terminal-candidate')
   report.update(status='candidate_recorded',frames=completed,terminal=final_info,all_hard_guards_passed=True)
 except BaseException as error:
  report.update(status='failed',frames=completed,error=repr(error),traceback=traceback.format_exc())
  try:report['recovery']=save('failed-candidate')
  except Exception as save_error:report['recovery_error']=repr(save_error)
  raise
 finally:
  report.update(wall_seconds=time.perf_counter()-started,pending_final_velocity_reaction=pending,total_generated_reaction_ns=impulse_generated.tolist(),total_routed_reaction_ns=impulse_routed.tolist(),native_projection_impulses_still_unattributed=True,position_corrections_have_no_additional_impulse_assumption=True)
  report['bindings_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in bindings.items())
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  positions.flush();solver_positions.flush();poses.flush();collider_poses.flush()
  bridge.close()
print('DONE',completed,report['status'],flush=True)
