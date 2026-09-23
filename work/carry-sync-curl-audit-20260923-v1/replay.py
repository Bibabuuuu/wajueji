"""Read-only replay of original manual commands; inspect physics and display lag."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import warp as wp
import newton
import run_layered_47_sparse_manual_contact_episode as manual
root=Path('/home/qunwang/wajueji')
out=root/'work/carry-sync-curl-audit-20260923-v1'
diagnosis=root/'work/surface-render-diagnosis-20260923'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
repo=bundle/'repo'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-163839-751464'
trace=[json.loads(x) for x in (run/'physics.ndjson').read_text().splitlines()]
assert len(trace)==1370
arrays=dict(np.load(start/'approach-handoff.npz'))
saved=dict(np.load(run/'final-001370.diagnostic.npz'))

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

kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,chain=manual._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
final_pose=assembly.assemble(saved['mechanism_qpos'],include_meshes=False)[1]['bucket']
local=(saved['particle_q']-final_pose[:3,3])@final_pose[:3,:3]
# Broad final bucket neighbourhood, includes cavity and heap. Diagnostic cohort only.
selected=np.flatnonzero(np.all((local>[-.5,-.5,-1.35])&(local<[1.,.5,.25]),axis=1))
np.save(out/'selected-particle-ids.npy',selected)
projector=manual.ContactCommandProjector(chain)
binding_paths=[start/'approach-handoff.npz',start/'approach-handoff.npz.json',run/'physics.ndjson',run/'final-001370.diagnostic.npz',repo/base['open_bucket_proxy']['path'],Path(__file__),root/'src/v4/mixed_material_r1.py',root/'src/v4/run_layered_47_excavation_preflight.py',root/'src/v4/run_layered_47_sparse_manual_contact_episode.py',root/'src/v4/soil_surface.py']
bindings={str(p):manual.motion.probe.bench.sha256(p) for p in binding_paths}
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
work=out/f'replay-build-{os.getpid()}';work.mkdir(exist_ok=False)
with manual.optimized_bridge(args.mechanism_header_dir,out/f'replay-optimization-{os.getpid()}'):
 fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],work)
 bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
 original_step=fixture.step;original_projection=fixture.solver.project_outside
 captured={};events=[];positions=[];poses=[];bucket_display_lag=[]
 frame=[0]
 def projection(state_in,state_out,dt,gap=None):
  before=state_in.particle_q.numpy()
  original_projection(state_in,state_out,dt,gap=gap)
  after=state_out.particle_q.numpy()
  delta=np.linalg.norm(after-before,axis=1)
  captured['projection_global_max_m']=float(delta.max())
  captured['projection_cohort_max_m']=float(delta[selected].max())
  captured['projection_cohort_gt_1cm']=int((delta[selected]>.01).sum())
  captured['projection_cohort_worst_id']=int(selected[np.argmax(delta[selected])])
  captured['pre_projection_cohort_q']=before[selected].copy()
 def step(dt,origins):
  measured,particles=original_step(dt,origins);captured['wrenches']=measured;return measured,particles
 fixture.step=step;fixture.solver.project_outside=projection
 previous=arrays['particle_q'][selected].copy();previous_pose=assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket'];mesh_pose=previous_pose
 try:
  with (out/'trace.ndjson').open('w') as journal:
   for i,source_row in enumerate(trace,1):
    frame[0]=i;bridge.command=np.asarray(source_row['applied_command_si'])
    row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,i)
    row.update(applied_command_si=bridge.command.tolist(),joint_bounds_ok=projector.physical_bounds_ok(np.asarray(bridge.state['mechanism_qpos'])),rigid_contact_substeps=bridge.state.get('contact_substeps',{}),**manual.contact.wrench_summary(captured['wrenches']['bucket']))
    guards=manual.hard_guard_checks(row);assert all(guards.values()),guards
    current=fixture.state.particle_q.numpy()[selected];pose=assembly.assemble(np.asarray(bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
    delta=current-previous
    predicted=(previous-previous_pose[:3,3])@previous_pose[:3,:3]@pose[:3,:3].T+pose[:3,3]
    if i%6==0:mesh_pose=pose
    display_prediction=(current-pose[:3,3])@pose[:3,:3]@mesh_pose[:3,:3].T+mesh_pose[:3,3]
    record={'frame':i,'hard_guards':guards,'command':bridge.command.tolist(),'controller_target':bridge.state['controller_target'],'controller_target_velocity':bridge.state['controller_target_velocity'],'axis_positions':np.asarray(bridge.state['mechanism_qpos'])[manual.motion.AXES].tolist(),'axis_velocity':np.asarray(bridge.state['mechanism_qvel'])[manual.motion.AXES-1].tolist(),'cohort_max_step_m':float(np.linalg.norm(delta,axis=1).max()),'cohort_p99_step_m':float(np.quantile(np.linalg.norm(delta,axis=1),.99)),'cohort_max_up_step_m':float(delta[:,2].max()),'cohort_p99_rigid_carry_error_m':float(np.quantile(np.linalg.norm(current-predicted,axis=1),.99)),'display_bucket_motion_staleness_p99_m':float(np.quantile(np.linalg.norm(display_prediction-current,axis=1),.99)),'bucket_force_world_n':row['bucket_mpm_force_world_n'],'equivalent_drive_load':bridge.state['equivalent_drive_load'],'actuator_force':bridge.state['actuator_force'],'wall_seconds':row['wall_seconds'],**{k:v for k,v in captured.items() if k.startswith('projection_')}}
    journal.write(json.dumps(record)+'\n');journal.flush();events.append(record)
    if i>=1010:positions.append(current.copy());poses.append(pose.copy())
    if i in (480,1013):
     state=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+i/60)
     np.savez_compressed(out/f'diagnostic-{i:06}.npz',**state)
     (out/f'bridge-state-{i:06}.json').write_text(json.dumps(bridge.state))
    previous=current;previous_pose=pose
    if i%60==0:print('REPLAY',i,'cohort_projection',record['projection_cohort_max_m'],'target',record['controller_target'][3],flush=True)
   final=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+len(trace)/60)
   np.savez_compressed(out/'final-001370.diagnostic.npz',**final)
   np.savez_compressed(out/'lift-cohort.npz',q=np.array(positions),poses=np.array(poses),ids=selected,first_frame=1010)
   comparisons={k:bool(np.array_equal(v,saved[k])) for k,v in final.items() if k in saved and v.dtype.kind in 'fiu'}
   result={'frames':len(trace),'cohort_count':len(selected),'restore':restored,'comparisons':comparisons,'all_numeric_arrays_exact':all(comparisons.values()),'source_bindings':bindings,'bindings_unchanged':all(manual.motion.probe.bench.sha256(p)==h for p,h in bindings.items()),'training_eligible':False}
   (out/'report.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('restore','source_bindings')}),flush=True)
 finally:bridge.close()
