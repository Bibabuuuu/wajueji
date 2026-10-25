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
mode=sys.argv[1]; assert mode in ('baseline','candidate')
parent=Path(__file__).parent;out=parent/mode;out.mkdir(exist_ok=False)
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
trace += [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(120)] + [{'applied_command_si':[0.,0.,0.,-.02]} for _ in range(30)] + [{'applied_command_si':[0.,0.,0.,0.]} for _ in range(60)]
assert len(trace)==1580
work=out/f'replay-build-{os.getpid()}';work.mkdir(exist_ok=False)
with manual.optimized_bridge(args.mechanism_header_dir,out/f'replay-optimization-{os.getpid()}'):

 original_render=runtime.render_capacity_bridge
 def render_candidate(*arguments):
  source,manifest=original_render(*arguments)
  decl='    std::array<double, 4> unloaded_feedforward{};'
  assert source.count(decl)==1
  source=source.replace(decl,decl+'\n    std::array<double, 4> external_load_compensation{};\n    double bucket_compensation_filter = 0.0;')
  emit='      array("unloaded_feedforward", unloaded_feedforward.data(), 4);'
  assert source.count(emit)==1
  source=source.replace(emit,emit+'\n      array("external_load_compensation", external_load_compensation.data(), 4);')
  before='        const double before = data->time;\n        mj_step(model.get(), data.get());'
  assert source.count(before)==1
  added="""        // Isolated A/B: activate only after the unchanged 480-frame prefix.
        // Compensate only half of attributed MPM bucket load. Native rock
        // contact reactions and unowned endpoint projection are not included.
        if (data->time >= 8.0 - 1e-9) {
          const double target_comp = std::clamp(-0.5 * equivalent_drive_load[3], -60000.0, 60000.0);
          const double alpha = model->opt.timestep / (0.1 + model->opt.timestep);
          const double filtered = bucket_compensation_filter + alpha * (target_comp - bucket_compensation_filter);
          const double max_change = 150000.0 * model->opt.timestep;
          bucket_compensation_filter += std::clamp(filtered - bucket_compensation_filter, -max_change, max_change);
          external_load_compensation[3] = bucket_compensation_filter;
          const int actuator = controller.hold_actuator[3];
          data->ctrl[actuator] += external_load_compensation[3] / model->actuator_gainprm[mjNGAIN * actuator];
        }
"""
  source=source.replace(before,added+before)
  (out/'candidate-bridge.cc').write_text(source)
  candidate={'isolated_candidate':True,'deployed':False,'gain':.5,'clamp_N':60000.,'slew_N_per_s':150000.,'filter_tau_s':.1,'activation_native_time_s':8.,'axis':'bucket_only','source_sha256':hashlib.sha256(source.encode()).hexdigest()}
  (out/'candidate-policy.json').write_text(json.dumps(candidate,indent=2))
  return source,{**manifest,'isolated_compensation_candidate':candidate}
 if mode=='candidate': runtime.render_capacity_bridge=render_candidate
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
    requested=bridge.command.copy()
    if i>1370:
     bridge.command=projector.project(np.asarray(bridge.state['mechanism_qpos']),requested)
     assert np.array_equal(bridge.command,requested),projector.reason
    row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,i)
    row.update(applied_command_si=bridge.command.tolist(),joint_bounds_ok=projector.physical_bounds_ok(np.asarray(bridge.state['mechanism_qpos'])),rigid_contact_substeps=bridge.state.get('contact_substeps',{}),**manual.contact.wrench_summary(captured['wrenches']['bucket']))
    guards=manual.hard_guard_checks(row)
    if not all(guards.values()):
     (out/'failed-guards.json').write_text(json.dumps({'frame':i,'guards':guards,'row':row},default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),indent=2))
     raise RuntimeError(str(guards))
    current=fixture.state.particle_q.numpy()[selected];pose=assembly.assemble(np.asarray(bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
    delta=current-previous
    predicted=(previous-previous_pose[:3,3])@previous_pose[:3,:3]@pose[:3,:3].T+pose[:3,3]
    if i%6==0:mesh_pose=pose
    display_prediction=(current-pose[:3,3])@pose[:3,:3]@mesh_pose[:3,:3].T+mesh_pose[:3,3]
    record={'frame':i,'hard_guards':guards,'command':bridge.command.tolist(),'controller_target':bridge.state['controller_target'],'controller_target_velocity':bridge.state['controller_target_velocity'],'axis_positions':np.asarray(bridge.state['mechanism_qpos'])[manual.motion.AXES].tolist(),'axis_velocity':np.asarray(bridge.state['mechanism_qvel'])[manual.motion.AXES-1].tolist(),'cohort_max_step_m':float(np.linalg.norm(delta,axis=1).max()),'cohort_p99_step_m':float(np.quantile(np.linalg.norm(delta,axis=1),.99)),'cohort_max_up_step_m':float(delta[:,2].max()),'cohort_p99_rigid_carry_error_m':float(np.quantile(np.linalg.norm(current-predicted,axis=1),.99)),'display_bucket_motion_staleness_p99_m':float(np.quantile(np.linalg.norm(display_prediction-current,axis=1),.99)),'bucket_force_world_n':row['bucket_mpm_force_world_n'],'equivalent_drive_load':bridge.state['equivalent_drive_load'],'actuator_force':bridge.state['actuator_force'],'external_load_compensation':bridge.state.get('external_load_compensation',[0.]*4),'stopped':manual.motion.stopped(row),'rigid_contact_substeps':bridge.state.get('contact_substeps',{}),'bucket_pose':pose.tolist(),'rock_qpos':[r['qpos'] for r in bridge.state['rocks']],'phase':('replay' if i<=1370 else 'hold' if i<=1490 else 'reverse' if i<=1520 else 'stop'),'wall_seconds':row['wall_seconds'],**{k:v for k,v in captured.items() if k.startswith('projection_')}}
    journal.write(json.dumps(record)+'\n');journal.flush();events.append(record)
    if i>=1010:positions.append(current.copy());poses.append(pose.copy())
    if i in (480,600,1370,1490,1520,1580):
     state=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+i/60)
     np.savez_compressed(out/f'diagnostic-{i:06}.npz',**state)
     (out/f'bridge-state-{i:06}.json').write_text(json.dumps(bridge.state))
    if i<=480: assert bridge.state.get('external_load_compensation',[0.]*4)==[0.0]*4
    assert abs(bridge.state.get('external_load_compensation',[0.]*4)[3])<=60000.0001
    previous=current;previous_pose=pose
    if i%60==0:print('REPLAY',i,'cohort_projection',record['projection_cohort_max_m'],'target',record['controller_target'][3],flush=True)
   final=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+len(trace)/60)
   np.savez_compressed(out/'final-001580.diagnostic.npz',**final)
   np.savez_compressed(out/'lift-cohort.npz',q=np.array(positions),poses=np.array(poses),ids=selected,first_frame=1010)
   prefix=dict(np.load(out/'diagnostic-000480.npz'));baseline=dict(np.load(root/'work/carry-sync-curl-audit-20260923-v1/diagnostic-000480.npz'))
   comparisons={k:bool(np.array_equal(v,baseline[k])) for k,v in prefix.items() if k in baseline and v.dtype.kind in 'fiu'}
   result={'frames':len(trace),'cohort_count':len(selected),'restore':restored,'comparisons':comparisons,'prefix_480_arrays_exact':all(comparisons.values()),'compensation_candidate_only':mode=='candidate','source_bindings':bindings,'bindings_unchanged':all(manual.motion.probe.bench.sha256(p)==h for p,h in bindings.items()),'training_eligible':False}
   (out/'report.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('restore','source_bindings')}),flush=True)
 except BaseException as error:
  recovery=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+frame[0]/60)
  np.savez_compressed(out/f'recovery-{frame[0]:06}.diagnostic.npz',**recovery)
  (out/'failure.json').write_text(json.dumps({'frame':frame[0],'error':repr(error),'traceback':traceback.format_exc(),'source_bindings':bindings},indent=2))
  raise
 finally:bridge.close()
