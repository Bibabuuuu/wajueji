"""Independently audit the saved bounded phase-policy run, including a failure."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha, LIMITS
from contact_bc_ground_guard import BucketGroundGuard
from contact_phase_live import LivePhaseObserver
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-phase-bc-probe-20260924-v1/contact600'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def ident(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
 r=read(RUN/'report.json');a=rows(RUN/'actions.ndjson');physical=rows(RUN/'physics/frames.ndjson')
 g=rows(RUN/'ground-guard.ndjson');trace=rows(RUN/'phase-trace.ndjson');n=len(physical)
 source=Path(r['source']['path']);policy=Path(r['policy']['path'])
 endpoint=RUN/('physics/failure.npz' if r['status']=='failed' else 'final.npz')
 with np.load(endpoint,allow_pickle=False) as z:final={k:z[k] for k in z.files}
 with np.load(source,allow_pickle=False) as z:initial={k:z[k] for k in z.files}
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
 assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
 pose=lambda state:assembly.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
 guard=BucketGroundGuard(assembly,initial['mechanism_qpos'])
 contract=read(r['phase_contract']['path'])
 observer=LivePhaseObserver(read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'),
  read(contract['cavity_config'])['bucket']['polygon_xz_m'],ContactPhaseConfig(**read(contract['contact_config'])['parameters']),
  CompletionConfig(**read(contract['completion_config'])['parameters']),pose(a[0]['before_state']))
 particles=np.load(RUN/'display/particles.npy',mmap_mode='r',allow_pickle=False)
 for i,t in enumerate(trace):
  assert observer.router.phase_for_decision(i+1)==a[i]['phase']
  got=observer.observe_end(frame=i+1,state=physical[i]['after_state'],pose=pose(physical[i]['after_state']),soil_force_n=t['bucket_mpm_force_norm_n'],rock_contact_substeps=physical[i]['rock_bucket_contact_substeps'],particles=lambda:particles[i+1])
  assert got['phase']==t['phase'] and got['observation']==t['observation']
 for check in g:
  frame=check['frame'];state=a[0]['before_state'] if frame==0 else a[frame-1]['before_state'] if check['stage']=='before_step' else physical[frame-1]['after_state']
  assert guard.measure(state['mechanism_qpos'])=={key:value for key,value in check.items() if key not in ('frame','stage')}
 meta=read(Path(str(endpoint)+'.json'));sm=read(Path(str(source)+'.json'))
 failed_ground=r['status']=='failed' and 'bc_bucket_ground_clearance' in r.get('failure','')
 checks=dict(frames=n==len(a)==r['frames_completed'] and [x['frame'] for x in physical]==list(range(1,n+1)),
  prefix=r['prefix_frames_verified']==120 and all(r['prefix_final_arrays_exact'].values()),
  restored=all(r['zero_step_arrays_exact'].values()),
  commands=all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(a,physical,strict=True)),
  state_chain=all(a[i]['before_state']==physical[i-1]['after_state'] for i in range(1,n)),
  previous_action=all(a[i]['previous_applied_action_si']==a[i-1]['applied_command_si'] for i in range(1,n)),
  previous_load_initialized_and_finite=a[0]['previous_drive_load'] is None and all(np.asarray(x['previous_drive_load']).shape==(4,) and np.isfinite(x['previous_drive_load']).all() for x in a[1:]),
  causal_terrain=all(x['terrain_anchor_frame']<x['frame'] and x['frame']-1-x['terrain_anchor_frame']<=2 for x in a),
  phase_timing=all(x['latest_allowed_phase_evidence_frame']==x['frame']-1 for x in a),
  phase_recomputed=True,ground_recomputed=True,
  phase_coverage=len(trace)==n-int(failed_ground),
  legacy_hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
  material=all(x['hencky_material_guard']['safe'] for x in physical),sweep=all(x['moving_bucket_sweep']['safe'] for x in physical),
  ground_stop=(all(x['safe'] for x in g[:-1]) and not g[-1]['safe'] and g[-1]['frame']==n) if failed_ground else all(x['safe'] for x in g),
  endpoint_matches=np.array_equal(final['mechanism_qpos'],physical[-1]['after_state']['mechanism_qpos']),
  finite=all(np.isfinite(v).all() for v in final.values() if v.dtype.kind in 'fciu'),
  particles=initial['particle_q'].shape==final['particle_q'].shape,
  rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
  lineage=meta['lineage_root']==sm['lineage_root'],inventory=meta['training_reset']==sm['training_reset'],
  elapsed=bool(np.max(np.abs(final['simulation_time_s']-initial['simulation_time_s']-n/60))<1e-11),
  source=sha(source)==r['source']['sha256'],policy=sha(policy)==r['policy']['sha256'],
  display=r['display_cache']['frames']==n and r['display_cache']['probe_status']==r['status'])
 for name,digest in r['code_sha256'].items():checks['code_'+name]=sha(ROOT/'src/v4'/name)==digest
 for name,digest in r['runtime_bindings'].items():checks['runtime_'+name]=sha(ROOT/'src/v4'/name)==digest
 def tip(state):
  p=pose(state);return p[:3,:3]@TIP+p[:3,3]
 original=rows(ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-115812-1458627/physics.ndjson')
 windows=[]
 for start in range(0,n,60):
  end=min(start+60,n)
  windows.append(dict(frames=[start+1,end],candidate=np.mean([x['action_normalized'] for x in a[start:end]],axis=0).tolist(),human=np.mean([x['applied_command_si'] for x in original[start:end]],axis=0).__truediv__(LIMITS).tolist()))
 t0=tip(a[0]['before_state']);t1=tip(physical[-1]['after_state']);th=tip(original[n-1]['after_state'])
 result=dict(passed=all(checks.values()),checks=checks,frames=n,process_status=r['status'],failure=r.get('failure'),
  events=r['events'],last_phase=trace[-1]['phase']['phase_for_next_decision'] if trace else 'approach_contact',
  phase_action_counts={p:sum(x['phase']==p for x in a) for p in ('approach_contact','penetrate','curl_load','lift','off_soil')},
  sampled_minimum_bucket_z_m=min(x['minimum_bucket_z_m'] for x in g),
  sampled_no_ground_penetration=all(x['minimum_bucket_z_m']>0 for x in g),
  final_tip_m=t1.tolist(),initial_tip_m=t0.tolist(),tip_displacement_m=(t1-t0).tolist(),human_same_elapsed_tip_m=th.tolist(),
  selected_goal=r['selected_goal'],command_windows=windows,last_action=a[-1]['action_normalized'],
  physics_wall_seconds=sum(x['wall_seconds'] for x in physical),wall_seconds=r['wall_seconds'],
  soil_contact_frames=sum(x['bucket_mpm_contact_samples']>0 for x in physical),
  rock_contact_frames=sum(x['rock_bucket_contact_substeps']>0 for x in physical),
  source_unchanged=r['source_unchanged'],autonomous_excavation_qualified=False,
  qualification='Evidence/interface/protection audit only; passed does not mean successful excavation.',
  inputs=[ident(source),ident(policy),ident(r['phase_contract']['path'])],
  outputs=[ident(RUN/x) for x in ('actions.ndjson','phase-trace.ndjson','terrain.ndjson','physics/frames.ndjson','ground-guard.ndjson','report.json')]+[ident(endpoint),ident(Path(str(endpoint)+'.json'))],
  reviewer=ident(Path(__file__)))
 (W/'contact-review.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('checks','inputs','outputs')},indent=2));assert result['passed'],checks

if __name__=='__main__':main()
