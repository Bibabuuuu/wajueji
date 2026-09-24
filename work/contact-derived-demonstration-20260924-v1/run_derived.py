"""Replayed actions with actual causal observations: a separate data domain."""
from collections import deque
import hashlib,json,time
from pathlib import Path
import numpy as np
from contact_bc_data import LIMITS
from contact_bc_inference import structured_observation
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig,FivePhaseCandidate
from contact_phase_data import PHASES
from contact_phase_geometry import SolidBucketGap
from contact_bc_ground_guard import BucketGroundGuard
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import bucket_mask
import run_layered_47_sparse_manual_contact_episode as manual
from run_layered_47_sparse_contact_handoff import wrench_summary

ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OLD=ROOT/'work/contact-phase-labels-20260924-v1'
OUT=ROOT/'runs/contact-derived-demonstration-v1/third-scoop'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def append(p,d):
 with p.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n')

def main():
 protocol=json.loads((WORK/'protocol.json').read_text());run=protocol['teacher_run'];folder=ROOT/'runs/layered-47-sparse-manual-contact-v1'/run
 teacher_report=json.loads((folder/'report.json').read_text());rows=[json.loads(s) for s in (folder/'physics.ndjson').read_text().splitlines()]
 source=teacher_report['source_checkpoint'];endpoint=teacher_report['cycle_checkpoints'][-1]
 assert sha(Path(source['path']))==source['sha256'] and sha(Path(endpoint['path']))==endpoint['sha256']
 manifest_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/tensor-manifest.json'
 manifest=json.loads(manifest_path.read_text());tensor=next(e for e in manifest['episodes'] if e['run']==run)
 assert sha(Path(tensor['path']))==tensor['sha256']
 with np.load(tensor['path'],allow_pickle=False) as data:eligible=set(map(int,data['frame']))
 assignment_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/pilot-assignment.json';assignment=json.loads(assignment_path.read_text())
 split=next(k for k in ('development_runs','validation_runs') if run in assignment[k]);goal=teacher_report['selected_cut']
 poly_path=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json')
 poly=np.array(json.loads(poly_path.read_text())['bucket']['polygon_xz_m']);local=np.c_[poly[:,0],np.zeros(len(poly)),poly[:,1]]
 def angle(pose):
  world=local@pose[:3,:3].T+pose[:3,3];radial=np.array([pose[1,1],-pose[0,1]]);radial/=np.linalg.norm(radial);chord=world[-1]-world[0]
  return float(np.rad2deg(np.arctan2(chord[:2]@radial,-chord[2])))
 report=dict(schema='derived_teacher_demonstration_v1',status='initializing',data_domain='replayed_teacher_actual_observations',teacher_run=run,
  source=source,original_endpoint=endpoint,teacher_tensor=tensor,split=split,lineage=tensor['lineage'],
  protocol=protocol,protocol_sha256=sha(WORK/'protocol.json'),assignment_sha256=sha(assignment_path),original_tensor_manifest_sha256=sha(manifest_path),
  teacher_sources=[dict(path=str(folder/n),sha256=sha(folder/n)) for n in ('report.json','physics.ndjson','input.ndjson','annotations.ndjson')],
  phase_order=PHASES,events=[],maximum_deviations=dict(bucket_CAD_point_m=0.,rock_center_m=0.,rock_orientation_rad=0.),
  training_eligible=False,policy_takeover=False,original_data_changed=False,
  inputs=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),poly_path,OLD/'phase-completion-v1.json',OLD/'phase-candidate-v1.json',ROOT/'src/v4/contact_phase_geometry.py',ROOT/'src/v4/contact_phase_completion.py',ROOT/'src/v4/contact_bc_inference.py',ROOT/'src/v4/contact_bc_data.py',ROOT/'src/v4/contact_bc_ground_guard.py')])
 start=time.perf_counter();r=None;values=[];sequence=0;last_expert=None;window=deque(maxlen=5)
 try:
  with CycleRuntime(source['path'],OUT,allow_training_reset=True) as r:
   captured=r.capture();report['zero_step_arrays_exact']={k:bool(np.array_equal(v,captured[k])) for k,v in r.arrays.items()};assert all(report['zero_step_arrays_exact'].values())
   report['runtime_bindings']=r.bindings;report['restore']=r.restored
   guard=BucketGroundGuard(r.assembly,r.bridge.state['mechanism_qpos']);assert guard.measure(r.bridge.state['mechanism_qpos'])['safe']
   report['ground_geometry_sha256']=guard.geometry_sha256
   observer=SolidBucketGap(r.proxy);vertices=manual._cache_rock_vertices(r.root,r.records)
   pose=r.assembly.assemble(r.bridge.state['mechanism_qpos'],include_meshes=False)[1]['bucket']
   router=FivePhaseCandidate(ContactPhaseConfig(**json.loads((OLD/'phase-candidate-v1.json').read_text())['parameters']),CompletionConfig(**json.loads((OLD/'phase-completion-v1.json').read_text())['parameters']),initial_mouth_angle_deg=angle(pose))
   terrain=None;anchor=0;previous_action=[0.]*4;previous_load=None;report['status']='running';write(OUT/'report.json',report)
   for n,teacher in enumerate(rows,start=1):
    if n>protocol['frames'] or time.perf_counter()-start>protocol['wall_budget_seconds']:raise RuntimeError('derived_demo_budget')
    before=manual.compact_state(r.bridge.state);phase_before=router.phase_for_decision(n)
    if terrain is None or (n-1)%3==0:
     terrain=manual.terrain_observation(r.fixture,r.bridge.state,vertices);anchor=n-1;append(OUT/'terrain.ndjson',dict(frame=anchor,terrain=terrain))
    reset=last_expert is None or n!=last_expert+1
    if n in eligible:
     sequence+=int(reset)
     encoded=structured_observation(frame=n,state=before,terrain=terrain,terrain_frame=anchor,goal=goal,previous_action=previous_action,
       previous_load=None if reset else previous_load,sequence_id=sequence)
     encoded.update(reset_learning_history=np.bool_(reset),requested_action_normalized=(np.asarray(teacher['requested_command_si'])/LIMITS).astype(np.float32),
       applied_action_normalized=(np.asarray(teacher['applied_command_si'])/LIMITS).astype(np.float32),
       phase_id=np.int8(PHASES.index(phase_before)),phase_one_hot=np.eye(5,dtype=np.float32)[PHASES.index(phase_before)],
       latest_allowed_phase_evidence_frame=np.int64(n-1))
     values.append(encoded);last_expert=n
    append(OUT/'decisions.ndjson',dict(frame=n,before_state=before,phase=phase_before,terrain_anchor_frame=anchor,
       previous_applied_action_si=previous_action,previous_drive_load=previous_load,expert_row=n in eligible,reset_learning_history=reset if n in eligible else None,
       teacher_requested_command_si=teacher['requested_command_si'],applied_command_si=teacher['applied_command_si']))
    row=r.step(teacher['applied_command_si']);state=row['after_state'];pose=r.assembly.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
    reference=r.assembly.assemble(teacher['after_state']['mechanism_qpos'],include_meshes=False)[1]['bucket']
    delta=float(np.linalg.norm(guard.vertices@pose[:3,:3].T+pose[:3,3]-(guard.vertices@reference[:3,:3].T+reference[:3,3]),axis=1).max())
    actual=np.array([x['qpos'] for x in state['rocks']]);wanted=np.array([x['qpos'] for x in teacher['after_state']['rocks']])
    rock_gap=float(np.linalg.norm(actual[:,:3]-wanted[:,:3],axis=1).max())
    qa=actual[:,3:]/np.linalg.norm(actual[:,3:],axis=1)[:,None];qb=wanted[:,3:]/np.linalg.norm(wanted[:,3:],axis=1)[:,None]
    # Chord formula avoids acos cancellation near identical orientations.
    chord=np.minimum(np.linalg.norm(qa-qb,axis=1),np.linalg.norm(qa+qb,axis=1));rotation=float((4*np.arcsin(np.clip(chord/2,0,1))).max())
    deviation=dict(bucket_CAD_point_m=delta,rock_center_m=rock_gap,rock_orientation_rad=rotation)
    for k,v in deviation.items():report['maximum_deviations'][k]=max(report['maximum_deviations'][k],v)
    assert delta<=protocol['maximum_bucket_CAD_point_deviation_m'] and rock_gap<=protocol['maximum_rock_center_deviation_m'] and rotation<=protocol['maximum_rock_orientation_deviation_rad'],'derived_trajectory_deviation'
    ground=guard.measure(state['mechanism_qpos']);assert ground['safe'],'derived_ground_margin'
    force=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n'];gap=None;observation=None
    if phase_before in ('lift','off_soil'):
     q=r.fixture.state.particle_q.numpy();observation=observer.observe(q,pose);gap=observation['remaining_soil_gap_m']
     if phase_before=='lift':
      if gap is not None and gap>=.003:window.append((n,q.copy(),np.array(state['mechanism_qpos'])))
      else:window.clear()
    phase=router.observe_end(frame_end=n,soil_force_n=force,rock_contact_substeps=row['rock_bucket_contact_substeps'],mouth_angle_deg=angle(pose),
       actual_boom_cylinder_m=state['mechanism_qpos'][9],bucket_origin_z_m=pose[2,3],remaining_soil_gap_m=gap,gap_frame_end=n if gap is not None else None)
    append(OUT/'phase-trace.ndjson',dict(frame=n,phase=phase,observation=observation,deviation=deviation,ground=ground,bucket_mpm_force_norm_n=force))
    if phase['event']:
     report['events'].append(phase['event']);print('EVENT',n,phase['event']['transition'],flush=True)
     if phase['event']['transition']=='lift_to_off_soil':
      assert len(window)==5
      for frame,q,mech in window:
       p=OUT/f'confirmation-{frame:06d}.npz';np.savez_compressed(p,particle_q=q,mechanism_qpos=mech)
       report.setdefault('confirmation_snapshots',[]).append(dict(frame=frame,path=str(p),sha256=sha(p)))
    previous_action=list(teacher['applied_command_si']);previous_load=list(r.bridge.state['equivalent_drive_load'])
    if n%120==0:report['frames_completed']=n;write(OUT/'report.json',report);print('DERIVED_FRAME',n,router.phase,round(time.perf_counter()-start,1),flush=True)
   final=r.capture()
   with np.load(endpoint['path'],allow_pickle=False) as original:
    errors=np.linalg.norm(final['particle_q']-original['particle_q'],axis=1);max_error=float(errors.max());p99=float(np.quantile(errors,.99))
    original_pose=r.assembly.assemble(original['mechanism_qpos'],include_meshes=False)[1]['bucket']
    old_count=int(bucket_mask(original['particle_q'],original_pose).sum());new_count=int(bucket_mask(final['particle_q'],pose).sum())
    relative=abs(new_count-old_count)/max(old_count,1)
    report['endpoint_comparison']=dict(particle_max_position_error_m=max_error,particle_p99_position_error_m=p99,
      original_cavity_particle_count=old_count,derived_cavity_particle_count=new_count,cavity_count_relative_difference=relative,
      bitwise_arrays={k:bool(np.array_equal(v,original[k])) for k,v in final.items()})
   assert max_error<=protocol['maximum_final_particle_position_deviation_m'] and p99<=protocol['p99_final_particle_position_deviation_m'] and relative<=protocol['maximum_final_cavity_particle_count_relative_difference'],'derived_endpoint_deviation'
   assert n==1490 and len(values)==tensor['rows'] and router.phase=='off_soil','incomplete_derived_demo'
   report['final_checkpoint']=r.save(OUT/'final.npz',stage='hold')
   dest=OUT/'derived-tensors.npz';arrays={k:np.stack([v[k] for v in values]) for k in values[0]};np.savez_compressed(dest,**arrays)
   report['tensors']=dict(path=str(dest),sha256=sha(dest),rows=len(values),phase_counts={p:int(np.count_nonzero(arrays['phase_id']==i)) for i,p in enumerate(PHASES)})
   report.update(status='derived_demo_completed_pending_independent_audit',frames_completed=n)
 except BaseException as error:
  report.update(status='failed',error=repr(error),frames_completed=r.frames if r else 0);raise
 finally:
  report['wall_s']=time.perf_counter()-start
  report['source_unchanged']=sha(Path(source['path']))==source['sha256'];report['teacher_tensor_unchanged']=sha(Path(tensor['path']))==tensor['sha256']
  if OUT.exists():write(OUT/'report.json',report)
if __name__=='__main__':main()
