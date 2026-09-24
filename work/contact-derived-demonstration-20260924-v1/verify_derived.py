"""Independent decisions/observations/physics-to-array audit of derived data."""
import hashlib,json
from pathlib import Path
import numpy as np
from contact_bc_data import STATE_FIELDS,LIMITS
from contact_phase_geometry import SolidBucketGap
from layered_excavation_approach_planner import _load_kinematics
from layered_47_cycle_geometry import bucket_mask
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent;RUN=ROOT/'runs/contact-derived-demonstration-v1/third-scoop'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 report_path=RUN/'report.json';r=json.loads(report_path.read_text());assert r['status']=='derived_demo_completed_pending_independent_audit'
 assert r['source_unchanged'] and r['teacher_tensor_unchanged'] and all(r['zero_step_arrays_exact'].values())
 assert r['protocol_sha256']==sha(WORK/'protocol.json')
 for item in r['inputs']+r['teacher_sources']+[r['source'],r['original_endpoint'],r['teacher_tensor'],r['tensors']]:assert sha(Path(item['path']))==item['sha256']
 p=r['protocol'];decisions=rows(RUN/'decisions.ndjson');physical=rows(RUN/'frames.ndjson');phases=rows(RUN/'phase-trace.ndjson');terrains={x['frame']:x['terrain'] for x in rows(RUN/'terrain.ndjson')}
 assert len(decisions)==len(physical)==len(phases)==1490==r['frames_completed']
 original=rows(ROOT/'runs/layered-47-sparse-manual-contact-v1'/r['teacher_run']/'physics.ndjson')
 names=['approach_contact','penetrate','curl_load','lift','off_soil'];prior='approach_contact';streak=0;confirmation=None;events=[]
 for n,(d,f,stage,t) in enumerate(zip(decisions,physical,phases,original,strict=True),start=1):
  assert d['frame']==f['frame']==stage['frame']==n
  assert f['hard_guard'] and all(f['hard_guard_checks'].values()) and stage['ground']['safe']
  assert stage['ground']['minimum_bucket_z_m']>.005
  assert d['applied_command_si']==f['applied_command_si']==t['applied_command_si']
  assert d['teacher_requested_command_si']==t['requested_command_si']
  if n>1:
   assert d['before_state']==physical[n-2]['after_state']
   assert d['previous_applied_action_si']==physical[n-2]['applied_command_si']
  assert d['phase']==prior==stage['phase']['phase_before_observation']
  after=stage['phase']['phase_for_next_decision'];event=stage['phase']['event']
  if event:
   assert names.index(after)==names.index(prior)+1 and event['confirmation_frame_end']==n and event['first_available_decision_frame']==n+1
   events.append(event)
  else:assert after==prior
  if prior=='lift':
   gap=stage['observation']['remaining_soil_gap_m'];streak=streak+1 if gap is not None and gap>=.003 else 0
   if streak==5:
    assert event and after=='off_soil';confirmation=n
  prior=after
 assert events==r['events'] and len(events)==4 and prior=='off_soil'
 with np.load(r['teacher_tensor']['path'],allow_pickle=False) as t:teacher_frames=t['frame'].copy()
 original_report=json.loads((ROOT/'runs/layered-47-sparse-manual-contact-v1'/r['teacher_run']/'report.json').read_text())
 goal=original_report['selected_cut'];previous_frame=None;sequence=0
 with np.load(r['tensors']['path'],allow_pickle=False) as tensors:
  a={k:tensors[k] for k in tensors.files}
  np.testing.assert_array_equal(a['frame'],teacher_frames)
  assert len(a['frame'])==1457 and all(np.isfinite(v).all() for v in a.values())
  for index,frame in enumerate(a['frame']):
   frame=int(frame);d=decisions[frame-1];anchor=max(k for k in terrains if k<frame);terrain=terrains[anchor]
   assert anchor==d['terrain_anchor_frame']==a['terrain_anchor_frame'][index]
   assert a['terrain_age_frames'][index]==frame-1-anchor<=2
   for key in STATE_FIELDS:np.testing.assert_array_equal(a[key][index],np.asarray(d['before_state'][key],dtype=np.float32))
   h=np.asarray(terrain['height_m'],dtype=float);known=np.isfinite(h)
   np.testing.assert_array_equal(a['terrain_known'][index],known);np.testing.assert_array_equal(a['terrain_height_m'][index],np.where(known,h,0).astype(np.float32))
   np.testing.assert_array_equal(a['terrain_origin_xy_m'][index],np.asarray(terrain['origin_xy_m'],dtype=np.float32))
   assert a['terrain_resolution_m'][index]==np.float32(terrain['resolution_m'])
   np.testing.assert_array_equal(a['goal_world_m'][index],np.asarray([*goal['entry_xy_m'],goal['surface_z_m']],dtype=np.float32))
   np.testing.assert_array_equal(a['applied_action_normalized'][index],(np.asarray(d['applied_command_si'])/LIMITS).astype(np.float32))
   np.testing.assert_array_equal(a['requested_action_normalized'][index],(np.asarray(d['teacher_requested_command_si'])/LIMITS).astype(np.float32))
   np.testing.assert_array_equal(a['previous_applied_action_si'][index],np.asarray(d['previous_applied_action_si'],dtype=np.float32))
   reset=previous_frame is None or frame!=previous_frame+1;sequence+=int(reset)
   assert a['reset_learning_history'][index]==reset and a['sequence_id'][index]==sequence and a['previous_drive_load_valid'][index]==(not reset)
   np.testing.assert_array_equal(a['previous_drive_load'][index],np.asarray([0.]*4 if reset else d['previous_drive_load'],dtype=np.float32))
   assert a['phase_id'][index]==names.index(d['phase']) and a['latest_allowed_phase_evidence_frame'][index]==frame-1
   np.testing.assert_array_equal(a['phase_one_hot'][index],np.eye(5,dtype=np.float32)[names.index(d['phase'])])
   previous_frame=frame
  counts={name:int(np.count_nonzero(a['phase_id']==i)) for i,name in enumerate(names)}
  assert counts==r['tensors']['phase_counts'] and all(counts.values())
 kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
 proxy_path=ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json';observer=SolidBucketGap(json.loads(proxy_path.read_text()))
 assert [x['frame'] for x in r['confirmation_snapshots']]==list(range(confirmation-4,confirmation+1))
 for snap in r['confirmation_snapshots']:
  assert sha(Path(snap['path']))==snap['sha256']
  with np.load(snap['path'],allow_pickle=False) as data:
   pose=assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket'];o=observer.observe(data['particle_q'],pose)
  assert o==phases[snap['frame']-1]['observation'] and o['remaining_soil_gap_m']>=.003
 with np.load(r['final_checkpoint']['path'],allow_pickle=False) as final,np.load(r['original_endpoint']['path'],allow_pickle=False) as source:
  error=np.linalg.norm(final['particle_q']-source['particle_q'],axis=1)
  assert float(error.max())==r['endpoint_comparison']['particle_max_position_error_m']<=p['maximum_final_particle_position_deviation_m']
  assert float(np.quantile(error,.99))==r['endpoint_comparison']['particle_p99_position_error_m']<=p['p99_final_particle_position_deviation_m']
  for label,arrays in (('original',source),('derived',final)):
   pose=assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket'];count=int(bucket_mask(arrays['particle_q'],pose).sum())
   assert count==r['endpoint_comparison'][label+'_cavity_particle_count']
 assert r['endpoint_comparison']['cavity_count_relative_difference']<=p['maximum_final_cavity_particle_count_relative_difference']
 for key,limit in [('bucket_CAD_point_m',p['maximum_bucket_CAD_point_deviation_m']),('rock_center_m',p['maximum_rock_center_deviation_m']),('rock_orientation_rad',p['maximum_rock_orientation_deviation_rad'])]:
  assert max(x['deviation'][key] for x in phases)==r['maximum_deviations'][key]<=limit
 assignment_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/pilot-assignment.json';assignment=json.loads(assignment_path.read_text())
 assert sha(assignment_path)==r['assignment_sha256'] and r['teacher_run'] in assignment[r['split']] and r['split']=='validation_runs'
 recontacts=[x['frame'] for x in phases if x['frame']>confirmation and x['observation'] and x['observation']['remaining_soil_gap_m']<.003]
 result=dict(schema='derived_demo_independent_audit_v1',passed=True,data_domain=r['data_domain'],teacher_run=r['teacher_run'],
  report=dict(path=str(report_path),sha256=sha(report_path)),tensors=r['tensors'],frames=1490,expert_rows=1457,phase_counts=counts,events=events,
  off_soil_confirmation_frame_end=confirmation,post_confirmation_below_gap_frames=recontacts,maximum_deviations=r['maximum_deviations'],endpoint_comparison=r['endpoint_comparison'],
  split=r['split'],original_sources_unchanged=True,original_tensor_unchanged=True,full_batch_complete=False,training_eligible=False,
  qualification='One reviewed derived-data sample with causal features and phase candidates; not whole-dataset admission, trained policy or autonomous excavation.',
  verifier_sha256=sha(Path(__file__)))
 (WORK/'verification-v1.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('passed','frames','expert_rows','phase_counts','off_soil_confirmation_frame_end','post_confirmation_below_gap_frames','maximum_deviations','endpoint_comparison')},indent=2))
if __name__=='__main__':main()
