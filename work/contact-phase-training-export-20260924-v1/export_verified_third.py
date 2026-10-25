"""Verify exact replay and export a versioned phase sidecar for original tensors."""
import hashlib,json
from pathlib import Path
import numpy as np
from contact_phase_data import align_phase_labels,PHASES
from contact_phase_geometry import SolidBucketGap
from layered_excavation_approach_planner import _load_kinematics
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-phase-teacher-replay-v1/third-scoop-v2'
OUT=ROOT/'local-data/contact-bc-phase-sidecars-20260924-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
 report_path=RUN/'report.json';r=json.loads(report_path.read_text())
 assert r['status']=='teacher_replay_completed_endpoint_bitwise_equal' and r['frames_completed']==1490
 assert len(r['final_arrays_exact'])==24 and all(r['final_arrays_exact'].values()) and all(r['zero_step_exact'].values())
 assert r['source_unchanged'] and r['original_endpoint_unchanged']
 for s in r['inputs']+r['source_inputs']+[r['source'],r['original_endpoint'],r['feature_file']]:assert sha(Path(s['path']))==s['sha256']
 teacher=ROOT/'runs/layered-47-sparse-manual-contact-v1'/r['run']
 original=[json.loads(s) for s in (teacher/'physics.ndjson').read_text().splitlines()]
 physical=[json.loads(s) for s in (RUN/'frames.ndjson').read_text().splitlines()]
 phases=[json.loads(s) for s in (RUN/'phase-trace.ndjson').read_text().splitlines()]
 assert len(original)==len(physical)==len(phases)==1490
 for n,(a,b,p) in enumerate(zip(original,physical,phases,strict=True),start=1):
  assert a['frame']==b['frame']==p['frame']==n and b['hard_guard'] and all(b['hard_guard_checks'].values())
  assert a['applied_command_si']==b['applied_command_si']
  for key in ('mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort','rocks'):
   if key=='rocks':
    for x,y in zip(a['after_state'][key],b['after_state'][key],strict=True):
     for name in ('qpos','qvel'):np.testing.assert_allclose(x[name],y[name],rtol=0,atol=1e-12)
   else:np.testing.assert_allclose(a['after_state'][key],b['after_state'][key],rtol=0,atol=1e-12)
 with np.load(r['original_endpoint']['path'],allow_pickle=False) as a,np.load(RUN/'final.npz',allow_pickle=False) as b:
  assert set(a.files)==set(b.files) and len(a.files)==24
  for k in a.files:assert np.array_equal(a[k],b[k])
 events=[p['phase']['event'] for p in phases if p['phase']['event']]
 assert events==r['events'] and [e['transition'] for e in events]==[PHASES[i]+'_to_'+PHASES[i+1] for i in range(4)]
 # Independently calculate the consecutive gap boundary from saved observations.
 streak=0;found=None
 for p in phases:
  o=p['observation']
  if p['phase']['phase_before_observation']=='lift':
   gap=None if o is None else o['remaining_soil_gap_m']
   streak=streak+1 if gap is not None and gap>=.003 else 0
   if streak==5 and found is None:found=p['frame']
 assert found==events[-1]['confirmation_frame_end']
 prior=json.loads((ROOT/'work/contact-phase-labels-20260924-v1/phase-completion-replay-v1.json').read_text())
 proxy=Path(next(i['path'] for i in prior['geometry_inputs'] if i['path'].endswith('proxy-manifest.json')))
 observer=SolidBucketGap(json.loads(proxy.read_text()));kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
 assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
 checked=[]
 for s in r.get('clear_candidate_snapshots',[]):
  path=Path(s['path']);assert sha(path)==s['sha256']
  with np.load(path,allow_pickle=False) as d:
   pose=assembly.assemble(d['mechanism_qpos'],include_meshes=False)[1]['bucket'];gap=observer.observe(d['particle_q'],pose)
  o=phases[s['frame']-1]['observation'];assert gap['remaining_soil_gap_m']==o['remaining_soil_gap_m']
  assert gap['ground_connected_particles']==o['ground_connected_particles'];checked.append(s)
 assert set(range(found-4,found+1)).issubset({s['frame'] for s in checked})
 manifest_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/tensor-manifest.json'
 manifest=json.loads(manifest_path.read_text());e=next(e for e in manifest['episodes'] if e['run']==r['run'])
 path=Path(e['path']);assert sha(path)==e['sha256']
 with np.load(path,allow_pickle=False) as tensor:
  labels=align_phase_labels(tensor['frame'],phases)
  # Separate event-interval reconstruction avoids trusting the alignment helper.
  expected=np.zeros(len(labels['frame']),dtype=np.int8)
  for index,event in enumerate(events,start=1):expected[labels['frame']>=event['first_available_decision_frame']]=index
  np.testing.assert_array_equal(labels['phase_id'],expected)
  assert len(expected)==e['rows']
 assignment_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/pilot-assignment.json'
 assignment=json.loads(assignment_path.read_text())
 split=next(k for k in ('development_runs','validation_runs','sealed_test_runs') if r['run'] in assignment[k])
 OUT.mkdir(parents=True,exist_ok=True);dest=OUT/(r['run']+'.phases.npz')
 if dest.exists():raise ValueError('refuse_to_overwrite_phase_sidecar')
 np.savez_compressed(dest,**labels)
 counts={name:int(np.count_nonzero(labels['phase_id']==i)) for i,name in enumerate(PHASES)}
 recontacts=[p['frame'] for p in phases if p['frame']>found and p['observation'] is not None and p['observation']['remaining_soil_gap_m']<.003]
 result=dict(schema='contact_bc_causal_phase_sidecar_v1',status='one_episode_candidate_exported',
  report=dict(path=str(report_path),sha256=sha(report_path)),phase_trace=dict(path=str(RUN/'phase-trace.ndjson'),sha256=sha(RUN/'phase-trace.ndjson')),
  source_tensor=e,sidecar=dict(path=str(dest),sha256=sha(dest)),phase_order=PHASES,rows=e['rows'],phase_counts=counts,
  events=events,geometry_confirmation_snapshots=checked,post_confirmation_below_threshold_frames=recontacts,
  split=split,split_assignment_sha256=sha(assignment_path),original_tensor_manifest_sha256=sha(manifest_path),
  original_tensor_unchanged=sha(path)==e['sha256'],training_eligible=False,normalization_changed=False,actions_changed=False,
  policy='Phase BEFORE each action; event at frame end n first applies to action n+1. Sidecar only; original keys, tensors, groups and loss choices unchanged.',
  human_boundaries_overwritten=False,earliest_boundary_scope='Earliest 5-frame gap confirmation after causal lift entry under the frozen candidate; not exact first geometric separation.',
  coverage=dict(complete_episode_count=1,total_original_episodes=len(manifest['episodes']),pending_runs=[x['run'] for x in manifest['episodes'] if x['run']!=r['run']]),
  source_implementation=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),ROOT/'src/v4/contact_phase_data.py')])
 dump(WORK/'export-verification-v1.json',result)
 print(json.dumps(dict(run=r['run'],rows=e['rows'],phase_counts=counts,events=events,recontact_count=len(recontacts),split=split),indent=2))

if __name__=='__main__':main()
