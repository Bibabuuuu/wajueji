"""Independently recompute saved probe gaps using frozen unpruned geometry."""
import hashlib,json,subprocess,types
from pathlib import Path
import numpy as np
from layered_excavation_approach_planner import _load_kinematics
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-phase-continuity-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=subprocess.check_output(['git','show','145f53e:src/v4/contact_phase_geometry.py'],cwd=ROOT,text=True)
m=types.ModuleType('baseline');exec(compile(source,'145f53e_geometry','exec'),m.__dict__)
prior=json.loads((ROOT/'work/contact-phase-labels-20260924-v1/phase-completion-replay-v1.json').read_text())
proxy=Path(next(x['path'] for x in prior['geometry_inputs'] if x['path'].endswith('proxy-manifest.json')))
observer=m.SolidBucketGap(json.loads(proxy.read_text()))
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
results=[]
for name,expected_count in (('first-hold-v2',5),('third-hold',6)):
 p=RUN/name/'phase-audit.json';r=json.loads(p.read_text())
 assert r['status']=='bounded_hold_completed' and r['frames_completed']==expected_count
 assert len(r['zero_step_arrays_exact'])==24 and all(r['zero_step_arrays_exact'].values())
 assert max(r['restore']['cycle_errors'].values())<=1e-12
 for item in r['implementation_inputs']+r['source_inputs']+[r['source'],r['feature_file'],r['final_checkpoint']]:assert sha(Path(item['path']))==item['sha256']
 rows=[json.loads(x) for x in (p.parent/'frames.ndjson').read_text().splitlines()]
 assert len(rows)==expected_count
 gaps=[];streak=0;phase='lift';confirmation=None
 for index,(f,physical) in enumerate(zip(r['frames'],rows,strict=True),start=1):
  s=Path(f['snapshot']['path']);assert sha(s)==f['snapshot']['sha256']
  assert f['local_frame']==physical['frame']==index
  assert f['frame_end']==r['source_frame_end']+index
  assert physical['hard_guard'] and all(physical['hard_guard_checks'].values())
  assert physical['applied_command_si']==[0,0,0,0]
  with np.load(s,allow_pickle=False) as d:
   assert np.array_equal(d['mechanism_qpos'],physical['after_state']['mechanism_qpos'])
   pose=assembly.assemble(d['mechanism_qpos'],include_meshes=False)[1]['bucket']
   fresh=observer.observe(d['particle_q'],pose)
  assert fresh==f['observation']
  gap=fresh['remaining_soil_gap_m'];gaps.append(gap)
  assert f['phase']['phase_before_observation']==phase
  if phase=='lift':
   streak=streak+1 if gap is not None and gap>=.003 else 0
   if streak==5:phase='off_soil';confirmation=f['frame_end']
  assert f['phase']['phase_for_next_decision']==phase
  assert f['phase']['off_soil_clear_frames']==streak
  event=f['phase']['event']
  if event:
   assert event['transition']=='lift_to_off_soil' and event['confirmation_frame_end']==confirmation
   assert event['first_available_decision_frame']==confirmation+1
 assert r['source_unchanged'] and phase==r['final_phase']
 if name=='first-hold-v2':assert confirmation is None and all(x<.003 for x in gaps)
 else:assert confirmation==1495
 results.append(dict(run=name,report=dict(path=str(p),sha256=sha(p)),frames=expected_count,
  zero_step_exact_arrays=24,all_hard_guards_passed=True,source_unchanged=True,
  baseline_gaps_m=gaps,confirmation_frame_end=confirmation,first_available_decision_frame=confirmation+1 if confirmation else None,
  final_phase=phase,median_observer_including_copy_pose_hash_s=float(np.median([f['observer_including_copy_pose_hash_wall_s'] for f in r['frames']])),
  median_physics_wall_s=float(np.median([f['physics_wall_s'] for f in r['frames']]))))
# The failed adapter's one physics frame is retained, not counted as success.
failure=RUN/'first-hold';failed=json.loads((failure/'phase-audit.json').read_text())
assert failed['status']=='failed' and 'KeyError' in failed['error']
assert len((failure/'frames.ndjson').read_text().splitlines())==1
assert sha(OUT/'probe_continuity_failed_v1.py')==failed['implementation_inputs'][0]['sha256']
with np.load(failure/'failure.npz',allow_pickle=False) as a,np.load(RUN/'first-hold-v2/frame-001.npz',allow_pickle=False) as b:
 for k in ('particle_q','mechanism_qpos'):assert np.array_equal(a[k],b[k])
report=dict(schema='phase_continuity_independent_audit_v1',status='passed',
 baseline_git_commit='145f53e',baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
 verifier_sha256=sha(Path(__file__)),successful_observed_frames=11,total_physics_frames_including_failed_adapter=12,
 failed_adapter=dict(report_path=str(failure/'phase-audit.json'),report_sha256=sha(failure/'phase-audit.json'),
   error=failed['error'],saved_failure_path=str(failure/'failure.npz'),saved_failure_sha256=sha(failure/'failure.npz'),
   original_script_sha256=sha(OUT/'probe_continuity_failed_v1.py'),first_frame_particles_and_mechanism_equal_retry=True),episodes=results,
 limitations=['Short held-state continuation only, not earliest off-soil boundary on the old demonstration.',
 '4cm ground connectivity and particle-center/proxy gap, not exact material topology or stone clearance.',
 'Observer timing exceeds a real-time 60Hz frame budget; not a 60Hz end-to-end qualification.'])
(OUT/'probe-verification-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(results,indent=2))
