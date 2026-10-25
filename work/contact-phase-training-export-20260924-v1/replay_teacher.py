"""Bounded exact teacher replay to observe missing off-soil evidence."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig,FivePhaseCandidate
from contact_phase_geometry import SolidBucketGap
from layered_47_cycle_runtime import CycleRuntime
from run_layered_47_sparse_contact_handoff import wrench_summary

ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OLD=ROOT/'work/contact-phase-labels-20260924-v1'
RUN='manual-contact-20260924-115812-1458627'
OUT=ROOT/'runs/contact-phase-teacher-replay-v1/third-scoop-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
 prior=json.loads((OLD/'phase-completion-replay-v1.json').read_text());e=next(x for x in prior['episodes'] if x['run']==RUN)
 root=ROOT/'runs/layered-47-sparse-manual-contact-v1'/RUN
 raw=json.loads((root/'report.json').read_text());src=Path(raw['source_checkpoint']['path'])
 for s in e['source_inputs']+[e['feature_file'],e['endpoint']['snapshot'],raw['source_checkpoint']]:assert sha(Path(s['path']))==s['sha256']
 rows=[json.loads(s) for s in (root/'physics.ndjson').read_text().splitlines()]
 trace=json.loads(Path(e['feature_file']['path']).read_text())['trace']
 assert len(rows)==1490 and len(trace)==1491
 config=CompletionConfig(**json.loads((OLD/'phase-completion-v1.json').read_text())['parameters'])
 router=FivePhaseCandidate(ContactPhaseConfig(**json.loads((OLD/'phase-candidate-v1.json').read_text())['parameters']),config,initial_mouth_angle_deg=trace[0]['mouth_chord_from_down_deg'])
 report=dict(schema='teacher_phase_replay_tolerance_v2',status='initializing',run=RUN,
  source=raw['source_checkpoint'],original_endpoint=e['endpoint']['snapshot'],source_inputs=e['source_inputs'],feature_file=e['feature_file'],
  frames_requested=1490,wall_budget_s=480,live_policy_connected=False,original_annotations_changed=False,
  inputs=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),root/'report.json',OLD/'phase-completion-v1.json',OLD/'phase-candidate-v1.json',ROOT/'src/v4/contact_phase_completion.py',ROOT/'src/v4/contact_phase_geometry.py',ROOT/'src/v4/contact_phase_candidate.py')],
  events=[],observed_gap_frames=0,matched_recorded_frames=0,
  state_atol=1e-12,state_rtol=0,maximum_state_absolute_error=0.,non_bitwise_frames=[],final_arrays_require_bitwise_equal=True)
 start=time.perf_counter();r=None
 try:
  with CycleRuntime(src,OUT,allow_training_reset=True) as r:
   initial=r.capture();report['zero_step_exact']={k:bool(np.array_equal(v,initial[k])) for k,v in r.arrays.items()}
   assert all(report['zero_step_exact'].values()),'restore_not_exact'
   for key in ('mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort'):
    assert np.array_equal(r.bridge.state[key],rows[0]['before_state'][key]),'initial_recorded_state:'+key
   report['runtime_bindings']=r.bindings;report['restore']=r.restored
   observer=SolidBucketGap(r.proxy);report['status']='running';write(OUT/'report.json',report)
   for n,(expected,f) in enumerate(zip(rows,trace[1:],strict=True),start=1):
    if time.perf_counter()-start>480:raise RuntimeError('teacher_replay_wall_budget')
    before=router.phase_for_decision(n);row=r.step(expected['applied_command_si'])
    for key in ('mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort','rocks'):
     actual=row['after_state'][key];wanted=expected['after_state'][key]
     if key=='rocks':
      for a,b in zip(actual,wanted,strict=True):
       for name in ('qpos','qvel'):
        error=float(np.max(np.abs(np.asarray(a[name])-np.asarray(b[name]))));report['maximum_state_absolute_error']=max(report['maximum_state_absolute_error'],error)
        assert error<=1e-12,f'recorded_rock_mismatch:{n}:{name}'
        if error and n not in report['non_bitwise_frames']:report['non_bitwise_frames'].append(n)
     else:
      error=float(np.max(np.abs(np.asarray(actual)-np.asarray(wanted))));report['maximum_state_absolute_error']=max(report['maximum_state_absolute_error'],error)
      assert error<=1e-12,f'recorded_state_mismatch:{n}:{key}'
      if error and n not in report['non_bitwise_frames']:report['non_bitwise_frames'].append(n)
    force=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n']
    assert force==expected['bucket_mpm_force_norm_n'],f'recorded_force_mismatch:{n}'
    assert row['rock_bucket_contact_substeps']==expected['rock_bucket_contact_substeps']
    report['matched_recorded_frames']=n
    gap=None;observation=None
    if before in ('lift','off_soil'):
     t=time.perf_counter();q=r.fixture.state.particle_q.numpy();mech=np.array(row['after_state']['mechanism_qpos'])
     pose=r.assembly.assemble(mech,include_meshes=False)[1]['bucket'];observation=observer.observe(q,pose)
     observation['wall_seconds']=time.perf_counter()-t;gap=observation['remaining_soil_gap_m'];report['observed_gap_frames']+=1
    phase=router.observe_end(frame_end=n,soil_force_n=force,rock_contact_substeps=row['rock_bucket_contact_substeps'],mouth_angle_deg=f['mouth_chord_from_down_deg'],actual_boom_cylinder_m=f['q_boom'],bucket_origin_z_m=f['origin_z_m'],remaining_soil_gap_m=gap,gap_frame_end=n if gap is not None else None)
    entry=dict(frame=n,phase=phase,observation=observation)
    with (OUT/'phase-trace.ndjson').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
    if before=='lift' and gap is not None and gap>=.003:
     snap=OUT/f'clear-candidate-{n:06d}.npz';np.savez_compressed(snap,particle_q=q,mechanism_qpos=mech)
     # At most the five confirming candidate frames for this first transition.
     entry=dict(frame=n,path=str(snap),sha256=sha(snap))
     report.setdefault('clear_candidate_snapshots',[]).append(entry)
    if phase['event']:
     report['events'].append(phase['event']);print('EVENT',n,phase['event']['transition'],flush=True)
    if n%120==0:
     write(OUT/'report.json',report);print('MATCHED_FRAME',n,'phase',router.phase,'wall_s',round(time.perf_counter()-start,1),flush=True)
   with np.load(e['endpoint']['snapshot']['path'],allow_pickle=False) as original:
    endpoint=r.capture();report['final_arrays_exact']={k:bool(np.array_equal(v,original[k])) for k,v in endpoint.items()}
   assert len(report['final_arrays_exact'])==24 and all(report['final_arrays_exact'].values()),'teacher_final_arrays_not_exact'
   report['final_checkpoint']=r.save(OUT/'final.npz',stage='hold')
   report['status']='teacher_replay_completed_endpoint_bitwise_equal';report['frames_completed']=r.frames
 except BaseException as error:
  report.update(status='failed',error=repr(error),frames_completed=r.frames if r else 0)
  raise
 finally:
  report['wall_s']=time.perf_counter()-start
  report['source_unchanged']=sha(src)==raw['source_checkpoint']['sha256']
  report['original_endpoint_unchanged']=sha(Path(e['endpoint']['snapshot']['path']))==e['endpoint']['snapshot']['sha256']
  if OUT.exists():write(OUT/'report.json',report)

if __name__=='__main__':main()
