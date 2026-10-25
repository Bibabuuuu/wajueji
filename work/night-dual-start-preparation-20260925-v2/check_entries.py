"""Bounded entry qualification. Deliberately has no overnight-search command."""
import argparse,fcntl,json,os,signal,subprocess,time
import numpy as np
from sources import *

def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def verify(manifest):
 for p,h in manifest['bindings'].items():
  if sha(p)!=h:raise ValueError('frozen_entry_input_changed:'+p)
def prepare():
 if (RUN/'entry-review.json').exists():raise ValueError('entry_check_already_started')
 validate_first_loop();reused_zero();first_source();second=second_evidence();protocol=read(FIRST_CODE/'first-protocol.json')
 for item in protocol['inputs']:
  if sha(item['path'])!=item['sha256']:raise ValueError('original_first_protocol_input_changed:'+item['path'])
 paths=[*W.glob('*.py'),FIRST,FIRST_GATE,Path(str(FIRST)+'.json'),POLICY,FIRST_CODE/'first-protocol.json',
   SECOND_CODE/'manifest.json',SECOND_RUN/'precheck.json']
 paths.extend(Path(p) for p in read(SECOND_CODE/'manifest.json')['bindings'])
 paths.extend(Path(p) for p in read(SECOND_RUN/'precheck.json')['bindings'])
 paths.extend(Path(i['path']) for i in protocol['inputs'])
 paths.extend([ROOT/'work/night-dual-start-preparation-20260925-v1/manifest.json',ROOT/'runs/night-dual-start-preparation-20260925-v1/entry-review.json',ROOT/'runs/night-dual-start-preparation-20260925-v1/first-120/console.log'])
 save(W/'manifest.json',dict(status='entry_checks_only',physical_version='frozen_D152_no_multiccd',
   maximum_new_physical_frames=120,first_source_sha256=FIRST_SHA,first_reference_sha256=FIRST_GATE_SHA,
   second=second,overnight_search_enabled=False,budget_seconds=28800,
   bindings={str(p):sha(p) for p in sorted(set(paths))}))
 print('PREPARED: first zero reused + remaining 120, second unchanged evidence only; no search')
def child(command,log,timeout=240):
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src/v4')+':'+str(ROOT),
      'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
 with Path(log).open('x') as output:
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=timeout)
  except BaseException:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  if code:raise RuntimeError('entry_child_failed:'+str(log)+':'+str(code))
def trial(frames):
 out=RUN/('first-zero' if frames==0 else 'first-120');out.mkdir(exist_ok=False)
 config=dict(status='authorized_first_entry_check',frames=frames,parameters=dict(boom_scale=1.,arm_scale=1.,curl_scale=1.))
 save(out/'config.json',config)
 protocol=read(FIRST_CODE/'first-protocol.json');protocol['entry_qualification_only']=True
 protocol['inputs'] += [dict(path=str(p),sha256=sha(p)) for p in [*W.glob('*.py'),out/'config.json']]
 save(out/'protocol.json',protocol)
 cmd=[PHYS,str(W/'first_entry.py'),'--config',str(out/'config.json'),'--source',str(FIRST),'--policy',str(POLICY),
      '--policy-sha256',POLICY_SHA,'--output',str(out/'contact'),'--frames',str(frames),'--training-python',TRAIN,
      '--coupling-hz','60','--capture-display','--phase-contract',str(out/'protocol.json')]
 child(cmd,out/'console.log')
 r=read(out/'contact/report.json')
 if r['status']!=('zero_step_passed' if frames==0 else 'bounded_probe_completed') or r['frames_completed']!=frames:
  raise ValueError('first_entry_wrong_completion')
 if not all(r['zero_step_arrays_exact'].values()) or max(r['restore']['cycle_errors'].values())>1e-12:
  raise ValueError('first_restore_not_exact')
 reference=FIRST if frames==0 else FIRST_GATE
 with np.load(reference) as old,np.load(out/'contact/final.npz') as new:
  if set(old.files)!=set(new.files):raise ValueError('first_reference_keys_changed')
  exact={k:bool(np.array_equal(old[k],new[k])) for k in old.files}
 review=dict(passed=all(exact.values()) and r['all_executed_frames_safe'],frames=frames,arrays_exact=exact,
   endpoint_sha256=sha(out/'contact/final.npz'),reference_sha256=sha(reference),wall_seconds=r['wall_seconds'],
   native_restore_errors=r['restore']['cycle_errors'],source_unchanged=r['source_unchanged'],
   report_sha256=sha(out/'contact/report.json'))
 save(out/'identity-review.json',review)
 if not review['passed'] or not review['source_unchanged']:raise ValueError('first_entry_identity_failed')
 return review

def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--prepare',action='store_true');group.add_argument('--check-first',action='store_true');args=parser.parse_args()
 if args.prepare:prepare();return
 manifest=read(W/'manifest.json');verify(manifest);validate_first_loop();first_source();second=second_evidence()
 RUN.mkdir(exist_ok=True,parents=True)
 lock=(RUN/'entry.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 result=RUN/'entry-review.json'
 if result.exists():
  lock.close();raise ValueError('immutable_entry_check_already_started; no automatic retry')
 started=time.monotonic();review=dict(status='running',manifest_sha256=sha(W/'manifest.json'),started_at_unix=time.time(),
   second=second,new_physical_frame_budget=120,overnight_search_started=False,neural_training_started=False)
 save(result,review)
 try:
  review['first_zero']=reused_zero();save(result,review)
  verify(manifest)
  review['first_120']=trial(120)
  verify(manifest);second_evidence()
  review.update(status='entry_checks_passed',new_physical_frames=120,
    long_search_ready=False,pending=['decide_physics_version','align_two_full_baselines','dual_candidate_dispatch_and_confirmation'])
 except BaseException as error:
  review.update(status='stopped',failure=repr(error));raise
 finally:
  review.update(wall_seconds=time.monotonic()-started,finished_at_unix=time.time());save(result,review);lock.close()
 print(json.dumps(review,indent=2))
if __name__=='__main__':main()
