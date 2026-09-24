"""Bounded local parameter experiment. --prepare never starts physics."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,shutil,fcntl
from artifacts import export_replay, export_comparison
from baseline import CONTACT as BASE_CONTACT,PREFIX as BASE_PREFIX,TAIL as BASE_TAIL,readiness as baseline_readiness,evidence_paths
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;TAIL=ROOT/'work/night-search-tail-20260925-v2'
RUN=ROOT/'runs/night-parameter-search-20260925-v2'
PHYS=Path('/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python')
TRAIN='/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'
SOURCE=ROOT/'runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz';POLICY=ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
from parameters import BASELINE,CANDIDATES
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def prepare():
 if (RUN/'batch.json').exists():raise ValueError('cannot_rebind_started_batch')
 base=read(ROOT/'work/contact-next-scoop-20260925-v1/protocol.json')
 paths=[*W.glob('*.py'),*TAIL.glob('*.py'),SOURCE,Path(str(SOURCE)+'.json'),POLICY,
 ROOT/'configs/v4/live-bucket-fill-local-v1.json']
 issues=baseline_readiness()
 if not issues:paths.extend(evidence_paths())
 paths.extend((ROOT/'work/contact-second-fixed-route-20260925-v1').glob('*.py'))
 paths.extend((ROOT/'work/contact-second-fixed-route-review-20260925-v1').glob('*.py'))
 paths.append(ROOT/'work/contact-next-scoop-20260925-v1/audit_candidate.py')
 paths.extend(Path(x['path']) for x in base['inputs'])
 paths.extend((ROOT/'work/contact-second-finish-unload-20260925-v4').glob('*.py'))
 paths.extend(Path(p) for p in read(ROOT/'work/contact-second-finish-unload-20260925-v4/protocol.json')['inputs'])
 config=read(ROOT/'configs/v4/live-bucket-fill-local-v1.json')
 paths.extend(Path(config[k]) for k in ('measurement_source','contract','front_rim','calibration'))
 save(W/'manifest.json',dict(status='prepared_awaiting_baseline' if issues else 'prepared_not_launched',baseline_blockers=issues,budget_seconds=28800,maximum_candidates=6,
  source=str(SOURCE),source_sha256=sha(SOURCE),policy=str(POLICY),policy_sha256=sha(POLICY),
  baseline_contact=str(BASE_CONTACT),baseline_prefix=str(BASE_PREFIX),baseline_tail=str(BASE_TAIL),candidates=CANDIDATES,
  fixed_scope='digging only: same second prepoint; modify curl_load below95pct stroke only; candidate ends after CLEAR_PILE; only best repeats with fixed transport/unload validation',
  candidate_budget=dict(contact_frames=3000,contact_wall_seconds=900,tail_frames=6000,tail_wall_seconds=2000),
  reserve_seconds=3600,hard_stop_on_any_process_failure=True,formal_qualification=False,
  bindings={str(p):sha(p) for p in sorted(set(paths))}))
 print('PREPARED; no physics or training launched')
def verify(manifest):
 for p,h in manifest['bindings'].items():
  if sha(p)!=h:raise ValueError('frozen_input_changed:'+p)
def run_process(command,env,log,timeout):
 with Path(log).open('x') as output:
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise TimeoutError('bounded_child_timeout:'+str(log))
  except BaseException:
   os.killpg(proc.pid,signal.SIGINT)
   try:proc.wait(timeout=45)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  if code:raise RuntimeError('candidate_process_stopped:'+str(log)+':'+str(code))
def environment():
 return {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src/v4')+':'+str(ROOT),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
def trial(name,parameters,frames=3000,mode='dig_only'):
 out=RUN/name;out.mkdir(exist_ok=False)
 config=dict(status='authorized_bounded_parameter_search',trial_id=name,parameters=parameters,formal_qualification=False)
 save(out/'config.json',config);env={**environment(),'NIGHT_TRIAL_CONFIG':str(out/'config.json')}
 contract=read(ROOT/'work/contact-next-scoop-20260925-v1/protocol.json')
 contract['search_parameters']=parameters
 contract['inputs'] += [dict(path=str(p),sha256=sha(p)) for p in [*W.glob('*.py'),out/'config.json']]
 save(out/'contact-protocol.json',contract)
 cmd=[str(PHYS),str(W/'contact_run.py'),'--source',str(SOURCE),'--policy',str(POLICY),'--policy-sha256',sha(POLICY),'--output',str(out/'contact'),'--frames',str(frames),'--training-python',TRAIN,'--coupling-hz','60','--capture-display','--phase-contract',str(out/'contact-protocol.json')]
 run_process(cmd,env,out/'contact.log',240 if frames<=120 else 990)
 if frames<=120:return out
 env['NIGHT_EVALUATION_MODE']=mode
 env.update(NIGHT_CONTACT_SOURCE=str(out/'contact/final.npz'),NIGHT_TAIL_PROTOCOL=str(out/'tail-protocol.json'),NIGHT_TAIL_OUTPUT=str(out/'tail'))
 run_process([str(PHYS),str(TAIL/'prepare_contract.py')],env,out/'tail-prepare.log',120)
 run_process([str(PHYS),str(TAIL/'run.py')],env,out/'tail.log',2090)
 # Budget stop/invalid metric is not success and must not silently rank.
 run_process([str(PHYS),str(W/'score.py'),str(out/'contact'),str(out/'tail'),str(out/'score.json')],env,out/'score.log',180)
 if not read(out/'score.json')['eligible']:raise ValueError('candidate_not_eligible:'+name)
 export_replay(out/'replay.json',out/'contact',out/'tail',name,'Development digging parameter trial; not a three-scoop result')
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if args.prepare and not args.run:prepare();return
 if not args.run or args.prepare:raise ValueError('choose --prepare or --run')
 manifest=read(W/'manifest.json');verify(manifest)
 issues=baseline_readiness()
 if manifest.get('status')!='prepared_not_launched' or issues:raise ValueError('baseline_not_admitted; no batch launched:'+str(issues))
 RUN.mkdir(parents=True,exist_ok=True)
 lock=(RUN/'batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (RUN/'batch.json').exists():raise ValueError('immutable_batch_already_started; no automatic resume/reset of deadline')
 started=time.time();started_monotonic=time.monotonic();deadline=started_monotonic+28800
 report=dict(status='running',started_at_unix=started,deadline_unix=started+28800,trials=[],manifest_sha256=sha(W/'manifest.json'),neural_training=False)
 save(RUN/'batch.json',report)
 def remaining():return deadline-time.monotonic()
 try:
  run_process([str(PHYS),str(W/'score.py'),str(BASE_CONTACT),str(BASE_TAIL),str(RUN/'baseline-score.json'),str(BASE_PREFIX)],environment(),RUN/'baseline-score.log',180)
  baseline=read(RUN/'baseline-score.json')
  if not baseline['eligible']:raise ValueError('baseline_not_eligible_for_working_volume_comparison')
  baseline_entry=dict(name='baseline',parameters=BASELINE,score=baseline,selection_status='baseline')
  best=baseline_entry.copy();save(RUN/'best.json',best)
  export_replay(RUN/'baseline-replay.json',BASE_CONTACT,BASE_TAIL,'BASELINE','Three continuous segments; original prefix planner stop preserved',prefix=BASE_PREFIX)
  smoke=trial('smoke-baseline-120',BASELINE,frames=120)
  # Same executable environment, source and parameters must reproduce old physics.
  import numpy as np
  with np.load(smoke/'contact/final.npz') as new,np.load(BASE_CONTACT/'gate-120.npz') as old:
   if set(new.files)!=set(old.files):raise ValueError('baseline_adapter_array_keys_changed')
   exact={k:bool(np.array_equal(new[k],old[k])) for k in old.files}
  save(smoke/'identity-review.json',dict(arrays_exact=exact,passed=all(exact.values())))
  if not all(exact.values()):raise ValueError('baseline_adapter_smoke_not_identical')
  from score import better
  for item in manifest['candidates']:
   if remaining()<990+120+2090+180+3600:report['stop_reason']='reserve_for_comparison_and_save';break
   if shutil.disk_usage(ROOT).free<55*1024**3:report['stop_reason']='disk_reserve_55GiB';break
   verify(manifest)
   name=item['name'];params={k:v for k,v in item.items() if k!='name'}
   report['active_trial']=name;save(RUN/'batch.json',report)
   out=trial(name,params);score=read(out/'score.json')
   entry=dict(name=name,parameters=params,score=score,selection_status='unconfirmed');report['trials'].append(entry)
   if better(score,best['score']):best=entry;save(RUN/'best.json',best)
   save(RUN/'batch.json',report)
  # Re-run only the selected improved parameter set as an explicitly planned confirmation.
  # A protected failure anywhere aborts; there are no retries.
  if best['name']!='baseline' and remaining()>990+120+2090+180+120:
   report['active_trial']='confirm-best';save(RUN/'batch.json',report)
   confirmed=trial('confirm-best',best['parameters'],mode='unload_validation');check=read(confirmed/'score.json')
   report['confirmation']=check
   if not check['eligible'] or check['working_fill_percent']<best['score']['working_fill_percent']-2. or check['receipt_fraction']<baseline['receipt_fraction']-.02:
    report['unconfirmed_candidate']=best
    best={**baseline_entry,'rejected_candidate':'confirmation_failed'}
   else:best.update(confirmation_tail=str(confirmed/'tail'),confirmation_score=check,selection_status='confirmed_development_candidate')
   save(RUN/'best.json',best)
  elif best['name']!='baseline':
   report['unconfirmed_candidate']=best;best={**baseline_entry,'reason':'no_budget_for_confirmation'};save(RUN/'best.json',best)
  report.update(status='completed',best=best,active_trial=None)
 except BaseException as error:
  report.update(status='stopped',failure=repr(error))
  if 'baseline_entry' in locals():
   if best['name']!='baseline':report['unconfirmed_candidate']=best
   best={**baseline_entry,'reason':'batch_stopped'};report['best']=best;save(RUN/'best.json',best)
  raise
 finally:
  report.update(finished_at_unix=time.time(),wall_seconds=time.monotonic()-started_monotonic);save(RUN/'batch.json',report)
  try:export_comparison(RUN,report)
  except Exception as error:
   report['comparison_export_error']=repr(error);save(RUN/'batch.json',report)
if __name__=='__main__':main()
