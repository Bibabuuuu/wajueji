"""Eight-hour unattended paired digging search. No polling agent is required."""
import argparse,fcntl,json,os,signal,subprocess,time,shutil
from pathlib import Path
from sources import *
from parameters import BASELINE,CANDIDATES as PARAMETER_CANDIDATES
from failures import TrialRejected,classify
from paired_score import better,assess
from artifacts import export_replay
TAIL=ROOT/'work/night-dual-tail-20260925-v1'
PREP=ROOT/'work/night-dual-start-preparation-20260925-v2'
CANDIDATES=[dict(name=c['name'],parameters={k:v for k,v in c.items() if k!='name'}) for c in PARAMETER_CANDIDATES]
# Two full worst-case confirmations plus score/export/termination allowance.
PAIR_RESERVE_SECONDS=7200
ONE_TRIAL_BUDGET=990+120+2090+180

def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def verify(manifest):
 for p,h in manifest['bindings'].items():
  if sha(p)!=h:raise ValueError('frozen_dual_input_changed:'+p)
def preparation_admission():
 first_source();second=second_evidence();prior=read(ROOT/'runs/night-dual-start-preparation-20260925-v2/entry-review.json')
 if prior['status']!='entry_checks_passed' or not prior['first_120']['passed'] or prior['first_120']['reference_sha256']!=FIRST_GATE_SHA:
  raise ValueError('dual_entry_checks_not_passed')
 for p,h in read(PREP/'manifest.json')['bindings'].items():
  if sha(p)!=h:raise ValueError('dual_entry_binding_changed:'+p)
 for p,h in read(PREP/'result-review.json')['frozen_report_bindings'].items():
  if sha(p)!=h:raise ValueError('dual_entry_evidence_changed:'+p)
 return prior['wall_seconds']+read(ROOT/'runs/night-dual-start-preparation-20260925-v1/entry-review.json')['wall_seconds']+second['prior_check_seconds']
def prepare():
 if (RUN/'batch.json').exists():raise ValueError('cannot_rebind_started_dual_batch')
 prior=preparation_admission();paths=[*W.glob('*.py'),*TAIL.glob('*.py'),PREP/'manifest.json',PREP/'result-review.json']
 paths.extend(Path(p) for p in read(PREP/'manifest.json')['bindings'])
 paths.extend(Path(p) for p in read(PREP/'result-review.json')['frozen_report_bindings'])
 for spec in SOURCE_SPECS.values():
  paths.extend([spec['protocol'],spec['source'],Path(str(spec['source'])+'.json'),spec['gate']])
  for row in read(spec['protocol'])['inputs']:
   if sha(row['path'])!=row['sha256']:raise ValueError('source_protocol_changed:'+row['path'])
   paths.append(Path(row['path']))
 save(W/'manifest.json',dict(status='authorized_current_physics_paired_search',maximum_total_wall_seconds=28800,
   prior_check_seconds=prior,remaining_seconds=28800-prior,candidates=CANDIDATES,maximum_parameter_pairs=len(CANDIDATES),
   maximum_baseline_trials=2,maximum_candidate_trials=2*len(CANDIDATES),maximum_confirmation_trials=2,
   confirmation_reserve_seconds=PAIR_RESERVE_SECONDS,physics='frozen_D152_full_response_60Hz_no_multiccd',
   scope='Only curl_load below95pct BC scaling; two saved prepoints; no neural weight updates; fixed tail',
   automatic_retry=False,action_guard_failure='reject_candidate_and_continue',unknown_or_integrity_failure='stop_entire_batch',formal_qualification=False,
   bindings={str(p):sha(p) for p in sorted(set(paths))}))
 print('PREPARED_DUAL_SEARCH',round(28800-prior,3))

def environment():
 return {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src/v4')+':'+str(ROOT),
   'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}

def child(command,env,log,timeout,deadline):
 seconds=min(timeout,deadline-time.monotonic()-50)
 if seconds<=0:raise TimeoutError('global_budget_preserve_exit')
 with Path(log).open('x') as output:
  proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=seconds)
  except BaseException:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGINT)
    try:proc.wait(timeout=45)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
  if code:raise RuntimeError('trial_process_failed:'+str(log)+':'+str(code))

def trial(pair_name,kind,params,deadline,baseline=False,mode='dig_only'):
 try:return execute_trial(pair_name,kind,params,deadline,baseline,mode)
 except (RuntimeError,ValueError) as error:
  if not baseline:
   detail=classify(RUN/pair_name/kind)
   if detail is not None:
    detail.update(trial=pair_name+'/'+kind,process_error=repr(error))
    save(RUN/pair_name/kind/'rejection.json',detail)
    raise TrialRejected(detail) from error
  raise

def execute_trial(pair_name,kind,params,deadline,baseline=False,mode='dig_only'):
 spec=SOURCE_SPECS[kind];out=RUN/pair_name/kind;out.mkdir(parents=True,exist_ok=False)
 config=dict(status='authorized_bounded_dual_search',trial_id=pair_name+'/'+kind,source_kind=kind,parameters=params,verify_baseline_prefix=baseline)
 save(out/'config.json',config);env={**environment(),'NIGHT_TRIAL_CONFIG':str(out/'config.json')}
 contract=read(spec['protocol']);contract['search_parameters']=params
 contract['inputs'] += [dict(path=str(p),sha256=sha(p)) for p in [*W.glob('*.py'),out/'config.json']]
 save(out/'contact-protocol.json',contract)
 child([PHYS,str(W/'contact_run.py'),'--source',str(spec['source']),'--policy',str(POLICY),'--policy-sha256',POLICY_SHA,
  '--output',str(out/'contact'),'--frames','3000','--training-python',TRAIN,'--coupling-hz','60','--capture-display',
  '--phase-contract',str(out/'contact-protocol.json')],env,out/'contact.log',990,deadline)
 if baseline and not read(out/'contact/baseline-prefix-review.json')['passed']:raise ValueError('baseline_prefix_missing')
 env.update(NIGHT_EVALUATION_MODE=mode,NIGHT_CONTACT_SOURCE=str(out/'contact/final.npz'),
   NIGHT_TAIL_PROTOCOL=str(out/'tail-protocol.json'),NIGHT_TAIL_OUTPUT=str(out/'tail'))
 child([PHYS,str(TAIL/'prepare_contract.py')],env,out/'tail-prepare.log',120,deadline)
 child([PHYS,str(TAIL/'run.py')],env,out/'tail.log',2090,deadline)
 child([PHYS,str(W/'score.py'),str(out/'contact'),str(out/'tail'),str(out/'score.json')],env,out/'score.log',180,deadline)
 score=read(out/'score.json')
 if not score['eligible']:raise ValueError('trial_not_eligible:'+pair_name+'/'+kind)
 score['source_sha256']=spec['sha'];save(out/'score.json',score)
 export_replay(out/'replay.json',out/'contact',out/'tail',pair_name+' | '+kind,'Independent saved-source trial, not continuous scoops')
 return score

def confirmation_passed(measured,candidate,baseline):
 for kind in ('first','second'):
  s=measured[kind]
  if (not s.get('eligible') or not s.get('unload_validated') or s.get('receipt_fraction') is None
      or s['receipt_fraction']<.98 or s['working_fill_percent']<candidate[kind]['working_fill_percent']-2
      or s['working_fill_percent']<baseline[kind]['working_fill_percent']-2):return False
 return True

def publish(report):
 rows=['# 双起点夜间挖掘参数搜索','',f"状态：{report['status']}",'固定D152物理；只调挖掘；不更新神经网络权重。','',
  '| 参数组 | 起点 | 工作满斗率% | 挖掘仿真秒 | 状态 |','| --- | --- | ---: | ---: | --- |']
 for entry in report.get('trials',[]):
  for kind,s in entry['scores'].items():rows.append(f"| {entry['name']} | {kind} | {s['working_fill_percent']:.3f} | {s['seconds']:.3f} | {entry['selection_status']} |")
 best=report.get('best')
 if best:
  save(RUN/'best.json',best);rows+=['',f"选择：{best['name']}，{best['selection_status']}",json.dumps(best['parameters'],ensure_ascii=False)]
  pair='baseline' if best['name']=='baseline' else 'confirm-best'
  for kind in ('first','second'):
   p=RUN/pair/kind/'replay.json'
   if p.exists():(RUN/('selected-'+kind+'-replay.json')).write_text(p.read_text())
 if report.get('failure'):rows+=['','停止原因：'+report['failure']]
 for rejection in report.get('rejections',[]):rows+=['',f"淘汰 {rejection['trial']}：{rejection['reason']}"]
 if report.get('unconfirmed_candidate'):rows+=['','存在未复验候选，未选作可用参数。']
 rows+=['','原始基线/示范/失败记录保持；不自动部署；不授予95%或连续三铲资格。']
 (RUN/'comparison.md').write_text('\n'.join(rows)+'\n')

def run():
 manifest=read(W/'manifest.json');verify(manifest);preparation_admission()
 if manifest['status']!='authorized_current_physics_paired_search':raise ValueError('dual_search_not_authorized')
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'batch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/'batch.json').exists():raise ValueError('immutable_batch_exists_no_retry')
  started=time.monotonic();started_at=time.time();deadline=started+manifest['remaining_seconds']
  report=dict(status='running',pid=os.getpid(),started_at_unix=started_at,deadline_unix=started_at+manifest['remaining_seconds'],
   prior_check_seconds=manifest['prior_check_seconds'],manifest_sha256=sha(W/'manifest.json'),trials=[],
   neural_training=False,automatic_retry=False,agent_monitoring=False,physics=manifest['physics'],rejections=[])
  save(RUN/'batch.json',report)
  def interrupted(signum,frame):raise TimeoutError('batch_deadline_or_external_stop:'+str(signum))
  signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
  signal.setitimer(signal.ITIMER_REAL,manifest['remaining_seconds']-50)
  baseline=None;best=None
  def pair(name,parameters,baseline_run=False,mode='dig_only'):
   scores={}
   for kind in ('first','second'):
    verify(manifest)
    if shutil.disk_usage(ROOT).free<55*1024**3:raise RuntimeError('disk_reserve_55GiB')
    report['active_trial']=name+'/'+kind;report['partial_pair_scores']=scores;save(RUN/'batch.json',report)
    scores[kind]=trial(name,kind,parameters,deadline,baseline=baseline_run,mode=mode)
    report['partial_pair_scores']=scores;save(RUN/'batch.json',report)
   return scores
  try:
   scores=pair('baseline',BASELINE,True)
   baseline=dict(name='baseline',parameters=BASELINE,scores=scores,selection_status='baseline')
   best=baseline.copy();report['trials'].append(baseline);report['best']=best;publish(report);save(RUN/'batch.json',report)
   for candidate in manifest['candidates']:
    if deadline-time.monotonic()<2*ONE_TRIAL_BUDGET+PAIR_RESERVE_SECONDS:
     report['stop_reason']='reserve_for_both_confirmations_and_save';break
    try:scores=pair(candidate['name'],candidate['parameters'])
    except TrialRejected as error:
     report['rejections'].append(error.detail);save(RUN/'batch.json',report);publish(report)
     continue
    entry={**candidate,'scores':scores,'selection_status':'unconfirmed'}
    report['trials'].append(entry)
    if better(scores,best['scores'],baseline['scores']):best=entry
    report['best']=baseline
    if best['name']!='baseline':report['unconfirmed_candidate']=best
    publish(report);save(RUN/'batch.json',report)
   if best['name']!='baseline':
    if deadline-time.monotonic()>=2*ONE_TRIAL_BUDGET+120:
     try:confirmed=pair('confirm-best',best['parameters'],mode='unload_validation')
     except TrialRejected as error:
      report['rejections'].append(error.detail);confirmed=None
     report['confirmation']=confirmed
     if confirmed is not None and confirmation_passed(confirmed,best['scores'],baseline['scores']):
      best={**best,'selection_status':'confirmed_two_source_development_candidate','confirmation_scores':confirmed};report.pop('unconfirmed_candidate',None)
     else:report['confirmation_rejected']=True;best=baseline
    else:report['stop_reason']='insufficient_confirmation_budget';best=baseline
   report.update(status='completed',best=best,active_trial=None)
  except BaseException as error:
   report.update(status='stopped',failure=repr(error))
   if baseline is not None:report['best']=baseline
   raise
  finally:
   signal.setitimer(signal.ITIMER_REAL,0)
   report.update(finished_at_unix=time.time(),wall_seconds=time.monotonic()-started)
   save(RUN/'batch.json',report)
   try:publish(report)
   except Exception as error:report['export_error']=repr(error);save(RUN/'batch.json',report)

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
 prepare() if a.prepare else run()
if __name__=='__main__':main()
