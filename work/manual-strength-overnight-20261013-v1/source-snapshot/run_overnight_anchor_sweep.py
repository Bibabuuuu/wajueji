"""Serial bounded model selection; frozen three-reset tests stay out of training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/qunwang/wajueji')
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
from manual_residual_sac_env import Budget, sha, write

out=ROOT/'runs/manual-strength-overnight-20261014/anchor-sweep-v1'
out.mkdir(exist_ok=False)
budget_path=ROOT/'runs/manual-strength-overnight-20261013-budget.json'
budget=Budget(budget_path,seconds=28800);budget.check()
grid=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/review.json'
grid_data=json.loads(grid.read_text())
assert grid_data['status']=='passed'
trials={x['name']:x for x in grid_data['trials']}
order=['seed42-weight50','seed42-weight10','seed44-weight50','seed43-weight100','seed42-weight300',
       'seed43-weight50','seed44-weight100','seed43-weight300','seed42-weight30']
review=dict(status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),budget_path=str(budget_path),
    deadline=budget.data['epoch_deadline'],evaluation_reserve_seconds=3600,grid_receipt=dict(path=str(grid),sha256=sha(grid)),
    selected_model=None,trials=[],new_gradient_updates=0,held_out_tests_not_used_for_training=True)
write(out/'review.json',review)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
env['PYTHONPATH']='local-data/runtime/sb3-2.9.0-py312:src/v4:.'
python='/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'
evaluator=ROOT/'src/v4/evaluate_manual_residual_frozen.py'
audit=ROOT/'work/manual-strength-overnight-20261013-v1/source-snapshot/audit_overnight_frozen.py'


def run_case(name,checkpoint,receipt,repetitions):
    path=out/name
    args=[python,str(evaluator),'--output',str(path),'--model',str(checkpoint),'--receipt',str(receipt),
          '--budget',str(budget_path),'--repetitions',str(repetitions),'--reserve','3600']
    with (out/(name+'.log')).open('w') as log:
        proc=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        return dict(status='failed',returncode=proc.returncode,path=str(path))
    with (out/(name+'-audit.log')).open('w') as log:
        proc=subprocess.run([python,str(audit),str(path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        return dict(status='audit_error',returncode=proc.returncode,path=str(path))
    report=json.loads((path/'read-only-audit.json').read_text())
    return dict(status=report['status'],path=str(path),audit_sha256=sha(path/'read-only-audit.json'),
                episodes=report['episodes'],goal_met=report['goal_met'],stretch_met=report['stretch_met'])


try:
    for name in order:
        budget.check()
        if budget.remaining()<3600+1500:
            review['selection_stopped_for_reserve']=True;break
        trial=trials[name]
        assert trial['cpu_preflight'] and sha(trial['checkpoint'])==trial['sha256']
        result=run_case(name,trial['checkpoint'],grid,1)
        review['trials'].append(dict(name=name,**result));write(out/'review.json',review)
        print('SWEEP_RESULT',name,result['status'],[(x['status'],x['fill_percent']) for x in result.get('episodes',[])],flush=True)
        if result['status']!='passed':continue
        ep=result['episodes'][0]
        if not ep['success']:continue
        if review['selected_model'] is None or ep['fill_percent']>review['selected_model']['screened_fill_percent']:
            review['selected_model']=dict(name=name,path=trial['checkpoint'],sha256=trial['sha256'],screened_fill_percent=ep['fill_percent'])
            write(out/'review.json',review)
        if ep['fill_percent']>=147.38132234025687:
            verification=run_case(name+'-independent-three',trial['checkpoint'],grid,3)
            review['verification']=verification;write(out/'review.json',review)
            if verification.get('stretch_met'):
                review['status']='goal_and_stretch_verified';break
    else:review['status']='screening_completed'
    if review['status']=='running':review['status']='screening_stopped'
except BaseException as error:
    review.update(status='failed',error=type(error).__name__+':'+str(error));raise
finally:
    review.update(remaining_seconds=budget.remaining(),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(out/'review.json',review)
