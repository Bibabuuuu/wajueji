"""Independent original-reset frozen inference; shared budget and full causal logs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import ManualResidualEnv, Budget, environment_contract, require_environment_contract, sha, write
from run_manual_residual_sac import episode
from collect_manual_recovery import save_transitions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--budget',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=1)
    p.add_argument('--reserve',type=float,default=3600.)
    a=p.parse_args()
    if a.repetitions<1 or a.reserve<0:raise ValueError('evaluation_parameters')
    receipt=json.loads(a.receipt.read_text());require_environment_contract(receipt,'error_strength_v1')
    budget=Budget(a.budget,seconds=28800);budget.check()
    if budget.data['seconds']!=28800 or budget.data['purpose']!='training_and_evaluation':raise ValueError('wrong_shared_budget')
    if a.output.exists():raise ValueError('immutable_evaluation_exists')
    torch.set_num_threads(1)
    frozen=SAC.load(a.model,device='cpu')
    before_updates=frozen._n_updates
    a.output.mkdir(parents=True)
    code={name:sha(ROOT/'src/v4'/name) for name in ['evaluate_manual_residual_frozen.py','run_manual_residual_sac.py',
        'manual_anchored_sac.py','manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py','collect_manual_recovery.py']}
    snap=a.output/'source-snapshot';snap.mkdir()
    for name in code:(snap/name).write_bytes((ROOT/'src/v4'/name).read_bytes())
    review=dict(status='running',environment_contract=environment_contract('error_strength_v1'),code=code,
        policy_checkpoint=dict(path=str(a.model),sha256=sha(a.model),updates=before_updates),
        budget_path=str(a.budget),budget_deadline=budget.data['epoch_deadline'],reserve_seconds=a.reserve,
        receipt=dict(path=str(a.receipt),sha256=sha(a.receipt)),requested_repetitions=a.repetitions,episodes=[],neural_updates=0)
    write(a.output/'review.json',review)
    env=None
    try:
        for i in range(a.repetitions):
            if budget.remaining()<a.reserve+330:
                review['not_started_insufficient_budget']=i+1;break
            env=ManualResidualEnv(a.output/f'repetition-{i+1}',budget,waiting_action_mode='error_strength_v1',collect=True)
            result=episode(env,lambda obs:frozen.predict(obs,deterministic=True)[0])
            records=[json.loads(s) for s in (env.directory/'policy.ndjson').read_text().splitlines()]
            for row in records:row['policy_source']='frozen_learned_policy'
            dataset=save_transitions(env,records,env.directory/'transitions.npz')
            dataset.pop('recovery_steps',None);dataset['environment_contract']=environment_contract('error_strength_v1')
            review['episodes'].append(dict(result=result,dataset=dataset))
            write(a.output/'review.json',review)
            print('FROZEN_RESULT',i+1,result['status'],result['physical_frames'],result.get('retained_fill_fraction'),flush=True)
        review['status']='completed' if len(review['episodes'])==a.repetitions else 'incomplete_budget'
        assert frozen._n_updates==before_updates and sha(a.model)==review['policy_checkpoint']['sha256']
    except BaseException as error:
        review.update(status='failed',error=type(error).__name__+':'+str(error));raise
    finally:
        if env is not None:env.close()
        review.update(remaining_seconds=budget.remaining(),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(a.output/'review.json',review)


if __name__=='__main__':main()
