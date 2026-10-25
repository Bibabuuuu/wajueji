"""Build the approved initial 1:1 pool; preserve exact real transition labels."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from manual_residual_sac_env import ROOT, OBS_SIZE, sha, write


def select_old_indices(dones, count=1350, baseline_count=400, seed=42):
    dones=np.asarray(dones).reshape(-1)
    if len(dones)<count or count<baseline_count or not dones[baseline_count-1]:
        raise ValueError('old_pool_baseline_or_size')
    required=np.union1d(np.arange(baseline_count),np.flatnonzero(dones))
    if len(required)>count:raise ValueError('too_many_required_rows')
    candidates=np.setdiff1d(np.arange(len(dones)),required)
    selected=np.random.default_rng(seed).choice(candidates,count-len(required),replace=False)
    return np.sort(np.concatenate([required,selected]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('immutable_mix_exists')
    collection=ROOT/'runs/manual-recovery-collection-20261013-v1'
    aligned=ROOT/'runs/manual-residual-sac-aligned-20261013'
    index=json.loads((collection/'dataset-index.json').read_text())
    transfer=json.loads((aligned/'reward-transfer-v2/transfer.json').read_text())
    audit=json.loads((collection/'independent-audit.json').read_text())
    assert index['environment_sha256']==sha(ROOT/'src/v4/manual_residual_sac_env.py')
    assert sha(collection/'independent-audit.json')==index['source_audit_sha256']
    assert sha(collection/'review.json')==index['source_review_sha256']
    assert audit['status']=='passed'
    new_path=Path(index['replay_path']);old_path=aligned/'batch-v2/replay-buffer.pkl'
    assert sha(new_path)==index['replay_sha256']
    sources={str(p):sha(p) for p in [new_path,old_path,collection/'dataset-index.json']}
    with new_path.open('rb') as f:new=pickle.load(f)
    with old_path.open('rb') as f:old=pickle.load(f)
    assert new.size()==1350 and old.size()==11084
    for pool in [new,old]:
        assert not pool.full and pool.n_envs==1 and not pool.optimize_memory_usage
        assert pool.observations.shape[-1]==OBS_SIZE and pool.actions.shape[-1]==5
        assert not pool.handle_timeout_termination and not pool.timeouts[:pool.size()].any()
        for name in ['observations','next_observations','actions','rewards','dones']:
            assert np.isfinite(getattr(pool,name)[:pool.size()]).all()
    assert int(new.dones[:new.size()].sum())==3
    assert not old.actions[:400].any() and int(old.dones[:400].sum())==1 and old.dones[399,0]==1
    assert np.array_equal(old.next_observations[:399],old.observations[1:400])
    indices=select_old_indices(old.dones[:old.size()])
    mixed=ReplayBuffer(30000,new.observation_space,new.action_space,device='cpu',
                       n_envs=1,handle_timeout_termination=False)
    for pool,rows in [(new,np.arange(new.size())),(old,indices)]:
        for i in rows:
            mixed.add(pool.observations[i],pool.next_observations[i],pool.actions[i],
                      pool.rewards[i],pool.dones[i],[{}])
    assert mixed.size()==2700 and mixed.buffer_size==30000
    checks={}
    for name in ['observations','next_observations','actions','rewards','dones','timeouts']:
        expected=np.concatenate([getattr(new,name)[:new.size()],getattr(old,name)[indices]])
        checks[name+'_exact']=bool(np.array_equal(getattr(mixed,name)[:2700],expected))
    assert all(checks.values())
    args.output.mkdir(parents=True)
    replay=args.output/'initial-replay.pkl'
    with replay.open('wb') as f:pickle.dump(mixed,f)
    with replay.open('rb') as f:loaded=pickle.load(f)
    assert loaded.size()==2700 and np.array_equal(loaded.rewards,mixed.rewards)
    assert all(sha(p)==expected for p,expected in sources.items())
    receipt=dict(status='passed',receiver_env_sha256=index['environment_sha256'],
                 replay_path=str(replay.resolve()),replay_sha256=sha(replay),
                 baseline=transfer['baseline'],fresh_actor_and_critic=True,
                 purpose='approved_recovery_initial_pool_mix',initial_new_to_old_ratio=[1,1],
                 transitions=2700,new_transitions=1350,old_transitions=1350,
                 old_selected_indices=indices.tolist(),all_old_terminals_retained=True,
                 complete_manual_baseline_retained=True,new_failure_retained=True,
                 exact_copy_checks=checks,sources=sources,seed=42,buffer_capacity=30000,
                 reward_relabelled=False,neural_updates=0,new_physics_frames=0,
                 ratio_scope='initial pool only; uniform replay; online additions change ratio',
                 source_old_terminals=int(old.dones[:old.size()].sum()),
                 mixed_terminals=int(mixed.dones[:2700].sum()))
    write(args.output/'transfer.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['baseline','old_selected_indices','sources']}))


if __name__=='__main__':main()
