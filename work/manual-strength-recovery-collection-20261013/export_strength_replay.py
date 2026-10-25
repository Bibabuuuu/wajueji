"""Package only audited, new-contract real data. No SAC model is created."""
import json
import pickle
from pathlib import Path
import shutil
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from manual_residual_sac_env import environment_contract, require_environment_contract, sha, write, OBS_SIZE

root=Path('/home/qunwang/wajueji')
source=root/'runs/manual-strength-recovery-collection-20261013-v1'
out=source/'training-ready'
out.mkdir(exist_ok=False)
review=json.loads((source/'review.json').read_text())
audit=json.loads((source/'read-only-audit.json').read_text())
require_environment_contract(review)
assert review['status']=='completed' and review['collection_enabled']
assert audit['status']=='passed' and all(audit['global_checks'].values())
assert len(review['episodes'])==4 and all(e['audit_passed'] for e in audit['episodes'])
contract=environment_contract();paths=[source/'review.json',source/'read-only-audit.json']
for e in review['episodes']:
    dataset=e['dataset'];require_environment_contract(dataset)
    assert dataset['complete'] and sha(dataset['path'])==dataset['sha256']
    paths.append(Path(dataset['path']))
bindings={str(p):sha(p) for p in paths}
replay=ReplayBuffer(30000,gym.spaces.Box(-10.,10.,(OBS_SIZE,),np.float32),
                    gym.spaces.Box(-1.,1.,(5,),np.float32),device='cpu',
                    n_envs=1,handle_timeout_termination=False)
expected={k:[] for k in ['observations','next_observations','actions','rewards','dones']}
counts={}
for e in review['episodes']:
    with np.load(e['dataset']['path'],allow_pickle=False) as data:
        n=len(data['actions']);assert n==e['dataset']['count']
        counts[e['label']]=n
        for key in expected:expected[key].append(data[key].astype(np.float32))
        for i in range(n):
            replay.add(data['observations'][i:i+1],data['next_observations'][i:i+1],
                       data['actions'][i:i+1],data['rewards'][i:i+1],data['dones'][i:i+1],[{}])
size=sum(counts.values());checks={}
for key,parts in expected.items():
    wanted=np.concatenate(parts);stored=getattr(replay,key)[:size]
    checks[key+'_exact_after_buffer_dtype_conversion']=np.array_equal(stored.reshape(wanted.shape),wanted)
checks['all_four_terminals_retained']=int(replay.dones[:size].sum())==4
checks['all_new_no_old_rows']=replay.size()==size and not replay.full
checks['contractual_timeout_not_bootstrapped']=not replay.handle_timeout_termination and not replay.timeouts[:size].any()
assert all(checks.values())
target=out/'initial-replay.pkl'
with target.open('wb') as f:pickle.dump(replay,f)
with target.open('rb') as f:loaded=pickle.load(f)
checks['serialized_reload_exact']=loaded.size()==size and all(np.array_equal(getattr(loaded,k),getattr(replay,k)) for k in expected)
checks['source_data_unchanged']=all(sha(p)==h for p,h in bindings.items())
assert all(checks.values())
receipt=dict(status='passed',environment_contract=contract,
             receiver_env_sha256=contract['code']['manual_residual_sac_env.py'],
             replay_path=str(target),replay_sha256=sha(target),
             baseline=review['episodes'][0]['result'],transitions=size,counts=counts,
             fresh_actor_and_critic=True,old_transitions=0,reward_relabelled=False,
             buffer_capacity=30000,terminals=4,checks=checks,sources=bindings,
             neural_updates=0,new_physics_frames=0,
             purpose='fresh_strength_recovery_pool_only; fresh networks are required when training is authorized')
write(out/'transfer.json',receipt)
write(out/'training-admission.json',dict(status='ready',environment_contract=contract,code=contract['code'],
      source_evidence={**bindings,str(out/'transfer.json'):sha(out/'transfer.json')},
      runtime_bindings=receipt['baseline']['runtime_bindings'],collection_admission=True,
      training_admission=True,training_budget_authorized=False,
      meaning='Data/control admission passed; no training permission or budget is granted by this file.'))
write(out/'package-review.json',dict(status='passed',checks=checks,transitions=size,counts=counts,
      training_data_ready=True,training_budget_authorized=False,neural_updates=0,new_physics_frames=0,
      code={str(Path(__file__).relative_to(root)):sha(__file__)},replay_sha256=sha(target)))
snapshot=out/'source-snapshot';snapshot.mkdir();shutil.copyfile(__file__,snapshot/'export_strength_replay.py')
print(json.dumps({'status':'passed','transitions':size,'counts':counts,'checks':checks,
                  'training_data_ready':True,'training_budget_authorized':False,'neural_updates':0}))
