"""CPU candidate preparation from one audited pool; no reset or physical step."""
import json
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from manual_anchored_sac import AnchoredSAC, load_success_anchors
from manual_residual_sac_env import ROOT, Budget, environment_contract, sha

class SpacesOnly(gym.Env):
    observation_space=gym.spaces.Box(-10.,10.,(221,),np.float32)
    action_space=gym.spaces.Box(-1.,1.,(5,),np.float32)
    def reset(self,**kwargs):raise RuntimeError('grid_must_not_reset')
    def step(self,action):raise RuntimeError('grid_must_not_step')

budget=Budget(ROOT/'runs/manual-strength-overnight-20261013-budget.json',seconds=28800)
out=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1'
out.mkdir(exist_ok=False)
review=ROOT/'runs/manual-strength-recovery-collection-20261013-v1/review.json'
obs,actions,counts=load_success_anchors(review)
transfer=json.loads((review.parent/'training-ready/transfer.json').read_text())
torch.set_num_threads(1);started=time.monotonic();trials=[]
report=dict(status='running',anchors=counts,trials=trials,environment_contract=environment_contract('error_strength_v1'),
    source_code={n:sha(ROOT/'src/v4'/n) for n in ['manual_anchored_sac.py','run_manual_residual_sac.py','manual_residual_sac_env.py']},
    dataset_receipt=dict(path=str(review),sha256=sha(review)),shared_budget=str(budget.path),new_physics_frames=0)
for seed,weight in [(42,10.),(42,30.),(42,50.),(42,300.),(43,50.),(43,100.),(43,300.),(44,50.),(44,100.)]:
    budget.check()
    if budget.remaining()<3600:break
    model=AnchoredSAC('MlpPolicy',SpacesOnly(),learning_rate=3e-4,buffer_size=30000,batch_size=64,
        gamma=1.,learning_starts=0,train_freq=1,gradient_steps=1,ent_coef=1e-5,
        replay_buffer_kwargs={'handle_timeout_termination':False},policy_kwargs={'net_arch':[64,64]},device='cpu',seed=seed)
    with torch.no_grad():
        model.actor.mu.weight.zero_();model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-4.6)
    model.load_replay_buffer(transfer['replay_path'])
    model.set_logger(configure(folder=None,format_strings=[]));model.set_anchors(obs,actions,weight)
    for i in range(1000):budget.check();model.train(1,64)
    name=f'seed{seed}-weight{int(weight)}'
    path=out/(name+'.zip');model.save(path)
    trials.append(dict(name=name,seed=seed,weight=weight,checkpoint=str(path),sha256=sha(path),anchor_error=model.anchor_error(),
                       cpu_preflight=bool(model.anchor_error()['maximum_abs']<.03 and all(torch.isfinite(p).all() for p in model.policy.parameters()))))
    report.update(wall_seconds=time.monotonic()-started,SAC_updates=len(trials)*1000)
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CPU_GRID',name,trials[-1]['anchor_error'],flush=True)
report['status']='passed';(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
