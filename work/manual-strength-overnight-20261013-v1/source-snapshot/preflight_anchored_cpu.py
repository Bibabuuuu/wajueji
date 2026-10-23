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
    def reset(self, **kwargs):
        raise RuntimeError('CPU_preflight_must_not_reset')
    def step(self, action):
        raise RuntimeError('CPU_preflight_must_not_step')

budget=Budget(ROOT/'runs/manual-strength-overnight-20261013-budget.json',seconds=28800)
budget.check()
out=ROOT/'work/manual-strength-overnight-20261013-v1/cpu-anchors-v2'
out.mkdir(exist_ok=False)
review=ROOT/'runs/manual-strength-recovery-collection-20261013-v1/review.json'
obs, actions, counts=load_success_anchors(review)
transfer=json.loads((review.parent/'training-ready/transfer.json').read_text())
torch.set_num_threads(1)
started=time.monotonic(); trials=[]
for weight in [100.,1000.]:
    model=AnchoredSAC('MlpPolicy',SpacesOnly(),learning_rate=3e-4,buffer_size=30000,batch_size=64,
        gamma=1.,learning_starts=0,train_freq=1,gradient_steps=1,ent_coef=1e-5,
        replay_buffer_kwargs={'handle_timeout_termination':False},policy_kwargs={'net_arch':[64,64]},device='cpu',seed=42)
    with torch.no_grad():
        model.actor.mu.weight.zero_(); model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_(); model.actor.log_std.bias.fill_(-4.6)
    model.load_replay_buffer(transfer['replay_path'])
    model.set_logger(configure(folder=None,format_strings=[]))
    model.set_anchors(obs,actions,weight)
    curve=[]
    for i in range(1000):
        budget.check();model.train(1,64)
        if i+1 in [100,200,500,1000]:curve.append(model.anchor_error())
    model.save(out/f'weight-{int(weight)}')
    trials.append(dict(weight=weight,curve=curve,checkpoint=str(out/f'weight-{int(weight)}.zip')))
report=dict(status='passed',anchors=counts,trials=trials,wall_seconds=time.monotonic()-started,
    new_physics_frames=0,SAC_updates=2000,environment_contract=environment_contract('error_strength_v1'),
    source_code={n:sha(ROOT/'src/v4'/n) for n in ['manual_anchored_sac.py','run_manual_residual_sac.py','manual_residual_sac_env.py']})
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
