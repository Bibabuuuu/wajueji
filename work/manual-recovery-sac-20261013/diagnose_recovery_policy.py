"""Frozen-model inference on previously audited states, without physics or updates."""
from pathlib import Path
import json
import pickle
import numpy as np
import torch
from stable_baselines3 import SAC

ROOT=Path('/home/qunwang/wajueji')
RUN=ROOT/'runs/manual-recovery-sac-20261013-v1'
torch.set_num_threads(1)
model=SAC.load(RUN/'batch-v1/sac-last.zip',device='cpu')
report=dict(new_physics_frames=0,gradient_updates=0,checkpoint_updates=model._n_updates,
            scope='counterfactual requests on recorded recovery states; not a physical success test',cases=[])
for label in ['early43','early52','loaded970']:
    with np.load(ROOT/'runs/manual-recovery-collection-20261013-v1'/label/'transitions.npz',allow_pickle=False) as data:
        mask=(data['observations'][:,-1]>.5)&(data['policy_source']=='fixed_correction')
        obs=data['observations'][mask];teacher=data['actions'][mask]
    if not len(obs):continue
    actions,_=model.predict(obs,deterministic=True)
    errors=obs[:,-6:-2]
    helpful=(actions[:,:4]*errors>1e-8)
    opposing=(actions[:,:4]*errors<-1e-8)
    with torch.no_grad():
        state=torch.as_tensor(obs,device=model.device)
        policy_q=torch.cat(model.critic(state,torch.as_tensor(actions,device=model.device)),dim=1).min(dim=1).values.cpu().numpy()
        teacher_q=torch.cat(model.critic(state,torch.as_tensor(teacher,device=model.device)),dim=1).min(dim=1).values.cpu().numpy()
    report['cases'].append(dict(label=label,waiting_teacher_states=len(obs),
            any_directed_axis_states=int(helpful.any(axis=1).sum()),
            no_directed_axis_states=int((~helpful.any(axis=1)).sum()),
            per_axis_helpful_counts=helpful.sum(axis=0).tolist(),
            per_axis_opposing_counts=opposing.sum(axis=0).tolist(),
            per_axis_mean_action=actions.mean(axis=0).tolist(),
            saturated_fraction=float((abs(actions)>=.95).mean()),
            median_critic_policy_minus_teacher=float(np.median(policy_q-teacher_q)),
            critic_prefers_policy_fraction=float((policy_q>teacher_q).mean())))
(RUN/'frozen-policy-diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
