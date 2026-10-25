"""SB3 SAC with an explicit successful-demonstration actor regularizer."""
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import require_environment_contract, sha


def load_success_anchors(review_path):
    review = json.loads(Path(review_path).read_text())
    require_environment_contract(review, 'error_strength_v1')
    if review['status'] != 'completed':
        raise ValueError('incomplete_anchor_collection')
    observations, actions, counts = [], [], {}
    for entry in review['episodes']:
        if not entry['result']['success']:
            continue
        dataset = entry['dataset']
        if sha(dataset['path']) != dataset['sha256']:
            raise ValueError('anchor_source_changed')
        with np.load(dataset['path'], allow_pickle=False) as data:
            # Intentionally induced stalls and stopping demonstrations are not targets.
            selected = data['policy_source'] == 'manual_follow'
            if entry['label'] == 'loaded_half':
                selected |= data['policy_source'] == 'fixed_wait_strength'
            observations.append(data['observations'][selected])
            actions.append(data['actions'][selected])
            counts[entry['label']] = int(selected.sum())
    obs, act = np.concatenate(observations), np.concatenate(actions)
    if obs.shape[1:] != (221,) or act.shape[1:] != (5,) or not np.isfinite(obs).all() or not np.isfinite(act).all():
        raise ValueError('anchor_shape_or_finite')
    if np.any(act):
        raise ValueError('neutral_follow_and_half_strength_targets_expected')
    _, unique = np.unique(obs, axis=0, return_index=True)
    unique.sort()
    return obs[unique].astype(np.float32), act[unique].astype(np.float32), dict(selected=counts, unique=len(unique), removed_duplicates=len(obs)-len(unique))


class AnchoredSAC(SAC):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['anchor_observations', 'anchor_actions', 'anchor_hook', 'anchor_wait_indices', 'anchor_follow_indices']

    def set_anchors(self, observations, actions, weight=100.):
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('anchor_weight')
        if observations.shape[1:] != (221,) or not len(observations) or actions.shape != (len(observations), 5):
            raise ValueError('anchor_shape')
        if not np.isfinite(observations).all() or not np.isfinite(actions).all() or np.any(abs(actions)>1):
            raise ValueError('anchor_values')
        if hasattr(self, 'anchor_hook'):
            self.anchor_hook.remove()
        self.anchor_observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        self.anchor_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.anchor_wait_indices = torch.where(self.anchor_observations[:, -1] == 1)[0]
        self.anchor_follow_indices = torch.where(self.anchor_observations[:, -1] == 0)[0]
        self.anchor_weight = float(weight)
        self.anchor_updates = getattr(self, 'anchor_updates', 0)
        self.anchor_hook = self.actor.optimizer.register_step_pre_hook(self._anchor_gradient)

    def _anchor_gradient(self, optimizer, args, kwargs):
        indices = []
        for candidates in [self.anchor_follow_indices, self.anchor_wait_indices]:
            if len(candidates):
                indices.append(candidates[torch.randint(len(candidates), (32,), device=self.device)])
        selected = torch.cat(indices)
        prediction = self.actor(self.anchor_observations[selected], deterministic=True)
        loss = self.anchor_weight*torch.mean((prediction-self.anchor_actions[selected])**2)
        # SB3 already backpropagated its actor loss. Add this gradient before its one optimizer step.
        loss.backward()
        self.anchor_updates += 1

    def anchor_error(self):
        with torch.no_grad():
            residual = self.actor(self.anchor_observations, deterministic=True)-self.anchor_actions
        return dict(mse=float((residual**2).mean().item()), maximum_abs=float(residual.abs().max().item()),
                    updates=self.anchor_updates, weight=self.anchor_weight)
