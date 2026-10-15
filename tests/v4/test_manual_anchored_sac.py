import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from manual_anchored_sac import AnchoredSAC


class Toy(gym.Env):
    observation_space = gym.spaces.Box(-10., 10., (221,), np.float32)
    action_space = gym.spaces.Box(-1., 1., (5,), np.float32)
    def reset(self, seed=None, options=None):
        return np.zeros(221, np.float32), {}
    def step(self, action):
        return np.zeros(221, np.float32), 0., True, False, {}


class Tests(unittest.TestCase):
    def test_gradient_constraint_and_saved_prediction(self):
        torch.set_num_threads(1)
        obs = np.zeros((64, 221), np.float32)
        obs[32:, -1] = 1
        obs[:, 0] = np.linspace(-1, 1, 64)
        actions = np.zeros((64, 5), np.float32)
        models = []
        for cls in [SAC, AnchoredSAC]:
            model = cls('MlpPolicy', Toy(), policy_kwargs={'net_arch':[64,64]}, seed=42,
                learning_starts=0, batch_size=64, buffer_size=128, ent_coef=1e-5, gamma=1., device='cpu',
                replay_buffer_kwargs={'handle_timeout_termination':False})
            model.set_logger(configure(folder=None, format_strings=[]))
            with torch.no_grad():
                model.actor.mu.weight.zero_(); model.actor.mu.bias.zero_()
            for x in obs:
                model.replay_buffer.add(x[None], x[None], np.zeros((1,5),np.float32), np.array([1.]), np.array([True]), [{}])
            if cls is AnchoredSAC:
                model.set_anchors(obs, actions, 100.)
            model.train(100, 64)
            models.append(model)
        free_error = float(np.mean(models[0].predict(obs, deterministic=True)[0]**2))
        anchored = models[1]
        self.assertLess(anchored.anchor_error()['mse'], free_error)
        self.assertEqual(anchored.anchor_updates, 100)
        self.assertEqual(anchored._n_updates, 100)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model'
            anchored.save(path)
            loaded = SAC.load(path, device='cpu')
            self.assertTrue(np.array_equal(loaded.predict(obs, deterministic=True)[0], anchored.predict(obs, deterministic=True)[0]))
        for p in anchored.policy.parameters():
            self.assertTrue(torch.isfinite(p).all())

    def test_input_rejected(self):
        model = AnchoredSAC('MlpPolicy', Toy(), device='cpu')
        with self.assertRaisesRegex(ValueError, 'anchor_shape'):
            model.set_anchors(np.zeros((1,215),np.float32), np.zeros((1,5),np.float32))
        with self.assertRaisesRegex(ValueError, 'anchor_weight'):
            model.set_anchors(np.zeros((1,221),np.float32), np.zeros((1,5),np.float32), -1)


if __name__ == '__main__':
    unittest.main()
