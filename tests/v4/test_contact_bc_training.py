import unittest
import numpy as np
from test_contact_bc_policy import data
from contact_bc_policy import encode_episode, fit_normalization
from train_contact_bc import command_feature_indices, training_episodes, floor_terrain_scale


class TrainingTest(unittest.TestCase):
    def test_scale_floor_only_changes_height_cells(self):
        _,spec,_=encode_episode(data());scale=np.full(len(spec['feature_names']),.001,np.float32)
        result=floor_terrain_scale(scale,spec,.25)
        for i,name in enumerate(spec['feature_names']):
            self.assertAlmostEqual(float(result[i]),.25 if name.startswith('terrain_height_m[') else .001)
        np.testing.assert_array_equal(scale,np.full_like(scale,.001))

    def test_excluded_episode_cannot_affect_normalization(self):
        good=dict(meta=dict(run='good'),x=np.array([[1.],[3.]],np.float32))
        bad=dict(meta=dict(run='bad'),x=np.array([[999.]],np.float32))
        fit=training_episodes([good,bad],dict(good=1.,bad=0.))
        self.assertEqual([e['meta']['run'] for e in fit],['good'])
        mean,_,_=fit_normalization(np.concatenate([e['x'] for e in fit]),[0])
        self.assertEqual(mean[0],2.)
        for weights in (dict(good=0.,bad=0.),dict(good=1.,bad=-1.),dict(good=1.,bad=float('nan'))):
            with self.assertRaises(ValueError):training_episodes([good,bad],weights)

    def test_command_mask_preserves_measured_state_history(self):
        _,spec,_=encode_episode(data());ids=command_feature_indices(spec)
        self.assertEqual(len(set(ids)),20)
        names=spec['feature_names'];selected={names[i] for i in ids}
        self.assertIn('history_minus_9[43]',selected)
        for name in ('mechanism_qpos[0]','mechanism_qvel[0]','controller_target[0]',
                     'history_minus_3[39]','history_minus_3[44]','history_minus_3_valid[0]'):
            self.assertNotIn(name,selected)


if __name__=='__main__':unittest.main()
