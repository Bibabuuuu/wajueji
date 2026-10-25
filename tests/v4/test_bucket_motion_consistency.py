from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

from bucket_motion_consistency import POLICY, restored_controller, predict_pose, ConsistentBucketSweepRuntime
from tests.v4.test_moving_bucket_sweep_candidate import MovingSweepTest


class BucketMotionConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        MovingSweepTest.setUpClass.__func__(cls)

    correct = MovingSweepTest.correct

    def test_saved_com_velocity_is_converted_once_without_mutating_source(self):
        old = {'bucket_linear_velocity_world_m_s': [1, 2, 3],
               'bucket_angular_velocity_world_rad_s': [0, 0, 2],
               'controller_target': [4, 5, 6, 7]}
        state = {'bucket_inertial_offset_world_m': [.25, 0, -.5]}
        new = restored_controller(old, state, None)
        self.assertEqual(new['bucket_linear_velocity_world_m_s'], [1, 1.5, 3])
        self.assertEqual(old['bucket_linear_velocity_world_m_s'], [1, 2, 3])
        self.assertEqual(restored_controller(new, {}, POLICY), new)
        self.assertEqual(new['controller_target'], old['controller_target'])

    def test_corrected_checkpoint_requires_opt_in_and_matching_sources(self):
        from pathlib import Path
        import layered_47_cycle_checkpoint as cp
        names = ('bucket_velocity_origin_candidate.py', 'bucket_motion_consistency.py')
        bindings = {name: cp.checkpoint_identity(Path(cp.__file__).with_name(name))['sha256']
                    for name in names}
        metadata = {'bucket_motion_policy': POLICY}
        with self.assertRaisesRegex(ValueError, 'explicit'):
            cp.validate_bucket_motion_policy(metadata, bindings)
        cp.validate_bucket_motion_policy(metadata, bindings, allowed=True)
        with self.assertRaisesRegex(ValueError, 'version_mismatch'):
            cp.validate_bucket_motion_policy(metadata, {}, allowed=True)
        cp.validate_bucket_motion_policy({}, {})

    def test_comoving_particle_is_not_squeezed_by_previous_wall(self):
        old = wp.transform_identity()
        new = predict_pose(old, [0, 0, 6], [0, 0, 0], 1 / 60)
        q, v, result = self.correct((0, 0, -.05), (0, 0, .05), (0, 0, 6),
                                    old=old, new=new, linear=(0, 0, 6))
        self.assertEqual(result['corrected_particles'], 0)
        np.testing.assert_allclose(q, [0, 0, .05], atol=1e-7)
        # Old time contract falsely treats this as crossing the fixed wall.
        _, _, old_result = self.correct((0, 0, -.05), (0, 0, .05), (0, 0, 6))
        self.assertEqual(old_result['corrected_particles'], 1)

    def test_world_spin_composes_on_left(self):
        old = wp.transform((1, 2, 3), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), .7))
        new = predict_pose(old, [1, 0, 0], [0, 0, 2], .25)
        expected = wp.mul(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), .5),
                          wp.transform_get_rotation(old))
        np.testing.assert_allclose(wp.transform_get_translation(new), [1.25, 2, 3])
        np.testing.assert_allclose(wp.transform_get_rotation(new), expected)

    def test_both_sweep_phases_share_predicted_endpoint_and_balance(self):
        a = ConsistentBucketSweepRuntime.__new__(ConsistentBucketSweepRuntime)
        a.fixture = SimpleNamespace(state=SimpleNamespace(particle_q=object()))
        a.bridge = SimpleNamespace(state={'bucket_linear_velocity_world_m_s': [2, 0, 0],
                                         'bucket_angular_velocity_world_rad_s': [0, 0, 0]})
        a.previous = object(); a.after_mpm = object()
        a.pending = a.old_pose = a.current = None
        a.generated = np.zeros(3); a.routed = np.zeros(3)
        endpoints = []
        def apply(state, prev, old, new, *args):
            endpoints.append((list(old), list(new)))
            return {'reaction_impulse_ns': [0, 2, 0], 'reaction_angular_impulse_nms': [0, 0, 0]}
        a.sweep = SimpleNamespace(apply=apply)
        def native(dt, origins):
            return {'bucket': {'force': np.zeros(3), 'torque': np.zeros(3)}}, {}
        actual = wp.transform((1.1, 0, 0), wp.quat_identity())
        with patch('bucket_motion_consistency.wp.copy'), patch(
                'bucket_motion_consistency._bucket_transform', return_value=wp.transform_identity()), patch(
                'moving_bucket_sweep_runtime._bucket_transform', return_value=actual):
            a.step_mpm(native, .5, {'bucket': [0, 0, 0]})
            result = a.finish_frame()
        np.testing.assert_array_equal(endpoints[0][1], endpoints[1][0])
        self.assertEqual(endpoints[0][1][0], 1.)
        np.testing.assert_array_equal(endpoints[1][1], list(actual))
        self.assertTrue(result['safe'])
        np.testing.assert_array_equal(a.generated - a.routed, [0, 2, 0])


if __name__ == '__main__':
    unittest.main()
