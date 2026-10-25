from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from moving_bucket_sweep_runtime import MovingBucketSweepRuntime


class MovingBucketSweepRuntimeTest(unittest.TestCase):
    def test_incremental_reaction_routes_once_and_preserves_pending_snapshot(self):
        adapter = MovingBucketSweepRuntime.__new__(MovingBucketSweepRuntime)
        adapter.fixture = SimpleNamespace(state=SimpleNamespace(particle_q=object()))
        adapter.bridge = SimpleNamespace(state={
            'bucket_linear_velocity_world_m_s': [0., 0., 0.],
            'bucket_angular_velocity_world_rad_s': [0., 0., 0.],
        })
        impulses = iter([2., 3., 5., 7.])

        def apply(*args):
            return {'reaction_impulse_ns': [0., next(impulses), 0.],
                    'reaction_angular_impulse_nms': [0., 0., 0.]}

        adapter.sweep = SimpleNamespace(apply=apply)
        adapter.previous = object()
        adapter.after_mpm = object()
        adapter.pending = adapter.old_pose = adapter.current = None
        adapter.generated = np.zeros(3)
        adapter.routed = np.zeros(3)

        def native(dt, origins):
            return {'bucket': {'force': np.zeros(3), 'torque': np.zeros(3)}}, {}

        with patch('moving_bucket_sweep_runtime.wp.copy'), patch(
                'moving_bucket_sweep_runtime._bucket_transform',
                return_value=wp.transform((10., 0., 0.), wp.quat_identity())):
            first, _ = adapter.step_mpm(native, .5, {'bucket': [8., 0., 0.]})
            np.testing.assert_array_equal(first['bucket']['force'], [0., 4., 0.])
            np.testing.assert_array_equal(first['bucket']['torque'], [0., 0., 8.])
            self.assertTrue(adapter.finish_frame()['safe'])
            snapshot = adapter.diagnostic_state()
            second, _ = adapter.step_mpm(native, .5, {'bucket': [8., 0., 0.]})
            np.testing.assert_array_equal(second['bucket']['force'], [0., 16., 0.])
            np.testing.assert_array_equal(second['bucket']['torque'], [0., 0., 32.])
            final = adapter.finish_frame()
            self.assertTrue(final['safe'])
            np.testing.assert_array_equal(final['incremental_impulse_balance_error_ns'], 0.)
            self.assertEqual(snapshot['pending_velocity_reaction']['reaction_impulse_ns'], [0., 3., 0.])
            self.assertEqual(adapter.pending['reaction_impulse_ns'], [0., 7., 0.])
            with self.assertRaisesRegex(RuntimeError, 'frame_not_started'):
                adapter.finish_frame()


if __name__ == '__main__':
    unittest.main()
