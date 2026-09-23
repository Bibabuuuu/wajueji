from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/v4'))
import run_layered_47_sparse_motion as motion


class SparseMotionTest(unittest.TestCase):
    def test_bounded_four_axes_and_stops(self):
        phases = motion.schedule()
        self.assertEqual(len(phases), 9)
        self.assertEqual(sum(p[1] for p in phases), 270)
        np.testing.assert_allclose(sum(n*c/60 for _, n, c, _ in phases), motion.RATES*.5)
        for name, count, command, axis in phases:
            self.assertEqual(count, 30)
            self.assertLessEqual(np.count_nonzero(command), 1)
            if name == 'hold' or name.endswith('_stop'):
                np.testing.assert_array_equal(command, np.zeros(4))

    def test_actual_motion_not_just_command_required(self):
        self.assertTrue(motion.motion_response(.015, .015))
        for actual in (0., -.015, .03, float('nan')):
            self.assertFalse(motion.motion_response(actual, .015))

    def test_stop_requires_every_axis_and_bucket(self):
        row = dict(axis_velocity=[0.]*4, bucket_linear_speed_m_s=.001, bucket_angular_speed_rad_s=.001)
        self.assertTrue(motion.stopped(row))
        for bad in ({'axis_velocity': [0, .002, 0, 0]}, {'bucket_linear_speed_m_s': .03},
                    {'bucket_angular_speed_rad_s': .03}):
            self.assertFalse(motion.stopped({**row, **bad}))

    def test_no_clearance_or_motion_envelope_bypass(self):
        with patch.object(motion.probe, 'classify_frame'):
            row = dict(strict=True, bucket_vertical_clearance_m=.4, axis_offset=[0.]*4)
            self.assertTrue(motion.qualify(row))
            for bad in ({'strict': False}, {'bucket_vertical_clearance_m': .149},
                        {'axis_offset': [.031, 0, 0, 0]}, {'axis_offset': [0, .007, 0, 0]}):
                self.assertFalse(motion.qualify({**row, **bad}))

    def test_command_bridge_reuses_subcycle_and_forwards_force_once(self):
        raw = object()
        bridge = motion.CommandBridge(raw)
        bridge.command = np.array([.03, 0, 0, 0])
        with patch.object(motion, 'subcycled_step', return_value={'ok': True}) as step:
            forces = {'rock': {'force': [1., 0, 0]}}
            self.assertEqual(bridge.step(1/60, 80, forces), {'ok': True})
            self.assertIs(step.call_args.args[0], raw)
            self.assertIs(step.call_args.args[4], forces)
        bridge.command = np.array([1., 0, 0, 0])
        with self.assertRaises(ValueError):
            bridge.step(1/60, 80, {})
