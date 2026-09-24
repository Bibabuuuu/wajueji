import unittest
import numpy as np
from load_tracking import LoadAwareReference


class LoadedTrackingTests(unittest.TestCase):
    def test_loaded_equilibrium_does_not_command_chasing(self):
        actual = np.array([0., .348, .378, .8516])
        target = actual + np.array([0., .0014, -.0064, .0004])
        tracker = LoadAwareReference(actual, target)
        for _ in range(180):
            np.testing.assert_allclose(tracker.command(actual, target, actual,
                                      np.zeros(4), loaded=True), 0, atol=1e-15)

    def test_actual_error_changes_correction_direction(self):
        desired = np.array([0., .35, .38, .8516])
        tracker = LoadAwareReference(desired, desired)
        actual = desired + np.array([0., 0., .003, 0.])
        command = tracker.command(actual, desired, desired, np.zeros(4), loaded=True)
        self.assertLess(command[2], 0.)

    def test_loaded_bucket_cannot_open_but_dump_can(self):
        q = np.array([0., .35, .38, .8516]); desired = q.copy(); desired[3] -= .1
        tracker = LoadAwareReference(q, q)
        self.assertEqual(tracker.command(q, q, desired, np.zeros(4), loaded=True)[3], 0.)
        self.assertLess(tracker.command(q, q, desired, np.zeros(4), loaded=False)[3], 0.)

    def test_bias_is_bounded_and_saturation_does_not_wind_up(self):
        q = np.zeros(4); tracker = LoadAwareReference(q, q)
        for _ in range(600):
            cmd = tracker.command(q, q, np.ones(4), np.zeros(4), loaded=False)
            self.assertTrue(np.all(abs(cmd) <= tracker.limits))
        np.testing.assert_array_equal(tracker.bias, np.zeros(4))

    def test_invalid_initial_compensation_rejected(self):
        with self.assertRaises(ValueError):
            LoadAwareReference(np.zeros(4), np.ones(4))


if __name__ == '__main__':
    unittest.main()
