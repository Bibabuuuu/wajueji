import unittest
import numpy as np
from manual_trajectory_residual import LIMITS_SI, compose_action, compare_arrays


class ManualResidualTests(unittest.TestCase):
    def test_restore_compares_string_identity_and_numeric_state(self):
        state = {'rigid_ids': np.array(['r1-rock-000']), 'particle_q': np.array([1.,2.])}
        exact, errors = compare_arrays(state, state)
        self.assertTrue(all(exact.values()))
        self.assertIsNone(errors['rigid_ids'])
        changed = {**state, 'rigid_ids': np.array(['r1-rock-001'])}
        self.assertFalse(compare_arrays(changed, state)[0]['rigid_ids'])
        with self.assertRaises(ValueError):
            compare_arrays({'particle_q': state['particle_q']}, state)

    def test_zero_preserves_saved_command_bytes(self):
        baseline = np.array([-.0, .012, -.02, .0675])
        actual, delta = compose_action(baseline, [0]*4, [0]*4)
        self.assertEqual(actual.tobytes(), baseline.tobytes())
        self.assertTrue(np.array_equal(delta, np.zeros(4)))

    def test_axis_units_sign_and_joint_envelope(self):
        actual, delta = compose_action([.11, 0, 0, .067], [1, -1, .5, 1], [.02, .01, .02, .01])
        np.testing.assert_array_equal(delta, [.02, -.01, .01, .01])
        np.testing.assert_array_equal(actual, [.12, -.01, .01, .0675])

    def test_rejects_unsafe_or_unconfigured_inputs(self):
        cases = [([0]*3, [0]*4, [0]*4), ([float('nan')]*4, [0]*4, [0]*4),
                 ([0]*4, [2, 0, 0, 0], [.01]*4), ([0]*4, [0]*4, [-.01]*4),
                 ([0]*4, [0]*4, LIMITS_SI*2), ([0]*4, [1, 0, 0, 0], [0]*4),
                 ([1]*4, [0]*4, [0]*4)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                compose_action(*case)


if __name__ == '__main__':
    unittest.main()
