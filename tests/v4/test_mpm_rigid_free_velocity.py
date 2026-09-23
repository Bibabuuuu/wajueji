import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/v4"))
from mpm_rigid_free_velocity import predict_free_linear_velocity


class PredictorTests(unittest.TestCase):
    def test_static_contact_balance_and_missing_gravity_regression(self):
        # Ideal rigid support removes the free normal velocity by impulse.
        # Omitting gravity in that solve leaves g*dt after rigid integration.
        for mass in (10.89, 53.51, 63.52):
            for dt in (1 / 120, 1 / 240):
                g = np.array([0., 0., -9.81])
                initial = np.zeros(3)
                predicted = predict_free_linear_velocity(initial, g, dt)
                impulse = -mass * predicted
                np.testing.assert_allclose(initial + dt*g + impulse/mass, 0., atol=1e-15)
                np.testing.assert_array_equal(initial, np.zeros(3))
                old_impulse = -mass * initial
                self.assertAlmostEqual((initial + dt*g + old_impulse/mass)[2], -9.81*dt)

    def test_no_contact_does_not_apply_prediction_twice(self):
        initial = np.array([0.1, 0.2, -0.3])
        original = initial.copy()
        predicted = predict_free_linear_velocity(initial, [0, 0, -9.81], 1/120)
        np.testing.assert_array_equal(initial, original)
        np.testing.assert_allclose(predicted, original + [0, 0, -9.81/120])
        with self.assertRaises(ValueError):
            predict_free_linear_velocity(initial, [0, 0, -9.81], 0)


if __name__ == "__main__":
    unittest.main()
