import unittest
import numpy as np
from run_layered_47_excavation_preflight import ExcavationFixture


class WrenchReductionTest(unittest.TestCase):
    def test_compaction_keeps_exact_force_torque_counts_and_bad_positions(self):
        rng = np.random.default_rng(729)
        impulse = rng.normal(size=(12000, 3)).astype(np.float32)
        position = rng.normal(size=(12000, 3)).astype(np.float32)
        ids = rng.integers(-1, 49, size=len(impulse))
        impulse[::2] = 0
        impulse[3] = np.nan
        impulse[7] = np.inf
        impulse[9] = 1e-14
        position[11] = np.nan
        active = np.isfinite(impulse).all(axis=1) & (np.linalg.norm(impulse, axis=1) > 1e-12)
        for body in range(1, 50):
            selected = np.array([body])
            origin = rng.normal(size=3)
            expected, count = ExcavationFixture._collect(
                impulse, position, ids, selected, origin, 1/60)
            actual, actual_count = ExcavationFixture._collect(
                impulse[active], position[active], ids[active], selected,
                origin, 1/60, prefiltered=True)
            self.assertEqual(actual_count, count)
            for field in expected:
                np.testing.assert_array_equal(actual[field], expected[field])


if __name__ == "__main__":
    unittest.main()
