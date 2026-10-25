from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

from soil_hencky_update_guard import summarize_material_state  # noqa: E402


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value.copy()


class State:
    def __init__(self):
        self.particle_elastic_strain = Array(np.repeat(
            np.eye(3, dtype=np.float32)[None], 3, axis=0))
        self.particle_Jp = Array(np.ones(3, dtype=np.float32))
        self.particle_stress = Array(np.zeros((3, 3, 3), dtype=np.float32))


class SoilHenckyUpdateGuardTest(unittest.TestCase):
    def test_safe_material_state(self):
        result = summarize_material_state(
            State(), np.zeros(3, dtype=np.int32))
        self.assertTrue(result["safe"])
        self.assertEqual(result["rejection_count"], 0)
        self.assertAlmostEqual(result["maximum_F_frobenius"], np.sqrt(3))

    def test_rejection_and_nonfinite_state_fail_closed(self):
        state = State()
        state.particle_Jp.value[1] = np.nan
        result = summarize_material_state(
            state, np.asarray([0, 2, 0], dtype=np.int32))
        self.assertFalse(result["safe"])
        self.assertEqual(result["rejection_count"], 1)
        self.assertEqual(result["rejection_codes"]["2"], 1)
        self.assertFalse(result["checks"]["all_material_arrays_finite"])

    def test_nonpositive_Jp_and_extreme_F_fail_closed(self):
        state = State()
        state.particle_Jp.value[0] = 0.0
        state.particle_elastic_strain.value[2, 0, 0] = 1001.0
        result = summarize_material_state(
            state, np.zeros(3, dtype=np.int32))
        self.assertFalse(result["checks"]["Jp_positive"])
        self.assertFalse(result["checks"]["elastic_F_frobenius_bounded"])


if __name__ == "__main__":
    unittest.main()
