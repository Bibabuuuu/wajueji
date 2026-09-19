import unittest

from diagnose_entry_drift_and_reverse import (
    entry_gate_summary,
    first_forward_crossing,
    quaternion_rotate_wxyz,
)


def row(step, x_value, z_value, action=(0.0, 0.0, 0.0, 0.0)):
    point = {
        "world_m": (x_value, 0.0, z_value),
        "relative_to_target_m": (x_value, 0.0, z_value),
    }
    return {
        "decision_step": step,
        "sim_time_s": step * 0.05,
        "action": tuple(action),
        "references": {
            "bucket_origin": point,
            "cutting_tip": point,
            "bucket_mouth": point,
        },
        "cylinder_position_normalized": (0.0, 0.0, 0.0),
        "axis_velocity_normalized": (0.0, 0.0, 0.0, 0.0),
    }


class GeometryTests(unittest.TestCase):
    def test_quaternion_rotation_uses_wxyz_and_normalizes(self):
        identity = quaternion_rotate_wxyz((2.0, 0.0, 0.0, 0.0),
                                          (1.0, 2.0, 3.0))
        self.assertEqual(identity, (1.0, 2.0, 3.0))
        half = 2 ** -0.5
        rotated = quaternion_rotate_wxyz((half, 0.0, half, 0.0),
                                         (1.0, 0.0, 0.0))
        self.assertAlmostEqual(rotated[0], 0.0)
        self.assertAlmostEqual(rotated[2], -1.0)

    def test_crossing_interpolates_same_reference_point(self):
        rows = [row(0, -0.5, 1.2), row(1, -0.3, 0.8)]
        crossing = first_forward_crossing(rows, "cutting_tip", -0.4)
        self.assertAlmostEqual(crossing["relative_z_m"], 1.0)
        self.assertAlmostEqual(crossing["decision_step_interpolated"], 0.5)
        self.assertEqual(crossing["bracket_steps"], [0, 1])


class EntryGateTests(unittest.TestCase):
    def test_vertical_window_exits_before_horizontal_gate(self):
        rows = [
            row(0, -0.8, 1.10, (0.0, -1.0, -1.0, 0.0)),
            row(1, -0.7, 1.00, (0.0, -1.0, -1.0, 0.0)),
            row(2, -0.6, 0.80, (0.0, -1.0, -1.0, 0.0)),
            row(3, -0.4, 0.60, (0.0, -1.0, -1.0, 0.0)),
        ]
        result = entry_gate_summary(rows)
        self.assertEqual(result["first_vertical_window"]["decision_step"], 1)
        self.assertEqual(
            result["first_below_after_entering_vertical_window"]["decision_step"], 2)
        self.assertTrue(result["vertical_window_exited_before_horizontal_ready"])
        self.assertIsNone(result["first_joint_ready"])

    def test_joint_gate_is_observed_when_x_and_z_overlap(self):
        rows = [row(0, -0.4, 1.0), row(1, -0.29, 0.95)]
        result = entry_gate_summary(rows)
        self.assertEqual(result["first_joint_ready"]["decision_step"], 1)
        self.assertFalse(result["vertical_window_exited_before_horizontal_ready"])


if __name__ == "__main__":
    unittest.main()
