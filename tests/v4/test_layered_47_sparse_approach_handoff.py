from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_sparse_approach_handoff as approach  # noqa: E402


class SparseApproachHandoffTest(unittest.TestCase):
    def row(self):
        return {
            "strict": True,
            "joint_bounds_ok": True,
            "bucket_terrain_clearance_m": 0.2,
            "axis_tracking_error": 0.001,
            "closure_m": 0.0001,
        }

    def test_frame_guard_keeps_physical_limits(self):
        self.assertTrue(approach.frame_passes(self.row()))
        bad_values = {
            "strict": False,
            "joint_bounds_ok": False,
            "bucket_terrain_clearance_m": 0.149,
            "axis_tracking_error": 0.026,
            "closure_m": 0.0011,
        }
        for name, value in bad_values.items():
            with self.subTest(name=name):
                self.assertFalse(approach.frame_passes({
                    **self.row(), name: value}))

    def test_sparse_route_is_frozen_to_60_and_4800(self):
        self.assertEqual(approach.HZ, 60)
        self.assertEqual(approach.RIGID_INTERNAL_HZ, 4800)
        self.assertEqual(approach.MAXIMUM_PLANNED_CLOSURE_M, 1e-4)
        self.assertEqual(approach.MAXIMUM_PHYSICAL_CLOSURE_M, 1e-3)
        self.assertEqual(
            approach.SPARSE_TARGET_TIP_HEIGHT_ABOVE_SURFACE_M, 0.60)
        self.assertEqual(
            approach.SUPPORTED_TARGET_TIP_HEIGHTS_ABOVE_SURFACE_M,
            (0.55, 0.60))
        self.assertEqual(approach.ENDPOINT_HOLD_FRAMES, 30)

    def test_rear_lower_route_adds_mechanism_and_truck_guards(self):
        row = {
            **self.row(),
            "bucket_terrain_clearance_m": 0.33,
            "mechanism_terrain_clearance_m": 0.03,
            "static_truck_aabb_separation_m": 0.03,
        }
        self.assertTrue(approach.frame_passes(
            row, approach.REAR_LOWER_ROUTE))
        for name, value in {
                "bucket_terrain_clearance_m": 0.319,
                "mechanism_terrain_clearance_m": 0.019,
                "static_truck_aabb_separation_m": 0.019,
                }.items():
            with self.subTest(name=name):
                self.assertFalse(approach.frame_passes(
                    {**row, name: value}, approach.REAR_LOWER_ROUTE))

    def test_unknown_route_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "sparse_approach_route"):
            approach.frame_passes(self.row(), "unknown")

    def test_rear_lower_distinguishes_grid_residual_from_contact(self):
        residual = {
            **self.row(),
            "strict": False,
            "stable": True,
            "guard_failures": ["bucket_mpm_contact_samples"],
            "bucket_mpm_effective_contact": False,
            "bucket_terrain_clearance_m": 0.714,
            "mechanism_terrain_clearance_m": 0.297,
            "static_truck_aabb_separation_m": 0.171,
        }
        self.assertTrue(approach.frame_passes(
            residual, approach.REAR_LOWER_ROUTE))
        self.assertFalse(approach.frame_passes({
            **residual,
            "bucket_mpm_effective_contact": True,
        }, approach.REAR_LOWER_ROUTE))
        self.assertFalse(approach.frame_passes({
            **residual,
            "guard_failures": [
                "bucket_mpm_contact_samples", "closure_m"],
        }, approach.REAR_LOWER_ROUTE))

    def test_contact_diagnostics_rejects_far_tiny_grid_impulse(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def numpy(self):
                return self.value

        class Solver:
            @staticmethod
            def collect_collider_impulses(_state):
                return (
                    Array([[0.0, 0.0, 5.0e-6]]),
                    Array([[5.95, 0.25, 1.65]]),
                    Array([7]),
                )

        class State:
            particle_q = Array([[5.66, 0.25, 1.38]])

        class Fixture:
            solver = Solver()
            state = State()
            bucket_collider_ids = np.asarray([7])

        result = approach._bucket_contact_diagnostics(Fixture())
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["classification"],
                         "raw_grid_impulse_diagnostic_only")
        self.assertFalse(result["effective_contact"])
        self.assertFalse(
            result["particle_center_distance_promotes_contact"])

    def test_near_grid_node_does_not_promote_tiny_impulse_to_contact(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def numpy(self):
                return self.value

        class Solver:
            @staticmethod
            def collect_collider_impulses(_state):
                return (
                    Array([[0.0, 0.0, 3.4e-6]]),
                    Array([[5.95, 0.25, 1.65]]),
                    Array([7]),
                )

        class State:
            particle_q = Array([[5.95, 0.25, 1.65]])

        class Fixture:
            solver = Solver()
            state = State()
            bucket_collider_ids = np.asarray([7])

        result = approach._bucket_contact_diagnostics(Fixture())
        self.assertFalse(result["effective_contact"])
        self.assertEqual(result["classification"],
                         "raw_grid_impulse_diagnostic_only")

    def test_larger_raw_impulse_still_does_not_guess_physical_contact(self):
        class Array:
            def __init__(self, value):
                self.value = np.asarray(value)

            def numpy(self):
                return self.value

        class Solver:
            @staticmethod
            def collect_collider_impulses(_state):
                return (
                    Array([[0.0, 0.0, 1.1e-4]]),
                    Array([[5.95, 0.25, 1.65]]),
                    Array([7]),
                )

        class State:
            particle_q = Array([[0.0, 0.0, 0.0]])

        class Fixture:
            solver = Solver()
            state = State()
            bucket_collider_ids = np.asarray([7])

        result = approach._bucket_contact_diagnostics(Fixture())
        self.assertFalse(result["effective_contact"])
        self.assertFalse(result["impulse_promotes_contact"])
        self.assertEqual(result["classification"],
                         "raw_grid_impulse_diagnostic_only")


if __name__ == "__main__":
    unittest.main()
