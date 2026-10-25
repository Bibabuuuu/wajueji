from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_sparse_approach_handoff_audit as audit  # noqa: E402


class SparseApproachHandoffAuditTest(unittest.TestCase):
    def row(self):
        return {
            "strict": True,
            "joint_bounds_ok": True,
            "bucket_terrain_clearance_m": 0.38,
            "axis_tracking_error": 0.001,
            "closure_m": 0.0001,
        }

    def test_audit_guard_preserves_sparse_headroom(self):
        self.assertTrue(audit.audit_frame_passes(self.row()))
        for name, value in {
                "strict": False,
                "joint_bounds_ok": False,
                "bucket_terrain_clearance_m": 0.349,
                "axis_tracking_error": 0.016,
                "closure_m": 0.0011,
                }.items():
            with self.subTest(name=name):
                self.assertFalse(audit.audit_frame_passes({
                    **self.row(), name: value}))

    def test_audit_is_one_second_zero_hold(self):
        self.assertEqual(audit.HZ, 60)
        self.assertEqual(audit.RIGID_INTERNAL_HZ, 4800)
        self.assertEqual(audit.AUDIT_FRAMES, 60)
        self.assertEqual(audit.MINIMUM_RESTORE_CLEARANCE_M, 0.35)
        self.assertEqual(
            audit.LOWERED_PREPOINT_MINIMUM_RESTORE_CLEARANCE_M, 0.32)

    def test_lowered_prepoint_uses_predeclared_clearance_threshold(self):
        self.assertEqual(audit.minimum_restore_clearance(0.60), 0.35)
        self.assertEqual(audit.minimum_restore_clearance(0.55), 0.32)
        row = {**self.row(), "bucket_terrain_clearance_m": 0.33}
        self.assertFalse(audit.audit_frame_passes(row))
        self.assertTrue(audit.audit_frame_passes(row, 0.32))
        with self.assertRaisesRegex(ValueError, "unsupported_target_height"):
            audit.minimum_restore_clearance(0.45)

    def test_rear_lower_restore_guard_checks_all_clearances(self):
        row = {
            **self.row(),
            "bucket_terrain_clearance_m": 0.33,
            "mechanism_terrain_clearance_m": 0.03,
            "static_truck_aabb_separation_m": 0.03,
        }
        self.assertTrue(audit.audit_frame_passes(
            row, 0.32, audit.approach.REAR_LOWER_ROUTE))
        for name, value in {
                "bucket_terrain_clearance_m": 0.319,
                "mechanism_terrain_clearance_m": 0.019,
                "static_truck_aabb_separation_m": 0.019,
                }.items():
            with self.subTest(name=name):
                self.assertFalse(audit.audit_frame_passes(
                    {**row, name: value}, 0.32,
                    audit.approach.REAR_LOWER_ROUTE))

    def test_rear_lower_restore_accepts_only_classified_grid_residual(self):
        row = {
            **self.row(),
            "strict": False,
            "stable": True,
            "guard_failures": ["bucket_mpm_contact_samples"],
            "bucket_mpm_effective_contact": False,
            "bucket_terrain_clearance_m": 0.714,
            "mechanism_terrain_clearance_m": 0.297,
            "static_truck_aabb_separation_m": 0.171,
        }
        self.assertTrue(audit.audit_frame_passes(
            row, 0.32, audit.approach.REAR_LOWER_ROUTE))
        self.assertFalse(audit.audit_frame_passes({
            **row, "bucket_mpm_effective_contact": True,
        }, 0.32, audit.approach.REAR_LOWER_ROUTE))


if __name__ == "__main__":
    unittest.main()
