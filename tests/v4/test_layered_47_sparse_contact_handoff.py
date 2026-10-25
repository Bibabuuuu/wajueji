from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_sparse_contact_handoff as contact  # noqa: E402


class SparseContactHandoffTest(unittest.TestCase):
    def row(self):
        return {
            "finite": True,
            "stable": True,
            "strict": True,
            "forbidden_contact_count": 0,
            "external_rigid_support_substeps": 0,
            "unmapped_active_collider_sample_count": 0,
            "nonfinite_collider_sample_count": 0,
            "bucket_mpm_contact_samples": 0,
            "rock_bucket_contact_substeps": 0,
            "closure_m": 0.0001,
            "maximum_rock_linear_speed_m_s": 0.01,
            "maximum_rock_angular_speed_rad_s": 0.1,
            "particle_max_speed_m_s": 0.1,
            "particle_p99_speed_m_s": 0.01,
            "rock_displacement_from_source_m": 0.001,
            "joint_bounds_ok": True,
            "bucket_terrain_clearance_m": 0.2,
            "axis_tracking_error": 0.001,
            "bucket_reference_vertex_error_m": 0.0001,
            "bucket_mpm_wrench_finite": True,
            "bucket_mpm_force_norm_n": 0.0,
            "command_si": [0.0] * 4,
        }

    def test_protocol_is_quarter_speed_first_contact_only(self):
        np.testing.assert_allclose(
            contact.CONTACT_COMMAND_LIMITS_M_S,
            contact.COMMAND_LIMITS_M_S * 0.25,
            rtol=0.0, atol=0.0)
        self.assertEqual(contact.HZ, 60)
        self.assertEqual(contact.RIGID_INTERNAL_HZ, 4800)
        self.assertEqual(contact.POST_CONTACT_HOLD_FRAMES, 30)
        self.assertEqual(contact.STOPPED_TAIL_FRAMES, 15)
        self.assertEqual(contact.TARGET_TIP_HEIGHT_ABOVE_SURFACE_M, 0.40)

    def test_expected_soil_contact_is_only_removed_hard_guard(self):
        row = {**self.row(),
               "strict": False,
               "bucket_mpm_contact_samples": 3,
               "bucket_mpm_force_norm_n": 12.0}
        self.assertTrue(contact.first_contact_frame_passes(row))
        for name, value in {
                "rock_bucket_contact_substeps": 1,
                "forbidden_contact_count": 1,
                "particle_p99_speed_m_s": 0.051,
                "closure_m": 0.0011,
                "bucket_mpm_wrench_finite": False,
                "bucket_terrain_clearance_m": 0.149,
                "bucket_reference_vertex_error_m": 0.0011,
                }.items():
            with self.subTest(name=name):
                self.assertFalse(contact.first_contact_frame_passes({
                    **row, name: value}))

    def test_precontact_still_requires_original_strict_gate(self):
        self.assertTrue(contact.precontact_frame_passes(self.row()))
        self.assertFalse(contact.precontact_frame_passes({
            **self.row(), "strict": False}))
        self.assertFalse(contact.precontact_frame_passes({
            **self.row(), "bucket_mpm_contact_samples": 1}))

    def test_contact_requires_nonzero_finite_force(self):
        row = {**self.row(),
               "strict": False,
               "bucket_mpm_contact_samples": 1}
        self.assertFalse(contact.first_contact_frame_passes(row))
        self.assertTrue(contact.first_contact_frame_passes({
            **row, "bucket_mpm_force_norm_n": 1.0e-6}))

    def test_wrench_summary_rejects_nonfinite_values(self):
        summary = contact.wrench_summary({
            "force": [1.0, 2.0, 2.0],
            "torque": [0.0, 3.0, 4.0],
            "point": [1.0, 0.0, 0.0],
        })
        self.assertTrue(summary["bucket_mpm_wrench_finite"])
        self.assertEqual(summary["bucket_mpm_force_norm_n"], 3.0)
        self.assertEqual(summary["bucket_mpm_torque_norm_nm"], 5.0)
        self.assertFalse(contact.wrench_summary({
            "force": [float("nan"), 0.0, 0.0],
            "torque": [0.0, 0.0, 0.0],
            "point": [0.0, 0.0, 0.0],
        })["bucket_mpm_wrench_finite"])

    def test_restore_audit_must_bind_both_immutable_inputs(self):
        checkpoint = {"path": "/tmp/a.npz", "sha256": "a" * 64,
                      "bytes": 1}
        metadata = {"path": "/tmp/a.npz.json", "sha256": "b" * 64,
                    "bytes": 1}
        document = {
            "schema":
                "layered_47_sparse_approach_handoff_restore_audit_v1",
            "status":
                "sparse_approach_handoff_restore_audit_passed",
            "contact_handoff_eligible": True,
            "handoff_checkpoint": checkpoint,
            "handoff_metadata": metadata,
            "checks": {"restore_exact": True},
        }
        self.assertTrue(contact.audit_report_passes(
            document, checkpoint, metadata))
        self.assertFalse(contact.audit_report_passes(
            {**document, "contact_handoff_eligible": False},
            checkpoint, metadata))
        self.assertFalse(contact.audit_report_passes(
            {**document, "handoff_metadata": {**metadata, "bytes": 2}},
            checkpoint, metadata))


if __name__ == "__main__":
    unittest.main()
