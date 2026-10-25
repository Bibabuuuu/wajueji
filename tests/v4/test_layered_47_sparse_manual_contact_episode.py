from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_sparse_manual_contact_episode as manual  # noqa: E402
from drive_load_observation import observe_drive_load  # noqa: E402


class FakeChain:
    def __init__(self):
        self.axes = np.asarray([1, 10, 12, 15])
        self.limits = {
            10: np.asarray([0.0, 1.0]),
            12: np.asarray([0.0, 1.0]),
            15: np.asarray([0.0, 0.86]),
        }
        self.q = None

    def advance(self, command, dt):
        if np.any(np.abs(command) >
                  manual.KINEMATIC_PREVIEW_COMMAND_LIMITS_SI + 1.0e-12):
            raise ValueError("command_budget")
        self.q = self.q.copy()
        self.q[self.axes] += np.asarray(command) * dt
        return self.q.copy(), None


class MarginChain(FakeChain):
    """Mirror the legacy all-joint margin gate before geometry solving."""

    def advance(self, command, dt):
        candidate = self.q.copy()
        candidate[self.axes] += np.asarray(command) * dt
        for index, (low, high) in self.limits.items():
            if not low + .005 <= candidate[index] <= high - .005:
                raise ValueError("joint_margin")
        return super().advance(command, dt)


class SparseManualContactEpisodeTest(unittest.TestCase):
    def test_budget_excludes_paused_time_and_restarts_after_notice(self):
        budget = manual.ManualSessionBudget(100.0)
        self.assertIsNone(budget.poll(2000.0, 0, True))
        self.assertIsNone(budget.poll(2100.0, 0, False))
        self.assertEqual(budget.poll(3000.0, 10, False), "wall_time_budget")
        self.assertIsNone(budget.poll(10000.0, 10, True))
        self.assertIsNone(budget.poll(10001.0, 10, False))
        self.assertIsNone(budget.poll(10002.0, 11, False))
        self.assertEqual(budget.active_seconds, 1.0)

    def test_simulation_budget_pauses_once_per_segment(self):
        budget = manual.ManualSessionBudget(0.0)
        self.assertEqual(budget.poll(0.0, manual.MAXIMUM_FRAMES, False),
                         "simulation_time_budget")
        self.assertIsNone(budget.poll(1000.0, manual.MAXIMUM_FRAMES, True))
        self.assertIsNone(budget.poll(1001.0, manual.MAXIMUM_FRAMES + 1, False))
        self.assertEqual(budget.poll(1002.0, 2 * manual.MAXIMUM_FRAMES, False),
                         "simulation_time_budget")

    def row(self):
        return {
            "finite": True,
            "forbidden_contact_count": 0,
            "external_rigid_support_substeps": 0,
            "rigid_contact_substeps": {
                "rock_ground": 0,
                "rock_bucket": 7,
                "rock_rock": 0,
                "rock_ramp": 0,
                "rock_truck": 0,
            },
            "unmapped_active_collider_sample_count": 0,
            "nonfinite_collider_sample_count": 0,
            "closure_m": 0.0001,
            "joint_bounds_ok": True,
            "particle_max_speed_m_s": 0.2,
            "maximum_rock_linear_speed_m_s": 0.1,
            "maximum_rock_angular_speed_rad_s": 0.2,
            "bucket_mpm_wrench_finite": True,
            "applied_command_si": [0.0] * 4,
            "bucket_mpm_contact_samples": 3,
            "bucket_mpm_force_norm_n": 2.0,
            "bucket_mpm_torque_norm_nm": 1.0,
            "rock_bucket_contact_substeps": 7,
            "hard_guard": True,
        }

    def initial_qpos(self):
        qpos = np.zeros(16)
        qpos[10] = 0.5
        qpos[12] = 0.5
        qpos[15] = 0.5
        return qpos

    def test_contact_is_expected_but_true_hard_faults_stop(self):
        row = self.row()
        self.assertTrue(all(manual.hard_guard_checks(row).values()))
        self.assertNotIn("bucket_mpm_contact_samples",
                         manual.hard_guard_checks(row))
        self.assertNotIn("rock_bucket_contact_substeps",
                         manual.hard_guard_checks(row))
        for name, value in {
                "forbidden_contact_count": 1,
                "closure_m": 0.0011,
                "bucket_mpm_wrench_finite": False,
                "joint_bounds_ok": False,
                }.items():
            with self.subTest(name=name):
                checks = manual.hard_guard_checks({**row, name: value})
                self.assertFalse(all(checks.values()))

    def test_velocity_references_warn_without_failing_hard_guard(self):
        row = {
            **self.row(),
            "particle_max_speed_m_s": 6.24,
            "maximum_rock_linear_speed_m_s": 5.01,
            "maximum_rock_angular_speed_rad_s": 25.01,
        }
        self.assertTrue(all(manual.hard_guard_checks(row).values()))
        notices = manual.advisory_checks(row)
        self.assertTrue(all(notices.values()))

    def test_enabled_hencky_material_guard_fails_closed(self):
        row = self.row()
        row["hencky_material_guard"] = {"enabled": True, "safe": True}
        self.assertTrue(manual.hard_guard_checks(row)[
            "hencky_material_state_safe"])
        row["hencky_material_guard"]["safe"] = False
        self.assertFalse(manual.hard_guard_checks(row)[
            "hencky_material_state_safe"])

    def test_hard_fault_latches_first_failure_without_overwrite(self):
        state = {
            "active": False,
            "frame": None,
            "failed_checks": [],
            "snapshot": None,
        }
        row = self.row()
        row["hard_guard_checks"] = manual.hard_guard_checks(row)
        row["hard_guard"] = True
        self.assertFalse(manual.latch_hard_fault(state, row, 1))
        failed = {**row, "joint_bounds_ok": False}
        failed["hard_guard_checks"] = manual.hard_guard_checks(failed)
        failed["hard_guard"] = False
        self.assertTrue(manual.latch_hard_fault(state, failed, 2))
        self.assertEqual(state["frame"], 2)
        self.assertEqual(state["failed_checks"], ["joint_bounds"])
        later = {**row, "forbidden_contact_count": 1}
        later["hard_guard_checks"] = manual.hard_guard_checks(later)
        later["hard_guard"] = False
        self.assertFalse(manual.latch_hard_fault(state, later, 3))
        self.assertEqual(state["frame"], 2)

    def test_rock_ground_is_allowed_but_ramp_and_truck_are_hard_stops(self):
        row = self.row()
        row["external_rigid_support_substeps"] = 36
        row["rigid_contact_substeps"]["rock_ground"] = 36
        self.assertTrue(all(manual.hard_guard_checks(row).values()))
        for contact in ("rock_ramp", "rock_truck"):
            with self.subTest(contact=contact):
                contacts = {**row["rigid_contact_substeps"], contact: 1}
                checks = manual.hard_guard_checks({
                    **row, "rigid_contact_substeps": contacts})
                self.assertFalse(checks["rock_ramp_truck_contact_zero"])

    def test_missing_rigid_contact_breakdown_fails_closed(self):
        row = self.row()
        del row["rigid_contact_substeps"]
        checks = manual.hard_guard_checks(row)
        self.assertFalse(checks["rigid_contact_breakdown_present"])

    def test_transport_allows_truck_contact_but_still_rejects_ramp(self):
        row = self.row()
        row["rigid_contact_substeps"]["rock_truck"] = 12
        checks = manual.hard_guard_checks(row, allow_truck=True)
        self.assertTrue(checks["rock_ramp_truck_contact_zero"])
        row["rigid_contact_substeps"]["rock_ramp"] = 1
        checks = manual.hard_guard_checks(row, allow_truck=True)
        self.assertFalse(checks["rock_ramp_truck_contact_zero"])

    def test_projector_preserves_multiaxis_manual_command(self):
        projector = manual.ContactCommandProjector(FakeChain())
        requested = np.asarray([0.01, -0.01, 0.01, 0.02])
        np.testing.assert_allclose(
            projector.project(self.initial_qpos(), requested), requested)
        self.assertIsNone(projector.reason)

    def test_projector_reparameterizes_full_bucket_preview_without_blocking(self):
        projector = manual.ContactCommandProjector(FakeChain())
        requested = np.asarray([0.0, 0.0, 0.0, -0.0675])
        actual = projector.project(self.initial_qpos(), requested)
        np.testing.assert_allclose(actual, requested)
        self.assertIsNone(projector.reason)
        # 150 ms at the requested rate is preserved despite the older
        # kinematic helper's 0.0225 m/s bucket limit.
        self.assertAlmostEqual(
            projector.chain.q[15], 0.5 - 0.0675 * 0.15, places=12)

    def test_projector_only_removes_outward_bucket_at_internal_target(self):
        projector = manual.ContactCommandProjector(FakeChain())
        qpos = self.initial_qpos()
        qpos[15] = manual.BUCKET_INTERNAL_UPPER_TARGET_M
        requested = np.asarray([0.01, -0.01, 0.01, 0.02])
        actual = projector.project(qpos, requested)
        np.testing.assert_allclose(actual[:3], requested[:3])
        self.assertEqual(actual[3], 0.0)
        self.assertEqual(projector.reason, "bucket_internal_upper_target")
        self.assertTrue(projector.bucket_end_stop_projected)
        reverse = projector.project(qpos, [0.0, 0.0, 0.0, -0.02])
        self.assertEqual(reverse[3], -0.02)

    def test_projector_zeros_command_that_previews_outside_joint_limits(self):
        projector = manual.ContactCommandProjector(FakeChain())
        qpos = self.initial_qpos()
        qpos[10] = 0.999
        actual = projector.project(qpos, [0.0, 0.0225, 0.0, 0.0])
        np.testing.assert_array_equal(actual, np.zeros(4))
        self.assertEqual(projector.reason, "joint_working_range")

    def test_bucket_soft_overshoot_allows_lift_and_small_retreat(self):
        chain = MarginChain()
        limits = chain.limits
        projector = manual.ContactCommandProjector(chain)
        qpos = self.initial_qpos()
        qpos[15] = .856188
        for requested in ([0, .0225, 0, 0], [0, .0225, 0, -.0001],
                          [0, 0, 0, -.02]):
            with self.subTest(requested=requested):
                before = qpos.copy()
                np.testing.assert_allclose(
                    projector.project(qpos, requested), requested)
                np.testing.assert_array_equal(qpos, before)
                self.assertLessEqual(chain.q[15], qpos[15])
                self.assertIs(chain.limits, limits)
                self.assertEqual(chain.limits[15][1], .86)

    def test_full_curl_near_end_clips_only_bucket_preview(self):
        projector = manual.ContactCommandProjector(MarginChain())
        qpos = self.initial_qpos()
        qpos[15] = .848
        requested = np.array([.01, .0225, .01, .0675])
        actual = projector.project(qpos, requested)
        np.testing.assert_array_equal(actual[:3], requested[:3])
        self.assertAlmostEqual(actual[3], (.852-.848)/.15)
        self.assertAlmostEqual(projector.chain.q[15], .852)
        self.assertTrue(projector.bucket_end_stop_projected)
        qpos[15] = .856188
        actual = projector.project(qpos, requested)
        np.testing.assert_array_equal(actual[:3], requested[:3])
        self.assertEqual(actual[3], 0)

    def test_soft_overshoot_does_not_relax_other_axes_or_closure(self):
        chain = MarginChain()
        limits = chain.limits
        projector = manual.ContactCommandProjector(chain)
        qpos = self.initial_qpos()
        qpos[15] = .856188
        qpos[10] = .997
        np.testing.assert_array_equal(
            projector.project(qpos, [0, .001, 0, 0]), np.zeros(4))
        self.assertEqual(projector.reason, "joint_margin")
        self.assertIs(chain.limits, limits)
        qpos[10] = .5
        def fail_closure(command, dt):
            raise ValueError("closure_unresolved")
        chain.advance = fail_closure
        np.testing.assert_array_equal(
            projector.project(qpos, [0, .001, 0, 0]), np.zeros(4))
        self.assertEqual(projector.reason, "closure_unresolved")
        self.assertIs(chain.limits, limits)

    def test_actual_hard_violation_and_nonfinite_state_still_rejected(self):
        projector = manual.ContactCommandProjector(MarginChain())
        for value in (.860001, -.001, float('nan'), float('inf')):
            qpos = self.initial_qpos()
            qpos[15] = value
            for request in ([0, .01, 0, 0], [0, 0, 0, -.01], [0]*4):
                with self.subTest(value=value, request=request):
                    np.testing.assert_array_equal(
                        projector.project(qpos, request), np.zeros(4))
                    self.assertEqual(projector.reason, "joint_working_range")

    def test_preview_cannot_cross_physical_bound(self):
        chain = MarginChain()
        original = chain.advance
        def bad_geometry(command, dt):
            predicted, error = original(command, dt)
            predicted[12] = 1.001
            return predicted, error
        chain.advance = bad_geometry
        projector = manual.ContactCommandProjector(chain)
        qpos = self.initial_qpos()
        qpos[15] = .856188
        np.testing.assert_array_equal(
            projector.project(qpos, [0, .001, 0, 0]), np.zeros(4))
        self.assertEqual(projector.reason, "joint_working_range")

    def test_episode_summary_never_auto_grants_success_or_training(self):
        rows = [self.row()]
        phases = [{"label": "penetrate_deepen"},
                  {"label": "curl_load"},
                  {"label": "penetrate_deepen"}]
        annotations = [{"label": "jam_observed"}]
        summary = manual.summarize_episode(rows, phases, annotations)
        self.assertTrue(summary["soil_contact_observed"])
        self.assertTrue(summary["rock_bucket_contact_observed"])
        self.assertEqual(summary["phase_labels_seen"],
                         ["penetrate_deepen", "curl_load"])
        self.assertIsNone(summary["automatic_success_label"])
        self.assertFalse(summary["training_eligible"])
        self.assertTrue(summary["posthoc_review_required"])
        self.assertFalse(summary["load_contact_thresholds_calibrated"])

    def test_drive_load_observation_keeps_mixed_units_per_axis(self):
        state = {
            "equivalent_drive_load": [2.0, 20.0, -30.0, 40.0],
            "external_load_compensation": [-2.0, -20.0, 30.0, -40.0],
            "unloaded_feedforward": [1.0, 2.0, 3.0, 4.0],
            "actuator_force": list(range(12)),
        }
        result = observe_drive_load(
            state, previous=[1.5, 18.0, -31.0, 35.0],
            marker_baseline=[1.0, 10.0, -20.0, 30.0])
        self.assertEqual(result["drive_load_axis_units"],
                         ["N*m", "N", "N", "N"])
        np.testing.assert_allclose(
            result["equivalent_drive_load_delta_from_previous"],
            [0.5, 2.0, 1.0, 5.0])
        np.testing.assert_allclose(
            result["equivalent_drive_load_delta_from_phase_marker"],
            [1.0, 10.0, -10.0, 10.0])
        self.assertIsNone(result["load_contact_classification"])
        self.assertFalse(result["load_thresholds_applied"])

    def test_protocol_is_bounded_and_manual(self):
        self.assertEqual(manual.HZ, 60)
        self.assertEqual(manual.RIGID_INTERNAL_HZ, 4800)
        self.assertEqual(manual.MAXIMUM_SIMULATION_SECONDS, 30.0)
        self.assertEqual(manual.FINAL_ZERO_HOLD_FRAMES, 30)
        self.assertEqual(manual.DECISION_OBSERVATION_STRIDE, 3)
        self.assertEqual(manual.PHASE_MARKERS["1"], "penetrate_deepen")
        self.assertIn("manual_contact_pilot_only",
                      manual.ACQUISITION_DOMAIN)

    def test_explicit_rear_face_target_replaces_legacy_strip_marker(self):
        execution = {
            "observation": {"selection": {
                "entry_xy_m": [5.6, 0.0], "surface_z_m": 1.4}},
            "manual_contact_target": {
                "entry_xy_m": [5.85, -0.05], "surface_z_m": 0.65,
                "source": "rear_lower_face_from_observed_profile"},
        }
        self.assertEqual(
            manual.selected_contact_target(execution),
            execution["manual_contact_target"])

    def test_legacy_execution_keeps_original_strip_marker(self):
        selection = {"entry_xy_m": [5.6, 0.0], "surface_z_m": 1.4}
        self.assertEqual(manual.selected_contact_target({
            "observation": {"selection": selection}}), selection)


if __name__ == "__main__":
    unittest.main()
