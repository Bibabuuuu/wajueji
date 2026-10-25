#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_mound_fortyseven_recovery_hold as recovery  # noqa: E402


PROBE = {
    "maximum_stable_rock_linear_speed_m_s": 0.01,
    "maximum_stable_rock_angular_speed_rad_s": 0.15,
    "maximum_stable_particle_p99_speed_m_s": 0.01,
    "minimum_world_z_m": -0.05,
}
POLICY = {
    "strict_particle_max_speed_m_s": 0.05,
    "transient_particle_max_speed_ceiling_m_s": 0.10,
    "maximum_transient_frames": 6,
    "maximum_consecutive_transient_frames": 2,
}


def row(maximum: float, *, p99: float = 0.001) -> dict:
    return {
        "frame": 1,
        "finite": True,
        "centers_inside": True,
        "external_rigid_support_substeps": 0,
        "forbidden_contact_count": 0,
        "global_momentum_residual_ns": 0.0,
        "maximum_rock_linear_speed_m_s": 0.001,
        "maximum_rock_angular_speed_rad_s": 0.01,
        "particle_p99_speed_m_s": p99,
        "particle_minimum_z_m": 0.0,
        "particle_max_speed_m_s": maximum,
    }


class FortySevenRecoveryHoldTest(unittest.TestCase):
    def test_strict_boundary_is_not_lowered(self):
        self.assertEqual(
            recovery.classify_hold_frame(row(0.05), PROBE, POLICY)[
                "classification"], "strict")
        self.assertEqual(
            recovery.classify_hold_frame(row(0.0500001), PROBE, POLICY)[
                "classification"], "bounded_sparse_transient")

    def test_transient_ceiling_and_p99_are_hard_limits(self):
        self.assertEqual(
            recovery.classify_hold_frame(row(0.100001), PROBE, POLICY)[
                "classification"], "hard_failure")
        self.assertEqual(
            recovery.classify_hold_frame(row(0.06, p99=0.010001), PROBE, POLICY)[
                "classification"], "hard_failure")

    def test_strict_recovery_tail_resets_after_transient(self):
        trace = [row(0.01) for _ in range(142)] + [row(0.0539)]
        trace += [row(0.01) for _ in range(60)]
        analysis = recovery.analyze_trace(trace, PROBE, POLICY)
        self.assertEqual(analysis["transient_frames"], 1)
        self.assertEqual(analysis["trailing_strict_recovery_frames"], 60)
        self.assertTrue(analysis["within_transient_budget"])

    def test_consecutive_transient_budget_is_enforced(self):
        trace = [row(0.06), row(0.07), row(0.08)]
        analysis = recovery.analyze_trace(trace, PROBE, POLICY)
        self.assertEqual(analysis["maximum_consecutive_transient_frames"], 3)
        self.assertFalse(analysis["within_transient_budget"])


if __name__ == "__main__":
    unittest.main()
