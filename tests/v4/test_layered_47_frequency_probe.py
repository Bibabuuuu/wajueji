import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/v4"))
import run_layered_47_frequency_probe as probe


class FrequencyProbeTest(unittest.TestCase):
    def test_diagnostic_abort_is_not_qualification(self):
        row = dict(safe=True, bucket_mpm_contact_samples=0, rock_bucket_contact_substeps=0,
                   maximum_rock_linear_speed_m_s=.02, maximum_rock_angular_speed_rad_s=.1,
                   particle_max_speed_m_s=.04, particle_p99_speed_m_s=.02,
                   maximum_rock_displacement_from_source_m=.0002)
        self.assertTrue(probe.diagnostic_safe(row))
        self.assertFalse(probe.diagnostic_safe({**row, "maximum_rock_displacement_from_source_m": .006}))
        self.assertFalse(probe.diagnostic_safe({**row, "safe": False}))
    def test_subcycle_keeps_force_and_integrates_one_impulse(self):
        calls = []
        def step(bridge, dt, substeps, command, wrenches):
            calls.append((dt, substeps, wrenches))
            return {"contact_substeps": {"rock_bucket": int(len(calls) == 1)},
                    "forbidden_contact_count": int(len(calls) == 1),
                    "closure_m": .0005 / len(calls)}
        wrench = {"rock": {"force": [12., 0., 0.]}}
        with patch.object(probe.practice, "_bridge_step", side_effect=step):
            result = probe.subcycled_bridge_step(None, 1/60, 80, np.zeros(4), wrench)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c[1] == 40 and c[2] is wrench for c in calls))
        self.assertAlmostEqual(sum(c[0] * c[2]["rock"]["force"][0] for c in calls), 12/60)
        self.assertEqual(result["contact_substeps"]["rock_bucket"], 1)
        self.assertEqual(result["forbidden_contact_count"], 1)
        self.assertEqual(result["closure_m"], .0005)

    def test_equal_duration_and_command_integral(self):
        for hz in (60, 120):
            schedule = probe.schedule(hz)
            self.assertEqual(sum(n for _, n, _ in schedule) / hz, 2.)
            np.testing.assert_allclose(sum(n * c / hz for _, n, c in schedule), [.015, 0, 0, 0])

    def test_internal_dt_and_base_unchanged(self):
        original = {"probe": {"output_dt_s": 1/120, "mujoco_internal_substeps": 40}, "mpm": {"material": 123}}
        before = copy.deepcopy(original)
        for hz in (60, 120):
            result = probe.probe_config(original, hz)
            self.assertAlmostEqual(result["probe"]["output_dt_s"] / result["probe"]["mujoco_internal_substeps"], 1/4800, places=18)
            self.assertEqual(result["mpm"], before["mpm"])
        self.assertEqual(original, before)

    def test_no_unbounded_frequency(self):
        with self.assertRaises(ValueError):
            probe.schedule(30)

    def test_simulation_throughput_not_fps(self):
        rows = [{"total_frame_wall_seconds": .2, "mpm_wall_seconds": .1, "mujoco_wall_seconds": .09}] * 60
        self.assertAlmostEqual(probe.timing(rows, 60)["simulation_seconds_per_wall_second"], 1/12)

    def test_strict_gate_retains_stability_and_contact_limits(self):
        row = dict(safe=True, particle_max_speed_m_s=.02, particle_p99_speed_m_s=.001,
                   maximum_rock_linear_speed_m_s=.002, maximum_rock_angular_speed_rad_s=.01,
                   bucket_mpm_contact_samples=0, rock_bucket_contact_substeps=0)
        limits = dict(maximum_stable_particle_speed_m_s=.05, maximum_stable_particle_p99_speed_m_s=.01,
                      maximum_stable_rock_linear_speed_m_s=.01, maximum_stable_rock_angular_speed_rad_s=.15)
        self.assertTrue(probe.strict_row(row, limits))
        for key, value in (("safe", False), ("particle_max_speed_m_s", .051), ("bucket_mpm_contact_samples", 1)):
            self.assertFalse(probe.strict_row({**row, key: value}, limits))
