#!/usr/bin/env python3

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_excavation_preflight as preflight  # noqa: E402
import run_layered_47_interactive_proxy_benchmark as benchmark  # noqa: E402


class InteractiveProxyBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.loaded = benchmark.load_config(
            ROOT / "configs/v4/layered-47-interactive-proxy-benchmark-v1.json",
            ROOT)

    def test_formal_control_profile_is_unchanged(self):
        formal = preflight.mpm_solver_profile(self.loaded["base"])
        self.assertEqual(
            self.loaded["profiles"]["formal_5cm_fixed"], formal)
        self.assertEqual(formal["voxel_size_m"], 0.05)
        self.assertEqual(formal["grid_type"], "fixed")
        self.assertEqual(formal["grid_padding"], 12)
        self.assertEqual(formal["max_iterations"], 50)
        self.assertEqual(formal["tolerance"], 1.0e-4)

    def test_only_voxel_and_grid_layout_change_across_profiles(self):
        formal = self.loaded["profiles"]["formal_5cm_fixed"]
        mutable = {"voxel_size_m", "grid_type", "grid_padding"}
        for name in (
                "formal_5cm_fixed", "voxel_10cm_fixed",
                "voxel_5cm_sparse", "voxel_10cm_sparse"):
            profile = self.loaded["profiles"][name]
            for key, value in formal.items():
                if key not in mutable:
                    self.assertEqual(profile[key], value)

    def test_iteration_sweep_changes_only_the_iteration_budget(self):
        control = self.loaded["profiles"]["voxel_10cm_sparse"]
        for name, expected in (
                ("voxel_10cm_sparse_iter25", 25),
                ("voxel_10cm_sparse_iter12", 12)):
            candidate = self.loaded["profiles"][name]
            self.assertEqual(candidate["max_iterations"], expected)
            for key, value in control.items():
                if key != "max_iterations":
                    self.assertEqual(candidate[key], value)

    def test_profile_cannot_change_frozen_solver_semantics(self):
        candidate = copy.deepcopy(
            self.loaded["profiles"]["voxel_10cm_sparse"])
        candidate["collider_velocity_mode"] = "backward"
        with self.assertRaisesRegex(
                ValueError,
                "excavation_solver_profile_frozen_collider_velocity_mode"):
            preflight.mpm_solver_profile(self.loaded["base"], candidate)

    def test_scope_cannot_grant_collection_or_training(self):
        self.assertFalse(self.config["scope"]["manual_collection_eligible"])
        self.assertFalse(self.config["scope"]["training_eligible"])
        self.assertFalse(self.config["scope"]["formal_a30_equivalent"])

    def test_percentile_helper(self):
        self.assertIsNone(benchmark._percentile([], 0.5))
        self.assertEqual(benchmark._percentile([1.0, 3.0], 0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
