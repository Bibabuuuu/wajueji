#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_interactive_proxy_practice as practice  # noqa: E402


class InteractiveProxyPracticeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.loaded = practice.load_config(
            ROOT / "configs/v4/layered-47-interactive-proxy-practice-v1.json",
            ROOT)

    def test_normalized_actions_map_to_frozen_physical_limits(self):
        action = [1.0, -1.0, 0.5, -0.5]
        command = practice.normalized_to_command(
            action, self.config["control"])
        np.testing.assert_allclose(
            command, [0.12, -0.0225, 0.01125, -0.03375],
            rtol=0.0, atol=1e-15)

    def test_rejects_out_of_range_action(self):
        with self.assertRaisesRegex(
                ValueError, "interactive_proxy_practice_action"):
            practice.normalized_to_command(
                [0.0, 0.0, 0.0, 1.01], self.config["control"])

    def test_practice_cannot_write_training_or_checkpoint(self):
        scope = self.config["scope"]
        self.assertTrue(scope["practice_only"])
        self.assertFalse(scope["checkpoint_written"])
        self.assertFalse(scope["manual_collection_eligible"])
        self.assertFalse(scope["training_eligible"])
        self.assertFalse(scope["writes_training_episode"])

    def test_hardware_mapping_is_the_validated_mapping(self):
        self.assertEqual(
            self.loaded["manual_config"]["axis_order"],
            list(practice.AXIS_NAMES))
        self.assertEqual(
            self.loaded["manual_config"]["gamepad"][
                "emergency_stop_button"], "back")

    def test_current_interactive_entries_require_gpu_density_surface(self):
        display = self.config["display"]
        self.assertEqual(
            display["soil_render_mode"],
            "gpu_density_surface_with_loose_particles")
        self.assertEqual(display["surface_voxel_m"], 0.04)
        self.assertEqual(display["surface_update_interval_frames"], 1)


if __name__ == "__main__":
    unittest.main()
