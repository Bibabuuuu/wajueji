#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_47_interactive_proxy_resume_audit as audit  # noqa: E402


class InteractiveProxyResumeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.loaded = audit.load_config(
            ROOT / "configs/v4/layered-47-interactive-proxy-resume-audit-v1.json",
            ROOT)

    def test_binds_selected_sparse_profile(self):
        profile = self.loaded["profiles"][
            self.config["benchmark_config"]["profile"]]
        self.assertEqual(profile["grid_type"], "sparse")
        self.assertEqual(profile["voxel_size_m"], 0.1)
        self.assertEqual(profile["grid_padding"], 2)

    def test_audit_is_read_only_and_nonqualifying(self):
        self.assertFalse(self.config["audit"]["writes_checkpoint"])
        self.assertFalse(self.config["scope"]["formal_a30_equivalent"])
        self.assertFalse(self.config["scope"]["manual_collection_eligible"])
        self.assertFalse(self.config["scope"]["training_eligible"])

    def test_requires_long_strict_hold(self):
        self.assertEqual(self.config["audit"]["frames"], 120)
        self.assertTrue(self.config["audit"]["all_frames_must_be_strict"])
        self.assertTrue(self.config["audit"]["writes_progress_journal"])


if __name__ == "__main__":
    unittest.main()
