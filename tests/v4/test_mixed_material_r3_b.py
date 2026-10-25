import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_coupled_bridge_r3_b import (  # noqa: E402
    render_bridge,
    sha256,
    validate_config,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-b.json"
REPORT_DIR = ROOT / "reports" / "v4" / "mixed-material-r3-b-2026-09-20-v1"


class MixedMaterialR3BContractTest(unittest.TestCase):
    def test_config_is_one_rock_small_patch_and_no_training(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertTrue(config["scope"]["bucket_mpm_contact_enabled"])
        self.assertTrue(config["scope"]["three_component_bucket_wrench_accounting"])
        self.assertEqual(config["probe"]["collider_velocity_mode"], "backward")
        self.assertEqual(
            config["probe"]["collider_velocity_semantics"],
            "previous_to_current_outer_step_pose_difference")
        self.assertEqual(
            set(config["probe"]["qualification_rounds"]),
            {"clear", "mpm_only", "rigid_only", "combined"})
        self.assertEqual(
            config["probe"]["dt_refinement_acceptance_impulses"],
            ["bucket_mpm_ns", "bucket_measured_total_ns"])
        self.assertFalse(config["scope"]["multi_rock"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])

    def test_bridge_is_deterministic_and_uses_interval_start_origin(self):
        first, first_manifest = render_bridge(CONFIG, ROOT)
        second, second_manifest = render_bridge(CONFIG, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertIn("excavation_mixed_coupled_bridge_r3_b_v1", first)
        self.assertIn("bucket_linear_velocity_world_m_s", first)
        self.assertIn("bucket_angular_velocity_world_rad_s", first)
        self.assertIn("bucket_interval_reference_point_world_m", first)
        self.assertIn("bucket_interval_reference_point.data()", first)
        self.assertIn("wrench.point.data(), wrench.body", first)
        self.assertNotIn("data->xpos + 3 * wrench.body, wrench.body", first)
        self.assertFalse(first_manifest["frozen_r1_r2_or_r3_a_modified"])

    def test_rejects_scope_expansion_and_hash_rebinding(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["multi_rock"] = True
        with self.assertRaisesRegex(ValueError, "r3_b_scope_expanded"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["r3_a_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)

    def test_formal_report_and_generated_bridge_are_bound(self):
        report = json.loads((REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (REPORT_DIR / "generated-bridge-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "r3_b_bucket_force_partition_probe_passed")
        self.assertTrue(report["physics_qualified"])
        self.assertEqual(len(report["checks"]), 26)
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(
            report["comparison"]["dt_refinement"]
            ["rigid_component_is_diagnostic_not_converged"])
        self.assertFalse(report["comparison"]["byte_exact_repeat"])
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(
            report["implementation"]["runner_sha256"],
            sha256(SOURCE / "run_mixed_material_r3_b_probe.py"))
        generated = REPORT_DIR / "generated-bridge.cc"
        self.assertEqual(manifest["generated_source_sha256"], sha256(generated))
        self.assertEqual(
            report["inputs"]["bridge_source_manifest"]["generated_source_sha256"],
            sha256(generated))
        self.assertFalse(manifest["frozen_r1_r2_or_r3_a_modified"])


if __name__ == "__main__":
    unittest.main()
