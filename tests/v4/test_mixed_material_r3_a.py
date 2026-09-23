import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_coupled_bridge_r3_a import (  # noqa: E402
    render_bridge,
    sha256,
    validate_config,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-a.json"
REPORT_ROOT = ROOT / "reports" / "v4" / "mixed-material-r3-a-2026-09-20-v1"


class MixedMaterialR3AContractTest(unittest.TestCase):
    def test_config_binds_frozen_inputs_and_scope(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertFalse(config["scope"]["multi_rock"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])
        self.assertFalse(config["scope"]["frozen_r1_or_r2_modified"])

    def test_bridge_transform_is_deterministic_and_bounded(self):
        first, manifest = render_bridge(CONFIG, ROOT)
        second, second_manifest = render_bridge(CONFIG, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(manifest, second_manifest)
        self.assertIn("excavation_mixed_coupled_bridge_r3_a_v1", first)
        self.assertIn("maximum_external_wrench_count", first)
        self.assertIn("unapproved R3-A wrench body", first)
        self.assertIn("mj_applyFT", first)
        self.assertNotIn("reserved_zero_only", first)
        self.assertFalse(manifest["frozen_r1_source_modified"])

    def test_rejects_scope_expansion(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["training"] = True
        with self.assertRaisesRegex(ValueError, "r3_a_scope_expanded"):
            validate_config(config, ROOT)

    def test_rejects_frozen_hash_rebinding(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["r1_bridge_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)

    def test_formal_probe_is_bound_passed_and_does_not_expand_scope(self):
        report = json.loads((REPORT_ROOT / "report.json").read_text(
            encoding="utf-8"))
        manifest = json.loads((REPORT_ROOT / "generated-bridge-r3-a-manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual(
            report["schema"],
            "excavation_mixed_material_r3_a_probe_report_v1")
        self.assertEqual(
            report["status"], "r3_a_one_rock_coupling_probe_passed")
        self.assertTrue(report["physics_qualified"])
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(report["comparison"]["deterministic_repeat"])
        self.assertGreater(
            report["comparison"]["coupled_minus_uncoupled_final_rock_z_m"],
            0.01)
        self.assertTrue(all(
            episode["reset_audit"]["passed"]
            for episode in report["episodes"].values()))
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(
            manifest["generated_source_sha256"],
            sha256(REPORT_ROOT / "generated-bridge-r3-a.cc"))
        self.assertEqual(
            report["inputs"]["bridge_source_manifest"]["generated_source_sha256"],
            manifest["generated_source_sha256"])
        self.assertIn("bucket-to-MPM contact", report["scope_limit"])
        self.assertIn("official 2.2 m envelope", report["scope_limit"])
        self.assertIn("training are not qualified", report["scope_limit"])


if __name__ == "__main__":
    unittest.main()
