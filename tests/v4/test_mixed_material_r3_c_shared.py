import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_multirock_candidate_r3_c import sha256  # noqa: E402
from mixed_material_r3_c_shared_contract import validate_config  # noqa: E402


CONFIG = (
    ROOT / "configs" / "v4"
    / "mixed-material-r3-c-shared-four-round-v1.json")
REPORT_DIR = (
    ROOT / "reports" / "v4"
    / "mixed-material-r3-c-shared-four-round-2026-09-20-v1")
ROCK_IDS = ("r1-rock-000", "r1-rock-001")


class MixedMaterialR3CSharedContractTest(unittest.TestCase):
    def test_config_freezes_one_shared_matrix_and_exact_four_rounds(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertEqual(config["probe"]["rock_ids"], list(ROCK_IDS))
        self.assertEqual(
            config["probe"]["qualification_rounds"],
            ["clear", "mpm_only", "rigid_only", "combined"])
        self.assertTrue(config["scope"]["one_shared_mpm_matrix"])
        self.assertTrue(config["scope"]["four_round_force_partition"])
        self.assertFalse(config["scope"]["repeat_qualification"])
        self.assertFalse(config["scope"]["dt_refinement_qualification"])
        self.assertFalse(config["scope"]["r3_c_physics_qualified"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])

    def test_rejects_hash_round_threshold_and_scope_drift(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["routing_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["probe"]["qualification_rounds"].reverse()
        with self.assertRaisesRegex(ValueError, "shared_probe_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["probe"]["minimum_active_component_impulse_ns"] = 0.0
        with self.assertRaisesRegex(ValueError, "shared_threshold_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["repeat_qualification"] = True
        with self.assertRaisesRegex(ValueError, "shared_scope_expanded"):
            validate_config(config, ROOT)

    def test_formal_report_passes_shared_gate_without_claiming_full_r3_c(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(
            report["status"], "r3_c_shared_matrix_four_round_passed")
        self.assertTrue(report["shared_four_round_qualified"])
        self.assertFalse(report["r3_c_physics_qualified"])
        self.assertFalse(report["training_qualified"])
        self.assertEqual(len(report["checks"]), 15)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            set(report["episodes"]),
            {"clear", "mpm_only", "rigid_only", "combined"})

        episodes = report["episodes"]
        clear = episodes["clear"]["accumulated_impulse_magnitudes"]
        mpm_only = episodes["mpm_only"]["accumulated_impulse_magnitudes"]
        rigid_only = episodes["rigid_only"]["accumulated_impulse_magnitudes"]
        combined = episodes["combined"]["accumulated_impulse_magnitudes"]
        self.assertEqual(clear["bucket_total_ns"], 0.0)
        self.assertGreater(mpm_only["bucket_mpm_ns"], 0.01)
        self.assertEqual(mpm_only["bucket_rigid_ns"], 0.0)
        self.assertEqual(rigid_only["bucket_mpm_ns"], 0.0)
        self.assertGreater(rigid_only["bucket_rigid_ns"], 0.01)
        self.assertGreater(combined["bucket_mpm_ns"], 0.01)
        self.assertGreater(combined["bucket_rigid_ns"], 0.01)
        for episode in episodes.values():
            ownership = episode["collider_ownership"]
            self.assertEqual(
                ownership["rock_id_to_newton_body_index"],
                {"r1-rock-000": 0, "r1-rock-001": 1})
            self.assertEqual(ownership["bucket_newton_body_index"], 2)
            self.assertTrue(ownership["mapping_disjoint"])
            self.assertEqual(ownership["unmapped_active_sample_count"], 0)
            self.assertEqual(ownership["nonfinite_sample_count"], 0)
            self.assertTrue(episode["mpm_history_reset"]["passed"])
        for rock_id in ROCK_IDS:
            self.assertGreater(mpm_only["rock_mpm_ns"][rock_id], 0.01)
            self.assertGreater(combined["rock_mpm_ns"][rock_id], 0.01)

    def test_formal_report_is_bound_to_config_runner_models_and_bridge(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        model_manifest = json.loads(
            (REPORT_DIR / "generated-model-manifest.json")
            .read_text(encoding="utf-8"))
        bridge_manifest = json.loads(
            (REPORT_DIR / "generated-bridge-manifest.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(
            report["implementation"]["runner_sha256"],
            sha256(SOURCE / "run_mixed_material_r3_c_shared_probe.py"))
        self.assertEqual(
            report["inputs"]["derived_model_sha256"]["rigid_enabled"],
            sha256(REPORT_DIR / "generated-model.xml"))
        self.assertEqual(
            report["inputs"]["derived_model_sha256"]
            ["bucket_rigid_disabled"],
            sha256(REPORT_DIR / "generated-model-bucket-rigid-disabled.xml"))
        self.assertEqual(
            report["inputs"]["bridge_source_sha256"],
            sha256(REPORT_DIR / "generated-bridge.cc"))
        self.assertEqual(
            model_manifest["candidate_model_sha256"],
            sha256(REPORT_DIR / "generated-model.xml"))
        self.assertEqual(
            bridge_manifest["generated_source_sha256"],
            sha256(REPORT_DIR / "generated-bridge.cc"))
        self.assertFalse(model_manifest["physics_qualified"])
        self.assertFalse(bridge_manifest["physics_qualified"])


if __name__ == "__main__":
    unittest.main()
