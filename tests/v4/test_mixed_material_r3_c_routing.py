import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_multirock_candidate_r3_c import sha256  # noqa: E402
from mixed_material_r3_c_routing_contract import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-c-routing-v1.json"
REPORT_DIR = (
    ROOT / "reports" / "v4"
    / "mixed-material-r3-c-routing-2026-09-20-v1")


class MixedMaterialR3CRoutingContractTest(unittest.TestCase):
    def test_config_freezes_two_isolated_routes_without_scope_expansion(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertEqual(
            config["routing_probe"]["rock_ids"],
            ["r1-rock-000", "r1-rock-001"])
        self.assertEqual(
            config["routing_probe"]["rounds"],
            ["rock_000_mpm_only", "rock_001_mpm_only"])
        self.assertTrue(config["scope"]["per_id_newton_body_and_collider_mapping"])
        self.assertTrue(config["scope"]["two_single_rock_mpm_routes"])
        self.assertFalse(config["scope"]["bucket_mpm_collision"])
        self.assertFalse(config["scope"]["four_round_force_partition"])
        self.assertFalse(config["scope"]["feedback_qualification"])
        self.assertFalse(config["scope"]["repeat_qualification"])
        self.assertFalse(config["scope"]["dt_refinement_qualification"])
        self.assertFalse(config["scope"]["r3_c_physics_qualified"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])

    def test_rejects_hash_route_threshold_and_scope_drift(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["r3_c_preflight_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["routing_probe"]["rounds"].reverse()
        with self.assertRaisesRegex(ValueError, "routing_probe_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["routing_probe"]["minimum_target_direct_impulse_ns"] = 0.0
        with self.assertRaisesRegex(ValueError, "routing_threshold_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["four_round_force_partition"] = True
        with self.assertRaisesRegex(ValueError, "routing_scope_expanded"):
            validate_config(config, ROOT)

    def test_formal_report_proves_both_routes_without_claiming_full_r3_c(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "r3_c_per_rock_routing_probe_passed")
        self.assertTrue(report["routing_physics_qualified"])
        self.assertFalse(report["r3_c_physics_qualified"])
        self.assertFalse(report["training_qualified"])
        self.assertEqual(len(report["checks"]), 13)
        self.assertTrue(all(report["checks"].values()))
        episodes = report["episodes"]
        expected_bodies = {"r1-rock-000": 0, "r1-rock-001": 1}
        for target, inactive in (
            ("r1-rock-000", "r1-rock-001"),
            ("r1-rock-001", "r1-rock-000"),
        ):
            episode = episodes[target]
            self.assertEqual(
                episode["rock_id_to_newton_body_index"], expected_bodies)
            self.assertEqual(len(episode["rock_id_to_collider_ids"][target]), 1)
            self.assertEqual(episode["rock_id_to_collider_ids"][inactive], [])
            impulses = episode["accumulated_direct_mpm_impulse_ns"]
            self.assertGreaterEqual(impulses[target], 0.01)
            self.assertLessEqual(impulses[inactive], 1.0e-12)
            self.assertGreater(episode["contact_frames"][target], 0)
            self.assertEqual(episode["contact_frames"][inactive], 0)
            self.assertEqual(episode["maximum_wrench_echo_residual"], 0.0)
            self.assertTrue(episode["mpm_history_reset"]["passed"])

    def test_formal_report_is_bound_to_config_runner_model_and_bridge(self):
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
            sha256(SOURCE / "run_mixed_material_r3_c_routing_probe.py"))
        self.assertEqual(
            report["inputs"]["derived_model_sha256"],
            sha256(REPORT_DIR / "generated-model.xml"))
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
