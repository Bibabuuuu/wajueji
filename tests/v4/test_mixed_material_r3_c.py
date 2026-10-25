import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_coupled_bridge_r3_c import render_bridge  # noqa: E402
from build_mixed_multirock_candidate_r3_c import (  # noqa: E402
    build_two_rock_candidate,
    sha256,
    validate_config,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-c.json"
INVENTORY = Path(
    "/home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json")
REPORT_DIR = (
    ROOT / "reports" / "v4"
    / "mixed-material-r3-c-bridge-preflight-2026-09-20-v1")


class MixedMaterialR3CContractTest(unittest.TestCase):
    def test_config_is_exact_two_rock_structural_gate(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertEqual(config["bridge"]["required_rock_count"], 2)
        self.assertEqual(
            config["bridge"]["allowed_external_wrench_bodies"],
            ["bucket", "r1-rock-000", "r1-rock-001"])
        self.assertEqual(len(config["two_rock_fixture"]["rocks"]), 2)
        self.assertTrue(config["scope"]["exactly_two_rocks"])
        self.assertFalse(config["scope"]["multi_rock_mpm_physics"])
        self.assertFalse(config["first_gate"]["gpu_mpm_step"])
        self.assertFalse(config["first_gate"]["physics_qualified"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])

    def test_two_rock_model_generation_is_deterministic_and_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.xml"
            second = Path(temporary) / "second.xml"
            first_manifest = build_two_rock_candidate(
                CONFIG, ROOT, INVENTORY, first)
            second_manifest = build_two_rock_candidate(
                CONFIG, ROOT, INVENTORY, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                first_manifest["expected_candidate_contract"],
                {"nq": 33, "nv": 30, "nu": 12, "nplugin": 0,
                 "rock_count": 2})
            bodies = {
                body.get("name"): body
                for body in ET.parse(first).getroot().iter("body")
            }
            config = json.loads(CONFIG.read_text(encoding="utf-8"))
            for rock in config["two_rock_fixture"]["rocks"]:
                body = bodies[rock["rock_id"]]
                self.assertEqual(len(list(body.iter("freejoint"))), 1)
                actual = [float(value) for value in body.get("pos").split()]
                self.assertEqual(actual, rock["initial_position_world_m"])
            self.assertFalse(
                first_manifest["frozen_r1_r2_r3_a_r3_b_modified"])
            self.assertFalse(first_manifest["physics_qualified"])

    def test_bridge_is_deterministic_and_exposes_per_rock_contract(self):
        first, first_manifest = render_bridge(CONFIG, ROOT)
        second, second_manifest = render_bridge(CONFIG, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertIn("rock_count != 2", first)
        self.assertIn('"maximum_external_wrench_count\\\":3"', first)
        self.assertIn("applied_rocks_mpm", first)
        self.assertIn("linear_velocity_world_m_s", first)
        self.assertIn("angular_velocity_world_rad_s", first)
        self.assertIn('body_name == "r1-rock-000"', first)
        self.assertIn('body_name == "r1-rock-001"', first)
        self.assertNotIn("applied_rock_mpm_force_world_n", first)
        self.assertEqual(first_manifest["maximum_external_wrench_count"], 3)
        self.assertFalse(first_manifest["frozen_r1_r2_r3_a_r3_b_modified"])
        self.assertFalse(first_manifest["physics_qualified"])

    def test_rejects_scope_expansion_hash_rebinding_and_close_rocks(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["multi_rock_mpm_physics"] = True
        with self.assertRaisesRegex(ValueError, "r3_c_scope_expanded"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["r3_b_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["two_rock_fixture"]["rocks"][1]["initial_position_world_m"] = (
            config["two_rock_fixture"]["rocks"][0]
            ["initial_position_world_m"])
        with self.assertRaisesRegex(ValueError, "r3_c_initial_separation"):
            validate_config(config, ROOT)

    def test_formal_preflight_report_and_artifacts_are_bound(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        model_manifest = json.loads(
            (REPORT_DIR / "generated-model-manifest.json")
            .read_text(encoding="utf-8"))
        bridge_manifest = json.loads(
            (REPORT_DIR / "generated-bridge-manifest.json")
            .read_text(encoding="utf-8"))
        self.assertEqual(
            report["status"], "r3_c_two_rock_bridge_preflight_passed")
        self.assertTrue(report["bridge_preflight_qualified"])
        self.assertFalse(report["physics_qualified"])
        self.assertFalse(report["training_qualified"])
        self.assertEqual(len(report["checks"]), 16)
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(
            report["implementation"]["runner_sha256"],
            sha256(SOURCE / "run_mixed_material_r3_c_bridge_preflight.py"))
        self.assertEqual(
            report["inputs"]["model_sha256"],
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
