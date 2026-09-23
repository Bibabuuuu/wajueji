import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_multirock_candidate_r3_c import sha256  # noqa: E402
from mixed_material_r3_c_convergence_contract import validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-c-convergence-v1.json"
REPORT_DIR = (
    ROOT / "reports" / "v4"
    / "mixed-material-r3-c-convergence-2026-09-20-v1")


class MixedMaterialR3CConvergenceTest(unittest.TestCase):
    def test_config_freezes_repeat_refinement_feedback_and_scope(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_config(config, ROOT)
        self.assertEqual(config["probe"]["coarse_output_dt_s"], 1.0 / 120.0)
        self.assertEqual(config["probe"]["refined_output_dt_s"], 1.0 / 240.0)
        self.assertEqual(config["probe"]["coarse_mujoco_internal_substeps"], 40)
        self.assertEqual(config["probe"]["refined_mujoco_internal_substeps"], 20)
        self.assertTrue(config["scope"]["reconstructed_repeat_qualification"])
        self.assertTrue(config["scope"]["dt_refinement_qualification"])
        self.assertTrue(config["scope"]["feedback_causality_qualification"])
        self.assertFalse(config["scope"]["complete_in_process_reset"])
        self.assertFalse(config["scope"]["official_envelope"])
        self.assertFalse(config["scope"]["training"])

    def test_rejects_hash_dt_threshold_and_scope_drift(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["shared_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen_input_sha256_mismatch"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["probe"]["refined_output_dt_s"] = 1.0 / 120.0
        with self.assertRaisesRegex(ValueError, "convergence_probe_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["probe"]["maximum_dt_impulse_relative_difference"] = 0.0
        with self.assertRaisesRegex(ValueError, "convergence_threshold_contract"):
            validate_config(config, ROOT)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["official_envelope"] = True
        with self.assertRaisesRegex(ValueError, "convergence_scope_expanded"):
            validate_config(config, ROOT)

    def test_formal_report_preserves_failure_and_passing_subgates(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "r3_c_convergence_probe_failed")
        self.assertFalse(report["convergence_qualified"])
        self.assertFalse(report["r3_c_physics_qualified"])
        self.assertFalse(report["training_qualified"])
        self.assertEqual(len(report["checks"]), 8)
        self.assertTrue(report["checks"]["coarse_reconstructed_repeat_bounded"])
        self.assertTrue(report["checks"]["coarse_repeat_is_byte_exact"])
        self.assertTrue(report["checks"]["feedback_changes_mechanism"])
        self.assertTrue(report["checks"]["all_episode_health_checks_pass"])
        self.assertFalse(report["checks"]["refined_reconstructed_repeat_bounded"])
        self.assertFalse(report["checks"]["dt_refinement_bounded"])
        comparison = report["comparison"]
        self.assertGreaterEqual(
            comparison["feedback"]["maximum_final_mechanism_qpos_difference_m"],
            1.0e-6)
        self.assertGreater(
            comparison["refined_repeat"]["maximum_final_state_difference"],
            1.0e-4)
        self.assertGreater(
            comparison["dt_refinement"]["maximum_final_state_difference"],
            0.005)
        self.assertGreater(
            comparison["dt_refinement"]["impulse_relative_differences"]
            ["rock_mpm_ns.r1-rock-000"], 0.2)
        self.assertFalse(
            report["reset_claims"]["complete_in_process_mixed_reset"])

    def test_formal_report_is_bound_to_inputs_runner_model_and_bridge(self):
        report = json.loads(
            (REPORT_DIR / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(
            report["implementation"]["runner_sha256"],
            sha256(SOURCE / "run_mixed_material_r3_c_convergence_probe.py"))
        self.assertEqual(
            report["inputs"]["derived_model_sha256"],
            sha256(REPORT_DIR / "generated-model.xml"))
        self.assertEqual(
            report["inputs"]["bridge_source_sha256"],
            sha256(REPORT_DIR / "generated-bridge.cc"))
        self.assertEqual(
            report["inputs"]["shared_report_sha256"],
            sha256(ROOT / "reports" / "v4"
                   / "mixed-material-r3-c-shared-four-round-2026-09-20-v1"
                   / "report.json"))


if __name__ == "__main__":
    unittest.main()
