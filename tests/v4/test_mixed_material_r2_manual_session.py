import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from mixed_material_r2_mechanism import sha256  # noqa: E402
from run_mixed_material_r2_manual_session import (  # noqa: E402
    _score,
    _validate_config,
)


CONFIG = ROOT / "configs/v4/mixed-material-r2-manual-session-v1.json"
SMOKE_REPORT_ROOT = (
    ROOT / "reports/v4/mixed-material-r2-manual-session-2026-09-20-smoke-v1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ManualSessionConfigTests(unittest.TestCase):
    def test_control_mapping_and_scope_are_frozen(self):
        config = read_json(CONFIG)
        _validate_config(config)
        self.assertEqual(config["selected_scene"],
                         "ordinary-small-rock-dominant-seed-26091912")
        self.assertEqual(config["manual_control"]["axis_order"],
                         ["rotation", "boom", "arm", "bucket"])
        self.assertEqual(config["manual_control"]["positive_keys"],
                         ["J", "I", "U", "M"])
        self.assertEqual(config["manual_control"]["negative_keys"],
                         ["L", "K", "O", "N"])
        self.assertTrue(config["manual_control"]["release_means_zero_hold"])
        self.assertFalse(config["scope"]["training"])
        self.assertFalse(config["scope"]["scripted_trajectory_optimization"])
        self.assertFalse(config["scope"]["second_scene_confirmation"])

    def test_rejects_scene_threshold_training_and_control_drift(self):
        cases = []
        scene = read_json(CONFIG)
        scene["selected_scene"] = "sparse-large-rock-seed-26091925"
        cases.append(scene)
        threshold = read_json(CONFIG)
        threshold["acceptance"]["minimum_carried_rock_count"] = 2
        cases.append(threshold)
        training = read_json(CONFIG)
        training["scope"]["training"] = True
        cases.append(training)
        control = read_json(CONFIG)
        control["manual_control"]["maximum_command"][1] = 0.03
        cases.append(control)
        for candidate in cases:
            with self.assertRaises(ValueError):
                _validate_config(candidate)


class ManualEffectiveCarryTests(unittest.TestCase):
    def test_off_ground_near_bucket_rocks_need_not_fit_strict_envelope(self):
        inventory = read_json(
            "/home/qunwang/codex-work/compact-bank-2026-09-11/"
            "inventory-contract.json")
        plan = {"rocks": [{
            "rock_id": f"source-{index:03d}",
            "volume_m3": 0.002,
            "mass_kg": 5.2,
        } for index in range(20)]}
        lift_start = {f"r2-rock-{index:03d}": [0.0, 0.0, 0.0]
                      for index in range(20)}
        samples = []
        for step in range(120):
            rocks = []
            for index in range(20):
                carried = index < 3
                rocks.append({
                    "rock_id": f"r2-rock-{index:03d}",
                    "position_world_m": [0.0, 0.0, 0.04 if carried else 0.0],
                    "center_bucket_m": ([0.4, 0.0, 0.05] if carried
                                        else [3.0, 0.0, 3.0]),
                    "speed_m_s": 0.0,
                    "ground_contact_substeps": 0 if carried else 1,
                    "bucket_contact_substeps": 0,
                    "full_envelope_inside_diagnostic": False,
                })
            samples.append({
                "time_s": (step + 1) / 120,
                "action_command": [0.0, 0.0, 0.0, 0.0],
                "rocks": rocks,
            })
        acceptance = read_json(CONFIG)["acceptance"]
        metrics = _score(
            samples, lift_start, 0.0, plan, inventory, acceptance,
            0.255164729950989, 0.0, 0, None)
        self.assertTrue(metrics["candidate"])
        self.assertEqual(metrics["carried_rock_count"], 3)
        self.assertTrue(all(
            not rock["final_full_envelope_inside_diagnostic"]
            for rock in metrics["per_rock"][:3]))
        self.assertGreater(metrics["preliminary_solid_volume_fill_rate"],
                           acceptance["scripted_baseline_fill_rate"])


@unittest.skipUnless(SMOKE_REPORT_ROOT.is_dir(), "formal manual smoke pending")
class ManualSessionSmokeReportTests(unittest.TestCase):
    def test_smoke_runs_viewer_bridge_and_recorder_without_claiming_success(self):
        report = read_json(SMOKE_REPORT_ROOT / "report.json")
        self.assertEqual(report["status"], "manual_session_smoke_completed")
        self.assertIsNone(report["runtime_error"])
        self.assertGreater(report["output_steps"], 0)
        self.assertGreater(report["demonstration_sample_count"], 0)
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertIn("lift_start", report["markers"])
        self.assertIn("evaluation_hold_start", report["markers"])
        self.assertFalse(report["demonstration_training_eligible"])
        self.assertFalse(report["manual_feasibility_demonstrated"])
        self.assertFalse(report["training_qualified"])
        self.assertFalse(report["physics_qualified"])


if __name__ == "__main__":
    unittest.main()
