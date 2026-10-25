import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from mixed_material_r2_mechanism import sha256  # noqa: E402
from run_mixed_material_r2_controlled_intake import (  # noqa: E402
    _point_in_polygon_with_tolerance,
    _validate_config,
)
from run_mixed_material_r2_intake_lift import (  # noqa: E402
    _validate_config as _validate_lift_config,
)


CONFIG_PATH = (
    ROOT / "configs" / "v4" / "mixed-material-r2-controlled-intake-v1.json")
REPORT_ROOT = (
    ROOT / "reports" / "v4" /
    "mixed-material-r2-controlled-intake-2026-09-20-v1")
LIFT_CONFIG_PATH = (
    ROOT / "configs" / "v4" / "mixed-material-r2-intake-lift-v1.json")
LIFT_REPORT_ROOT = (
    ROOT / "reports" / "v4" / "mixed-material-r2-intake-lift-2026-09-20-v2")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ControlledIntakeConfigTests(unittest.TestCase):
    def test_config_keeps_two_scene_diagnostic_scope(self):
        config = read_json(CONFIG_PATH)
        _validate_config(config)
        self.assertEqual(len(config["selected_scenes"]), 2)
        self.assertTrue(any(scene.startswith("ordinary-small-rock-dominant-")
                            for scene in config["selected_scenes"]))
        self.assertTrue(any(scene.startswith("sparse-large-rock-")
                            for scene in config["selected_scenes"]))
        self.assertTrue(config["acceptance"][
            "force_and_torque_are_diagnostic_not_capacity_thresholds"])
        self.assertFalse(config["scope"]["all_six_scene_qualification"])
        self.assertFalse(config["scope"]["fill_rate"])
        self.assertFalse(config["scope"]["physics_qualified"])

    def test_rejects_scope_expansion_external_wrench_and_command_overrun(self):
        cases = []
        expanded = read_json(CONFIG_PATH)
        expanded["scope"]["fill_rate"] = True
        cases.append(expanded)
        wrench = read_json(CONFIG_PATH)
        wrench["integration"]["external_wrench_count"] = 1
        cases.append(wrench)
        command = read_json(CONFIG_PATH)
        command["phases"][1]["command"][1] = -0.03
        cases.append(command)
        force_claim = read_json(CONFIG_PATH)
        force_claim["acceptance"][
            "force_and_torque_are_diagnostic_not_capacity_thresholds"] = False
        cases.append(force_claim)
        for candidate in cases:
            with self.assertRaises(ValueError):
                _validate_config(candidate)

    def test_polygon_tolerance_is_bounded(self):
        polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        self.assertTrue(_point_in_polygon_with_tolerance([0.5, 0.5], polygon, 0.0))
        self.assertTrue(_point_in_polygon_with_tolerance([-0.0005, 0.5],
                                                         polygon, 0.001))
        self.assertFalse(_point_in_polygon_with_tolerance([-0.002, 0.5],
                                                          polygon, 0.001))


class IntakeLiftConfigTests(unittest.TestCase):
    def test_short_lift_is_bounded_and_does_not_claim_long_hold(self):
        config = read_json(LIFT_CONFIG_PATH)
        _validate_lift_config(config)
        self.assertEqual([item["name"] for item in config["additional_phases"]],
                         ["short_lift", "hold_after_lift"])
        self.assertEqual(config["additional_phases"][0]["command"],
                         [0.0, 0.02, 0.0, 0.0])
        self.assertFalse(config["scope"]["ten_second_hold_qualification"])
        self.assertFalse(config["scope"]["all_six_scene_qualification"])
        self.assertFalse(config["scope"]["fill_rate"])

    def test_rejects_fake_force_threshold_and_long_hold_claim(self):
        force_claim = read_json(LIFT_CONFIG_PATH)
        force_claim["acceptance"][
            "force_and_torque_are_diagnostic_not_capacity_thresholds"] = False
        long_hold = read_json(LIFT_CONFIG_PATH)
        long_hold["scope"]["ten_second_hold_qualification"] = True
        for candidate in (force_claim, long_hold):
            with self.assertRaises(ValueError):
                _validate_lift_config(candidate)


@unittest.skipUnless(REPORT_ROOT.is_dir(), "formal controlled-intake report pending")
class ControlledIntakeFormalReportTests(unittest.TestCase):
    def test_formal_report_is_bound_and_does_not_overclaim(self):
        config = read_json(CONFIG_PATH)
        report = read_json(REPORT_ROOT / "report.json")
        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG_PATH))
        self.assertFalse(report["all_six_scene_qualification"])
        self.assertFalse(report["lift_hold_qualified"])
        self.assertFalse(report["dump_qualified"])
        self.assertFalse(report["mpm_coupling_qualified"])
        self.assertFalse(report["fill_rate_qualified"])
        self.assertFalse(report["physics_qualified"])
        self.assertEqual([scene["scene_id"] for scene in report["scenes"]],
                         config["selected_scenes"])


@unittest.skipUnless(LIFT_REPORT_ROOT.is_dir(), "formal intake-lift report pending")
class IntakeLiftFormalReportTests(unittest.TestCase):
    def test_formal_lift_report_is_bound_and_scoped(self):
        report = read_json(LIFT_REPORT_ROOT / "report.json")
        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["status"], "short_lift_hold_needs_review")
        self.assertEqual(report["inputs"]["config_sha256"],
                         sha256(LIFT_CONFIG_PATH))
        self.assertFalse(report["ten_second_hold_qualified"])
        self.assertFalse(report["all_six_scene_qualification"])
        self.assertFalse(report["fill_rate_qualified"])
        self.assertFalse(report["physics_qualified"])
        for scene in report["scenes"]:
            scene_report = read_json(
                LIFT_REPORT_ROOT / scene["scene_id"] / "report.json")
            metrics = scene_report["metrics"]
            self.assertEqual(scene_report["status"],
                             "short_lift_hold_scene_needs_review")
            self.assertFalse(metrics["checks"][
                "same_intake_end_rock_retained_and_lifted"])
            carried = set(metrics[
                "lifted_carried_from_intake_end_full_rock_ids"])
            self.assertLessEqual(carried,
                                 set(metrics["intake_end_full_rock_ids"]))
            self.assertLessEqual(carried,
                                 set(metrics["final_retained_full_rock_ids"]))


if __name__ == "__main__":
    unittest.main()
