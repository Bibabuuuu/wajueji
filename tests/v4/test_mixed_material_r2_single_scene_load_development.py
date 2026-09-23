import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

from mixed_material_r2_mechanism import sha256  # noqa: E402
from run_mixed_material_r2_single_scene_load_development import (  # noqa: E402
    _insert_phase,
    _validate_config,
)
from run_mixed_material_r2_single_scene_load_development_v2 import (  # noqa: E402
    _extend_lift,
    _validate_config as _validate_config_v2,
)


CONFIG = ROOT / "configs/v4/mixed-material-r2-single-scene-load-development-v1.json"
BASE_CONFIG = ROOT / "configs/v4/mixed-material-r2-controlled-intake-v1.json"
REPORT_ROOT = (
    ROOT / "reports/v4/mixed-material-r2-single-scene-load-development-2026-09-20-v1")
CONFIG_V2 = ROOT / "configs/v4/mixed-material-r2-single-scene-load-development-v2.json"
LIFT_CONFIG = ROOT / "configs/v4/mixed-material-r2-intake-lift-v1.json"
REPORT_ROOT_V2 = (
    ROOT / "reports/v4/mixed-material-r2-single-scene-load-development-2026-09-20-v2")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SingleSceneLoadDevelopmentConfigTests(unittest.TestCase):
    def test_only_one_geometry_selected_change_is_authorized(self):
        config = read_json(CONFIG)
        _validate_config(config)
        candidate = _insert_phase(read_json(BASE_CONFIG), config)
        self.assertEqual(candidate["selected_scenes"], [config["selected_scene"]])
        self.assertEqual(candidate["phases"][1],
                         config["trajectory_change"]["phase"])
        without_insert = copy.deepcopy(candidate["phases"])
        without_insert.pop(1)
        self.assertEqual(without_insert, read_json(BASE_CONFIG)["phases"])
        self.assertFalse(config["scope"]["trajectory_grid_search"])
        self.assertFalse(config["scope"]["second_scene_confirmation"])

    def test_rejects_command_threshold_and_scope_drift(self):
        cases = []
        command = read_json(CONFIG)
        command["trajectory_change"]["phase"]["command"][0] = -0.08
        cases.append(command)
        duration = read_json(CONFIG)
        duration["trajectory_change"]["phase"]["duration_s"] = 2.0
        cases.append(duration)
        threshold = read_json(CONFIG)
        threshold["acceptance"]["minimum_carried_rock_count"] = 2
        cases.append(threshold)
        grid = read_json(CONFIG)
        grid["scope"]["trajectory_grid_search"] = True
        cases.append(grid)
        for candidate in cases:
            with self.assertRaises(ValueError):
                _validate_config(candidate)


class SingleSceneLoadDevelopmentV2ConfigTests(unittest.TestCase):
    def test_only_lift_duration_is_extended(self):
        config = read_json(CONFIG_V2)
        _validate_config_v2(config)
        source = read_json(LIFT_CONFIG)
        candidate = _extend_lift(source, config)
        self.assertEqual(candidate["additional_phases"][0]["duration_s"], 2.0)
        candidate["additional_phases"][0]["duration_s"] = 1.5
        self.assertEqual(candidate, source)
        self.assertFalse(config["scope"]["trajectory_grid_search"])
        self.assertFalse(config["scope"]["second_scene_confirmation"])

    def test_rejects_lift_threshold_and_scope_drift(self):
        duration = read_json(CONFIG_V2)
        duration["lift_phase_extension"]["candidate_duration_s"] = 2.5
        threshold = read_json(CONFIG_V2)
        threshold["acceptance"]["minimum_carried_rock_count"] = 2
        grid = read_json(CONFIG_V2)
        grid["scope"]["trajectory_grid_search"] = True
        for candidate in (duration, threshold, grid):
            with self.assertRaises(ValueError):
                _validate_config_v2(candidate)


@unittest.skipUnless(REPORT_ROOT.is_dir(), "formal single-scene report pending")
class SingleSceneLoadDevelopmentFormalTests(unittest.TestCase):
    def test_report_is_hash_bound_and_scoped(self):
        report = read_json(REPORT_ROOT / "report.json")
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(report["scene_id"],
                         "ordinary-small-rock-dominant-seed-26091912")
        self.assertFalse(report["trajectory_grid_search"])
        self.assertFalse(report["second_scene_confirmation"])
        self.assertFalse(report["all_six_scene_qualification"])
        self.assertFalse(report["physics_qualified"])
        if report["status"] == "single_scene_load_development_candidate":
            self.assertGreaterEqual(report["candidate_carried_rock_count"], 3)
            self.assertGreater(
                report["candidate_preliminary_solid_volume_fill_rate"],
                report["baseline_preliminary_solid_volume_fill_rate"])


@unittest.skipUnless(REPORT_ROOT_V2.is_dir(), "formal v2 single-scene report pending")
class SingleSceneLoadDevelopmentV2FormalTests(unittest.TestCase):
    def test_v2_report_is_hash_bound_and_meets_frozen_goal(self):
        report = read_json(REPORT_ROOT_V2 / "report.json")
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG_V2))
        self.assertEqual(report["status"], "single_scene_load_development_candidate")
        self.assertEqual(report["candidate_carried_rock_count"], 3)
        self.assertEqual(report["candidate_carried_rock_ids"], [
            "r2-rock-007", "r2-rock-011", "r2-rock-019"])
        self.assertGreater(
            report["candidate_preliminary_solid_volume_fill_rate"],
            report["baseline_preliminary_solid_volume_fill_rate"])
        self.assertFalse(report["trajectory_grid_search"])
        self.assertFalse(report["second_scene_confirmation"])
        self.assertFalse(report["physics_qualified"])


if __name__ == "__main__":
    unittest.main()
