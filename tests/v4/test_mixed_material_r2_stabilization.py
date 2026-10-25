import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src" / "v4"))

from mixed_material_r2_mechanism import build_candidate, sha256, write_json  # noqa: E402
from run_mixed_material_r2_stabilization import _validate_config  # noqa: E402
from select_sparse_large_stable_candidates import (  # noqa: E402
    _validate_config as _validate_selection_config,
)


CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r2-stabilization.json"
R2_CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r2.json"
PREFLIGHT_ROOT = ROOT / "reports" / "v4" / "scene-assembly-preflight-2026-09-19-v1"
STABLE_ROOT = ROOT / "reports" / "v4" / "mixed-material-r2-stabilization-2026-09-20-v1"
FIRST_SCENE = "ordinary-small-rock-dominant-seed-26091911"
SELECTION_CONFIG_PATH = (
    ROOT / "configs" / "v4" / "sparse-large-stable-admission-v1.json")
SELECTION_ROOT = (
    ROOT / "reports" / "v4" / "sparse-large-stable-admission-2026-09-20-v1")
R2_V2_CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r2-v2.json"
STABILIZATION_V2_CONFIG_PATH = (
    ROOT / "configs" / "v4" / "mixed-material-r2-stabilization-v2.json")
R2_V2_ROOT = ROOT / "reports" / "v4" / "mixed-material-r2-2026-09-20-v2"
STABLE_V2_ROOT = (
    ROOT / "reports" / "v4" / "mixed-material-r2-stabilization-2026-09-20-v2")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class R2StabilizationContractTests(unittest.TestCase):
    def test_stabilization_contract_and_formal_review_status(self):
        config = read_json(CONFIG_PATH)
        _validate_config(config)
        report = read_json(STABLE_ROOT / "report.json")
        self.assertEqual(report["status"], "r2_stable_initial_states_need_review")
        self.assertEqual(report["scene_count"], 6)
        self.assertFalse(report["source_scene_manifests_modified"])
        self.assertFalse(report["source_assembly_plans_modified"])
        self.assertFalse(report["rock_count_or_size_modified"])
        statuses = [record["status"] for record in report["scenes"]]
        self.assertEqual(statuses.count("stable_initial_state_verified"), 4)
        self.assertEqual(statuses.count("stable_initial_state_needs_review"), 2)

    def test_rejects_relaxed_boundary_or_excavation_scope(self):
        config = read_json(CONFIG_PATH)
        relaxed = copy.deepcopy(config)
        relaxed["spatial_review"][
            "require_original_boundary_safety_retention"] = False
        expanded = copy.deepcopy(config)
        expanded["scope"]["controlled_excavation"] = True
        for candidate in (relaxed, expanded):
            with self.assertRaises(ValueError):
                _validate_config(candidate)

    def test_stable_pose_override_is_bound_and_deterministic(self):
        stable_path = STABLE_ROOT / FIRST_SCENE / "stable-initial-state.json"
        stable = read_json(stable_path)
        plan_path = PREFLIGHT_ROOT / FIRST_SCENE / "assembly-plan.json"
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first" / "model.xml"
            second = Path(temporary) / "second" / "model.xml"
            first_manifest = build_candidate(
                R2_CONFIG_PATH, plan_path, first, ROOT, stable_path)
            second_manifest = build_candidate(
                R2_CONFIG_PATH, plan_path, second, ROOT, stable_path)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["initial_pose_source"],
                             "stable_initial_state_candidate")
            self.assertEqual(first_manifest["initial_pose_override_sha256"],
                             sha256(stable_path))
            body = next(element for element in ET.parse(first).iter("body")
                        if element.get("name") == "r2-rock-000")
            expected = stable["rocks"][0]["position_world_m"]
            actual = [float(value) for value in body.get("pos").split()]
            self.assertEqual(actual, expected)

    def test_corrupt_stable_identity_is_rejected(self):
        stable = read_json(STABLE_ROOT / FIRST_SCENE / "stable-initial-state.json")
        stable["rocks"][0]["source_rock_id"] = "wrong-rock"
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            corrupt = temporary / "stable.json"
            write_json(corrupt, stable)
            output = temporary / "model.xml"
            with self.assertRaisesRegex(
                    ValueError, "stable_initial_state_identity_mismatch:0"):
                build_candidate(
                    R2_CONFIG_PATH,
                    PREFLIGHT_ROOT / FIRST_SCENE / "assembly-plan.json",
                    output, ROOT, corrupt)
            self.assertFalse(output.exists())


class SparseReplacementAndFinalV2Tests(unittest.TestCase):
    def test_fixed_sequence_preserves_rejection_and_selects_exactly_two(self):
        config = read_json(SELECTION_CONFIG_PATH)
        _validate_selection_config(config)
        report = read_json(SELECTION_ROOT / "report.json")
        self.assertEqual(report["status"], "sparse_large_replacements_selected")
        self.assertTrue(report["rejected_candidates_preserved"])
        self.assertEqual(report["final_sparse_large_scene_count"], 3)
        self.assertEqual(report["final_total_first_round_scene_count"], 6)
        self.assertEqual(report["selected_scene_ids"], [
            "sparse-large-rock-seed-26091925",
            "sparse-large-rock-seed-26091926",
        ])
        self.assertEqual(
            [(item["scene_id"], item["status"])
             for item in report["candidates"]],
            [
                ("sparse-large-rock-seed-26091924", "rejected"),
                ("sparse-large-rock-seed-26091925", "selected"),
                ("sparse-large-rock-seed-26091926", "selected"),
            ],
        )
        self.assertLess(report["candidates"][0]["minimum_boundary_clearance_m"],
                        0.08)
        self.assertTrue(all(
            item["minimum_boundary_clearance_m"] >= 0.08
            for item in report["candidates"][1:]))

    def test_final_v2_chain_has_six_verified_stable_scenes(self):
        r2_config = read_json(R2_V2_CONFIG_PATH)
        stabilization_config = read_json(STABILIZATION_V2_CONFIG_PATH)
        _validate_config(stabilization_config)
        expected_scenes = [
            "ordinary-small-rock-dominant-seed-26091911",
            "ordinary-small-rock-dominant-seed-26091912",
            "ordinary-small-rock-dominant-seed-26091913",
            "sparse-large-rock-seed-26091922",
            "sparse-large-rock-seed-26091925",
            "sparse-large-rock-seed-26091926",
        ]
        self.assertEqual(r2_config["first_admission"]["scene_ids"],
                         expected_scenes)
        self.assertEqual(stabilization_config["source_r2_config"]["sha256"],
                         sha256(R2_V2_CONFIG_PATH))
        self.assertEqual(
            stabilization_config["baseline_model_root"]["load_report_sha256"],
            sha256(R2_V2_ROOT / "report.json"))
        self.assertEqual(
            stabilization_config["baseline_model_root"][
                "bridge_init_report_sha256"],
            sha256(R2_V2_ROOT / "bridge-init-report.json"))
        self.assertEqual(
            stabilization_config["baseline_model_root"][
                "settling_report_sha256"],
            sha256(R2_V2_ROOT / "settling-report.json"))
        report = read_json(STABLE_V2_ROOT / "report.json")
        self.assertEqual(report["status"],
                         "r2_six_scene_stable_initial_states_verified")
        self.assertEqual([item["scene_id"] for item in report["scenes"]],
                         expected_scenes)
        self.assertTrue(all(
            item["status"] == "stable_initial_state_verified"
            and item["minimum_boundary_clearance_m"] >= 0.08
            and item["rebaked_maximum_displacement_m"] <= 0.005
            for item in report["scenes"]))


if __name__ == "__main__":
    unittest.main()
