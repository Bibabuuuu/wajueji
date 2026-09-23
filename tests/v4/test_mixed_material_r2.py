import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from mixed_material_r2_mechanism import (  # noqa: E402
    _validate_config,
    build_candidate,
    sha256,
    write_json,
)
from build_mixed_rigid_bridge_r2 import render_bridge  # noqa: E402
from run_mixed_material_r2_load import build_all  # noqa: E402
from run_mixed_material_r2_bridge_init import build_bridge_init  # noqa: E402
from run_mixed_material_r2_settling import run_settling  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r2.json"
PREFLIGHT_ROOT = (
    ROOT / "reports" / "v4" / "scene-assembly-preflight-2026-09-19-v1"
)
FIRST_SCENE = "ordinary-small-rock-dominant-seed-26091911"
JAM_SCENE = "jam-boundary-seed-26091931"
SOURCE_MODEL = (
    ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
    / "mechanism-auto8" / "mechanism.xml"
)
MUJOCO_ROOT = Path(
    "/home/qunwang/codex-work/excavator-workspace-20260907/bundle"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class R2CapacityDecisionTests(unittest.TestCase):
    def test_one_bridge_capacity_twenty_and_first_scope_are_explicit(self):
        config = read_json(CONFIG_PATH)
        _validate_config(config)
        self.assertEqual(config["capacity"]["maximum_rock_count"], 20)
        self.assertFalse(config["capacity"]["placeholder_rocks_allowed"])
        self.assertFalse(config["capacity"]["runtime_add_remove_allowed"])
        self.assertEqual(len(config["first_admission"]["scene_ids"]), 6)
        self.assertEqual(config["first_admission"]["required_rock_count"], 20)
        self.assertEqual(config["first_admission"]["deferred_scene_class"],
                         "jam-boundary")
        self.assertEqual(config["first_admission"]["deferred_scene_count"], 3)

    def test_rejects_capacity_split_placeholder_and_physics_scope_drift(self):
        config = read_json(CONFIG_PATH)
        cases = []
        invalid = copy.deepcopy(config)
        invalid["capacity"]["maximum_rock_count"] = 13
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid["capacity"]["placeholder_rocks_allowed"] = True
        cases.append(invalid)
        invalid = copy.deepcopy(config)
        invalid["scope"]["time_stepping"] = True
        cases.append(invalid)
        for candidate in cases:
            with self.assertRaises(ValueError):
                _validate_config(candidate)


class R2CandidateAssemblyTests(unittest.TestCase):
    def test_twenty_real_rocks_build_with_stable_names_and_addresses(self):
        source_before = sha256(SOURCE_MODEL)
        plan_path = PREFLIGHT_ROOT / FIRST_SCENE / "assembly-plan.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.xml"
            manifest = build_candidate(CONFIG_PATH, plan_path, output, ROOT)
            root = ET.parse(output).getroot()
            rocks = [body for body in root.iter("body")
                     if body.get("name", "").startswith("r2-rock-")]
            proxies = [geom for geom in root.iter("geom")
                       if geom.get("name", "").startswith("r2_bucket_")]
            self.assertEqual(len(rocks), 20)
            self.assertEqual(len(proxies), 18)
            terrains = [geom for geom in root.iter("geom")
                        if geom.get("name") == "r2_surface_terrain"]
            self.assertEqual(len(terrains), 1)
            self.assertEqual(terrains[0].get("type"), "hfield")
            self.assertEqual(terrains[0].get("contype"), "1024")
            self.assertEqual(terrains[0].get("conaffinity"), "256")
            heightfield_path = output.parent / "surface-heightfield.hfield"
            self.assertTrue(heightfield_path.is_file())
            self.assertEqual(manifest["surface_support"]["asset_sha256"],
                             sha256(heightfield_path))
            self.assertLessEqual(
                manifest["surface_support"]["maximum_interpolation_error_m"],
                0.00015)
            self.assertEqual(manifest["expected_candidate_contract"], {
                "nq": 159, "nv": 138, "nu": 12, "nplugin": 0,
                "rock_count": 20,
            })
            self.assertEqual(manifest["placeholder_rock_count"], 0)
            self.assertFalse(manifest["physics_qualified"])
            for slot, record in enumerate(manifest["rocks"]):
                self.assertEqual(record["body_name"], f"r2-rock-{slot:03d}")
                self.assertEqual(record["qpos_address"], 19 + 7 * slot)
                self.assertEqual(record["qvel_address"], 18 + 6 * slot)
        self.assertEqual(sha256(SOURCE_MODEL), source_before)

    def test_two_fresh_models_are_byte_identical(self):
        plan_path = PREFLIGHT_ROOT / FIRST_SCENE / "assembly-plan.json"
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first" / "model.xml"
            second = Path(temporary) / "second" / "model.xml"
            first_manifest = build_candidate(CONFIG_PATH, plan_path, first, ROOT)
            second_manifest = build_candidate(CONFIG_PATH, plan_path, second, ROOT)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (first.parent / "surface-heightfield.hfield").read_bytes(),
                (second.parent / "surface-heightfield.hfield").read_bytes())

    def test_jam_scene_and_corrupt_binding_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            jam_output = temporary / "jam.xml"
            with self.assertRaisesRegex(ValueError, "scene_not_in_first_admission"):
                build_candidate(
                    CONFIG_PATH, PREFLIGHT_ROOT / JAM_SCENE / "assembly-plan.json",
                    jam_output, ROOT)
            self.assertFalse(jam_output.exists())

            plan = read_json(PREFLIGHT_ROOT / FIRST_SCENE / "assembly-plan.json")
            plan["rocks"][1]["full_scene_binding"]["qpos_address"] += 1
            corrupt_plan = temporary / "corrupt-plan.json"
            write_json(corrupt_plan, plan)
            corrupt_output = temporary / "corrupt.xml"
            with self.assertRaisesRegex(ValueError, "invalid_full_scene_binding:1"):
                build_candidate(CONFIG_PATH, corrupt_plan, corrupt_output, ROOT)
            self.assertFalse(corrupt_output.exists())


class R2BridgeSourceTests(unittest.TestCase):
    def test_bridge_is_derived_deterministically_without_modifying_r1(self):
        base = ROOT / "src" / "v4" / "mixed_rigid_bridge.cc"
        before = sha256(base)
        first, first_manifest = render_bridge(CONFIG_PATH, ROOT)
        second, second_manifest = render_bridge(CONFIG_PATH, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(sha256(base), before)
        self.assertIn("rock_count > 20", first)
        self.assertIn('"r2-rock-%03d"', first)
        self.assertIn('"r2_rock_free_%03d"', first)
        self.assertIn("excavation_mixed_rigid_bridge_state_v2", first)
        self.assertIn("runtime_add_remove_allowed", first)
        self.assertIn('StartsWith(other_name, "r2_surface_")', first)
        self.assertNotIn('"r1_', first)
        self.assertNotIn("r1-rock", first)
        self.assertTrue(first_manifest["actual_count_from_loaded_model"])
        self.assertFalse(first_manifest["placeholder_rocks_allowed"])


class R2LoadOnlyPipelineTests(unittest.TestCase):
    @unittest.skipUnless(
        (MUJOCO_ROOT / "mujoco-custom/include/mujoco/mujoco.h").is_file() and
        (MUJOCO_ROOT / "lib/libmujoco.so.2.3.8").is_file(),
        "frozen MuJoCo 2.3.8 bundle unavailable",
    )
    def test_all_six_models_load_without_claiming_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_root = Path(temporary) / "report"
            report = build_all(
                CONFIG_PATH, model_root, ROOT, "g++",
                MUJOCO_ROOT / "mujoco-custom/include",
                MUJOCO_ROOT / "lib",
            )
            self.assertEqual(report["status"], "r2_six_scene_load_only_passed")
            self.assertEqual(report["scene_count"], 6)
            self.assertEqual(report["total_loaded_rock_records"], 120)
            self.assertEqual(report["bridge_implementation_count_planned"], 1)
            self.assertFalse(report["placeholder_rocks_used"])
            self.assertFalse(report["runtime_add_remove_used"])
            self.assertFalse(report["physics_qualified"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                len(list((Path(temporary) / "report").glob("*/model.xml"))), 6)
            bridge_report = build_bridge_init(
                CONFIG_PATH, model_root, model_root, ROOT, "g++",
                MUJOCO_ROOT / "mujoco-custom/include",
                MUJOCO_ROOT / "lib",
                ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
                / "mechanism-auto8",
                ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
                / "mechanism-auto8" / "excavator_simulator",
            )
            self.assertEqual(
                bridge_report["status"], "r2_six_scene_bridge_init_passed")
            self.assertEqual(bridge_report["total_initialized_rock_records"], 120)
            self.assertEqual(bridge_report["bridge_source_count"], 1)
            self.assertFalse(bridge_report["time_advanced"])
            self.assertFalse(bridge_report["mj_step_called"])
            self.assertFalse(bridge_report["physics_qualified"])
            self.assertTrue(all(bridge_report["checks"].values()))
            settling_report = run_settling(
                CONFIG_PATH, model_root, model_root, ROOT, "g++",
                MUJOCO_ROOT / "mujoco-custom/include",
                MUJOCO_ROOT / "lib",
                ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
                / "mechanism-auto8",
                ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
                / "mechanism-auto8" / "excavator_simulator",
            )
            self.assertEqual(
                settling_report["status"], "r2_six_scene_settling_smoke_passed")
            self.assertEqual(settling_report["total_preserved_rock_identities"], 120)
            self.assertEqual(settling_report["total_internal_substeps"], 14400)
            self.assertFalse(settling_report["physics_qualified"])
            self.assertTrue(all(settling_report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
