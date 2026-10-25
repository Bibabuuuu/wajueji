import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_formal_scene_model_adapter import (  # noqa: E402
    build_model,
    load_and_validate_config,
    validate_scene_manifest,
)
from mixed_material_r0 import normalized_prototypes  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "formal-scene-model-adapter-v1.json"


def make_scene(config):
    prototypes = normalized_prototypes([
        "angular_block", "angular_octa", "angular_tetra"])
    specifications = [
        ("source-small-A", "angular_block", "small_0p16_0p24", 0.20,
         [4.2, -0.4, 0.4]),
        ("source-medium-B", "angular_octa", "medium_0p24_0p32", 0.28,
         [4.8, 0.0, 0.3]),
        ("source-large-C", "angular_tetra", "large_0p32_0p40", 0.36,
         [5.4, 0.4, 0.25]),
    ]
    records = []
    density = config["rocks"]["density_kg_m3"]
    for source_id, prototype_id, size_class, diameter, position in specifications:
        prototype = prototypes[prototype_id]
        volume = float(prototype["unit_volume_m3"]) * diameter ** 3
        inertia = density * prototype["unit_inertia_m5"] * diameter ** 5
        records.append({
            "rock_id": source_id,
            "prototype_id": prototype_id,
            "size_class": size_class,
            "equivalent_diameter_m": diameter,
            "density_kg_m3": density,
            "volume_m3": volume,
            "mass_kg": density * volume,
            "inertia_about_com_body_kg_m2": inertia.tolist(),
            "position_world_m": position,
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "asset_path": f"{config['rocks']['asset_root']}/{prototype_id}.obj",
            "asset_sha256": config["rocks"]["assets"][prototype_id],
        })
    return {
        "schema": config["scene_contract"]["schema"],
        "status": config["scene_contract"]["status"],
        "scenario_name": config["scene_contract"]["scenario_name"],
        "coordinate_frame": "world",
        "rock_count": len(records),
        "checks": {name: True for name in
                   config["scene_contract"]["required_true_checks"]},
        "rocks": records,
    }


class FormalSceneModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.capacity_path = load_and_validate_config(CONFIG, ROOT)

    def test_config_binds_dynamic_capacity_and_visual_stable_contract(self):
        self.assertTrue(self.capacity_path.is_file())
        self.assertEqual(self.config["scene_contract"]["maximum_rock_count"], 1024)
        self.assertEqual(
            self.config["scene_contract"]["schema"],
            "excavation_fullscale_visual_stable_scene_manifest_v1")
        self.assertFalse(self.config["scope"]["physics_step"])
        self.assertFalse(self.config["scope"]["formal_scene_qualified"])

    def test_rejects_capacity_asset_limit_and_scope_config_attacks(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        value = copy.deepcopy(base)
        value["frozen_capacity_config"]["sha256"] = "0" * 64
        attacks.append((value, "capacity_binding_mismatch"))
        value = copy.deepcopy(base)
        value["rocks"]["assets"]["angular_block"] = "0" * 64
        attacks.append((value, "asset_mismatch"))
        value = copy.deepcopy(base)
        value["scene_contract"]["maximum_rock_count"] = 887
        attacks.append((value, "scene_contract_drift"))
        value = copy.deepcopy(base)
        value["scope"]["physics_step"] = True
        attacks.append((value, "scope_expanded"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, error) in enumerate(attacks):
                path = Path(temporary) / f"config-attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        ValueError, error):
                    load_and_validate_config(path, ROOT)

    def test_accepts_recomputed_three_class_fixture_and_maps_runtime_ids(self):
        scene = make_scene(self.config)
        rocks = validate_scene_manifest(self.config, scene, ROOT)
        self.assertEqual([rock["runtime_rock_id"] for rock in rocks],
                         ["r1-rock-000", "r1-rock-001", "r1-rock-002"])
        self.assertEqual([rock["source_rock_id"] for rock in rocks],
                         ["source-small-A", "source-medium-B", "source-large-C"])

    def test_rejects_identity_check_id_asset_mass_pose_and_velocity_attacks(self):
        base = make_scene(self.config)
        attacks = []
        value = copy.deepcopy(base)
        value["schema"] = "old_exact_volume_manifest"
        attacks.append((value, "identity_mismatch"))
        value = copy.deepcopy(base)
        value["checks"]["stable_rigid_skeleton"] = False
        attacks.append((value, "required_checks_missing"))
        value = copy.deepcopy(base)
        value["rocks"][1]["rock_id"] = value["rocks"][0]["rock_id"]
        attacks.append((value, "duplicate_or_invalid_source_id"))
        value = copy.deepcopy(base)
        value["rocks"][0]["asset_sha256"] = "0" * 64
        attacks.append((value, "asset_binding"))
        value = copy.deepcopy(base)
        value["rocks"][0]["mass_kg"] += 1.0
        attacks.append((value, "mass_inertia_recompute"))
        value = copy.deepcopy(base)
        value["rocks"][0]["orientation_quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        attacks.append((value, "quaternion_not_unit"))
        value = copy.deepcopy(base)
        value["rocks"][0]["linear_velocity_world_mps"][0] = 0.1
        attacks.append((value, "frozen_velocity_nonzero"))
        for candidate, error in attacks:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                validate_scene_manifest(self.config, candidate, ROOT)

    def test_model_generation_is_deterministic_and_preserves_source_mapping(self):
        scene = make_scene(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            scene_path.write_text(
                json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            first = root / "first.xml"
            second = root / "second.xml"
            first_manifest = build_model(CONFIG, scene_path, ROOT, first)
            second_manifest = build_model(CONFIG, scene_path, ROOT, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["rock_count"], 3)
            self.assertEqual(first_manifest["expected_nq"], 40)
            self.assertEqual(first_manifest["expected_nv"], 36)
            self.assertEqual(first_manifest["rock_id_mapping"], [
                {"source_rock_id": "source-small-A", "runtime_rock_id": "r1-rock-000"},
                {"source_rock_id": "source-medium-B", "runtime_rock_id": "r1-rock-001"},
                {"source_rock_id": "source-large-C", "runtime_rock_id": "r1-rock-002"},
            ])
            tree = ET.parse(first)
            worldbody = tree.getroot().find("worldbody")
            bodies = [body for body in worldbody.findall("body")
                      if (body.get("name") or "").startswith("r1-rock-")]
            self.assertEqual([body.get("name") for body in bodies],
                             ["r1-rock-000", "r1-rock-001", "r1-rock-002"])
            meshes = [mesh for mesh in tree.getroot().find("asset").findall("mesh")
                      if (mesh.get("name") or "").startswith("formal_rock_mesh_")]
            self.assertEqual(len(meshes), 3)
            self.assertTrue(first_manifest["time_zero_ready_for_separate_runner"])
            self.assertFalse(first_manifest["time_zero_executed"])
            self.assertFalse(first_manifest["physics_step_executed"])
            self.assertFalse(first_manifest["formal_scene_qualified"])


if __name__ == "__main__":
    unittest.main()
