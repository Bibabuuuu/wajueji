import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_formal_scene_mpm_fill import build, load_and_validate_config  # noqa: E402
from build_formal_scene_model_adapter import (  # noqa: E402
    load_and_validate_config as load_adapter_config,
)
from mixed_material_r0 import normalized_prototypes  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "formal-scene-mpm-fill-v1.json"


def make_scene(adapter):
    prototype_id = "angular_block"
    prototype = normalized_prototypes([prototype_id])[prototype_id]
    diameter = 0.20
    density = adapter["rocks"]["density_kg_m3"]
    volume = float(prototype["unit_volume_m3"]) * diameter ** 3
    inertia = density * prototype["unit_inertia_m5"] * diameter ** 5
    return {
        "schema": adapter["scene_contract"]["schema"],
        "status": adapter["scene_contract"]["status"],
        "scenario_name": adapter["scene_contract"]["scenario_name"],
        "coordinate_frame": "world",
        "rock_count": 1,
        "checks": {name: True for name in
                   adapter["scene_contract"]["required_true_checks"]},
        "rocks": [{
            "rock_id": "stable-source-rock-A",
            "prototype_id": prototype_id,
            "size_class": "small_0p16_0p24",
            "equivalent_diameter_m": diameter,
            "density_kg_m3": density,
            "volume_m3": volume,
            "mass_kg": density * volume,
            "inertia_about_com_body_kg_m2": inertia.tolist(),
            "position_world_m": [4.5, 0.0, 0.5],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "asset_path": f"{adapter['rocks']['asset_root']}/{prototype_id}.obj",
            "asset_sha256": adapter["rocks"]["assets"][prototype_id],
        }],
    }


class FormalSceneMpmFillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.adapter_path = load_and_validate_config(CONFIG, ROOT)
        cls.adapter, _ = load_adapter_config(cls.adapter_path, ROOT)

    def test_config_is_carve_only_and_volume_is_advisory(self):
        self.assertTrue(self.adapter_path.is_file())
        self.assertEqual(
            self.config["carve_policy"],
            "remove_particle_center_inside_closed_convex_rock_only")
        self.assertTrue(self.config["reference_only"][
            "volume_or_fraction_error_is_not_a_standalone_failure"])
        self.assertFalse(self.config["scope"]["physics_step"])
        self.assertFalse(self.config["scope"]["physics_qualified"])

    def test_rejects_adapter_bank_policy_and_scope_attacks(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        value = copy.deepcopy(base)
        value["frozen_adapter_config"]["sha256"] = "0" * 64
        attacks.append((value, "adapter_binding_mismatch"))
        value = copy.deepcopy(base)
        value["frozen_bank"]["q_count"] = 1
        attacks.append((value, "bank_contract_drift"))
        value = copy.deepcopy(base)
        value["carve_policy"] = "remove_extra_particles_to_hit_target_volume"
        attacks.append((value, "carve_policy_drift"))
        value = copy.deepcopy(base)
        value["scope"]["mpm_or_gpu_run"] = True
        attacks.append((value, "scope_expanded"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, error) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        ValueError, error):
                    load_and_validate_config(path, ROOT)

    def test_carve_is_deterministic_partitioned_and_leaves_no_center_overlap(self):
        scene = make_scene(self.adapter)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            scene_path.write_text(
                json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            first = root / "first"
            second = root / "second"
            first_report = build(CONFIG, scene_path, ROOT, first)
            second_report = build(CONFIG, scene_path, ROOT, second)
            self.assertEqual(first_report, second_report)
            for name in ("layered-bank.npz", "metrics.json", "report.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["retained_particle_count"] + metrics["removed_particle_count"],
                metrics["original_particle_count"])
            self.assertGreater(metrics["removed_particle_count"], 0)
            self.assertEqual(
                metrics["retained_particle_center_inside_rock_count"], 0)
            self.assertTrue(all(metrics["checks"].values()))
            with np.load(first / "layered-bank.npz", allow_pickle=False) as archive:
                self.assertEqual(len(archive["q"]), metrics["retained_particle_count"])
                self.assertEqual(len(archive["removed_index"]),
                                 metrics["removed_particle_count"])
            self.assertFalse(first_report["physics_step_executed"])
            self.assertFalse(first_report["formal_scene_qualified"])

    def test_existing_output_is_rejected_without_overwrite(self):
        scene = make_scene(self.adapter)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output_dir_exists"):
                build(CONFIG, scene_path, ROOT, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
