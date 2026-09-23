import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_formal_scene_model_adapter import (  # noqa: E402
    load_and_validate_config as load_adapter_config,
)
from mixed_material_r0 import normalized_prototypes  # noqa: E402
from prepare_formal_scene_geometry import (  # noqa: E402
    load_and_validate_config,
    prepare,
)


CONFIG = ROOT / "configs" / "v4" / "formal-scene-geometry-preparation-v1.json"
ADAPTER_CONFIG = ROOT / "configs" / "v4" / "formal-scene-model-adapter-v1.json"


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


class FormalSceneGeometryPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.adapter_path, cls.fill_path = load_and_validate_config(
            CONFIG, ROOT)
        cls.adapter, _ = load_adapter_config(ADAPTER_CONFIG, ROOT)

    def test_config_binds_both_stages_and_forbids_scope_expansion(self):
        self.assertTrue(self.adapter_path.is_file())
        self.assertTrue(self.fill_path.is_file())
        self.assertTrue(self.config["scope"]["atomic_geometry_bundle"])
        self.assertFalse(self.config["scope"]["physics_step"])
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        value = copy.deepcopy(base)
        value["frozen_mpm_fill_config"]["sha256"] = "0" * 64
        attacks.append((value, "binding_mismatch"))
        value = copy.deepcopy(base)
        value["output_layout"]["model"] = "surprise.xml"
        attacks.append((value, "output_layout_drift"))
        value = copy.deepcopy(base)
        value["scope"]["physics_step"] = True
        attacks.append((value, "scope_expanded"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, error) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        ValueError, error):
                    load_and_validate_config(path, ROOT)

    def test_bundle_is_deterministic_cross_bound_and_atomic(self):
        scene = make_scene(self.adapter)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_path = root / "scene.json"
            scene_path.write_text(
                json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            first = root / "first"
            second = root / "second"
            first_report = prepare(CONFIG, scene_path, ROOT, first)
            second_report = prepare(CONFIG, scene_path, ROOT, second)
            self.assertEqual(first_report, second_report)
            expected = [
                "actual-n-scene.xml",
                "actual-n-model-manifest.json",
                "geometry-bundle-report.json",
                "mpm-fill/layered-bank.npz",
                "mpm-fill/metrics.json",
                "mpm-fill/report.json",
            ]
            for relative in expected:
                self.assertEqual((first / relative).read_bytes(),
                                 (second / relative).read_bytes())
            self.assertTrue(all(first_report["checks"].values()))
            self.assertEqual(first_report["rock_count"], 1)
            self.assertFalse(first_report["time_zero_executed"])
            self.assertFalse(first_report["physics_step_executed"])
            self.assertFalse(first_report["formal_scene_qualified"])
            self.assertFalse(any(first.parent.glob(f".{first.name}.tmp-*")))

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
                prepare(CONFIG, scene_path, ROOT, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
