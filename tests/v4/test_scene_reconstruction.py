import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "v4"))

from build_scene_reconstruction import (  # noqa: E402
    _build_scene,
    _load_prototypes,
    _sat_axis_gap,
    _surface_height,
    _validate_config,
    build,
    load_config,
    sha256,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "v4" / "scene-reconstruction-v1.json"


class ConfigurationValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_rejects_bad_schema_source_nonfinite_quaternion_and_open_region(self):
        cases = []
        invalid = copy.deepcopy(self.config)
        invalid["schema"] = "wrong"
        cases.append((invalid, "invalid_scene_reconstruction_schema"))
        invalid = copy.deepcopy(self.config)
        invalid["sources"]["synthetic_baseline"]["status"] = "guessed"
        cases.append((invalid, "unknown_source_status"))
        invalid = copy.deepcopy(self.config)
        invalid["surface"]["base_height_m"] = float("nan")
        cases.append((invalid, "non_finite_parameter"))
        invalid = copy.deepcopy(self.config)
        invalid["rocks"]["orientation_base_quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        cases.append((invalid, "orientation_base_quaternion_not_unit"))
        invalid = copy.deepcopy(self.config)
        invalid["region"]["polygon_xy_m"] = invalid["region"]["polygon_xy_m"][:-1]
        cases.append((invalid, "region_polygon_not_closed"))
        for candidate, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_config(candidate)

    def test_rejects_invalid_ratios_and_jam_leakage(self):
        invalid = copy.deepcopy(self.config)
        invalid["rocks"]["prototype_mix"]["angular_tetra"] = 0.3
        with self.assertRaisesRegex(ValueError, "invalid_prototype_mix"):
            _validate_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["scene_classes"][0]["diameter_bins"][1]["maximum_m"] = 0.3
        invalid["scene_classes"][0]["diameter_bins"][1]["minimum_m"] = 0.3
        with self.assertRaisesRegex(ValueError, "jam_rock_leaked_into_regular_scene"):
            _validate_config(invalid)


class SceneGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        cls.prototypes = _load_prototypes(cls.config, REPOSITORY_ROOT)

    def test_fixed_seed_is_identical_and_different_seed_changes_layout(self):
        scene_config = self.config["scene_classes"][0]
        first_manifest, first_metrics = _build_scene(
            self.config, scene_config, scene_config["seeds"][0], self.prototypes)
        second_manifest, second_metrics = _build_scene(
            copy.deepcopy(self.config), copy.deepcopy(scene_config),
            scene_config["seeds"][0], self.prototypes)
        changed_manifest, _ = _build_scene(
            self.config, scene_config, scene_config["seeds"][1], self.prototypes)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_metrics, second_metrics)
        first_positions = [item["position_world_m"] for item in first_manifest["rocks"]]
        changed_positions = [item["position_world_m"] for item in changed_manifest["rocks"]]
        self.assertNotEqual(first_positions, changed_positions)

    def test_mass_inertia_total_and_distribution_constraints_recompute(self):
        for scene_config in self.config["scene_classes"]:
            manifest, metrics = _build_scene(
                self.config, scene_config, scene_config["seeds"][0], self.prototypes)
            total_mass = 0.0
            for rock in manifest["rocks"]:
                prototype = self.prototypes[rock["prototype_id"]]
                scale = rock["equivalent_diameter_m"]
                expected_volume = prototype["unit_volume_m3"] * scale ** 3
                expected_mass = self.config["rocks"]["density_kg_m3"] * expected_volume
                expected_inertia = (self.config["rocks"]["density_kg_m3"]
                                    * prototype["unit_inertia_m5"] * scale ** 5)
                self.assertAlmostEqual(rock["volume_m3"], expected_volume, places=12)
                self.assertAlmostEqual(rock["mass_kg"], expected_mass, places=12)
                np.testing.assert_allclose(
                    rock["inertia_about_com_body_kg_m2"], expected_inertia,
                    rtol=0.0, atol=1e-12)
                total_mass += expected_mass
            self.assertAlmostEqual(metrics["total_mass_kg"], total_mass, places=11)
            self.assertTrue(metrics["all_admission_checks_passed"])
            if scene_config["scene_class"] != "jam-boundary":
                self.assertGreaterEqual(metrics["small_count_fraction"], 0.8)
                self.assertLessEqual(metrics["large_count"], 1)
                self.assertLessEqual(metrics["large_equivalent_volume_fraction"], 0.2)
                self.assertEqual(metrics["jam_boundary_count"], 0)
            else:
                self.assertGreaterEqual(metrics["jam_boundary_count"], 1)

    def test_surface_boundary_and_sat_checks_recompute(self):
        scene_config = self.config["scene_classes"][1]
        manifest, metrics = _build_scene(
            self.config, scene_config, scene_config["seeds"][0], self.prototypes)
        transformed = []
        for rock in manifest["rocks"]:
            prototype = self.prototypes[rock["prototype_id"]]
            from mixed_material_r0 import quaternion_matrix_wxyz
            rotation = quaternion_matrix_wxyz(rock["orientation_quaternion_wxyz"])
            vertices = (prototype["vertices"] * rock["equivalent_diameter_m"]
                        @ rotation.T + np.asarray(rock["position_world_m"]))
            clearances = [
                vertex[2] - _surface_height(
                    self.config["surface"], vertex[0], vertex[1])
                for vertex in vertices
            ]
            self.assertGreaterEqual(
                min(clearances) + 1e-10,
                self.config["placement"]["ground_clearance_m"])
            self.assertGreaterEqual(
                rock["placement_diagnostics"]["boundary_clearance_m"] + 1e-10,
                self.config["region"]["boundary_safety_distance_m"])
            transformed.append((rock, vertices))
        gaps = []
        for (first, first_vertices), (second, second_vertices) in __import__(
                "itertools").combinations(transformed, 2):
            intersects, gap = _sat_axis_gap(
                first_vertices, self.prototypes[first["prototype_id"]]["faces"],
                second_vertices, self.prototypes[second["prototype_id"]]["faces"])
            self.assertFalse(intersects)
            gaps.append(gap)
        self.assertAlmostEqual(
            min(gaps), metrics["minimum_sat_separating_axis_gap_m"], places=12)


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.first_output = cls.root / "first"
        cls.index = build(CONFIG_PATH, cls.first_output, REPOSITORY_ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_nine_isolated_scenes_and_hash_bindings(self):
        self.assertEqual(self.index["scene_count"], 9)
        self.assertTrue(self.index["all_scenes_passed"])
        for record in self.index["scenes"]:
            directory = self.first_output / record["scene_id"]
            expected = {
                "scene-manifest.json", "scene-metrics.json", "preview.svg",
                "report.json", "README.md",
            }
            self.assertEqual({item.name for item in directory.iterdir()}, expected)
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["scene_manifest_sha256"],
                             sha256(directory / "scene-manifest.json"))
            self.assertEqual(report["scene_metrics_sha256"],
                             sha256(directory / "scene-metrics.json"))
            self.assertEqual(report["preview_sha256"], sha256(directory / "preview.svg"))
            original = json.loads(
                (directory / "scene-manifest.json").read_text(encoding="utf-8"))
            original_hash = sha256(directory / "scene-manifest.json")
            original["rocks"] = original["rocks"][:-1]
            self.assertNotEqual(original_hash, __import__("hashlib").sha256(
                (json.dumps(original, indent=2, sort_keys=True, allow_nan=False)
                 + "\n").encode("utf-8")).hexdigest())

    def test_two_fresh_builds_are_byte_identical(self):
        second_output = self.root / "second"
        build(CONFIG_PATH, second_output, REPOSITORY_ROOT)
        first_files = sorted(
            path.relative_to(self.first_output) for path in self.first_output.rglob("*")
            if path.is_file())
        second_files = sorted(
            path.relative_to(second_output) for path in second_output.rglob("*")
            if path.is_file())
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            self.assertEqual((self.first_output / relative).read_bytes(),
                             (second_output / relative).read_bytes(), relative)

    def test_impossible_density_fails_without_formal_output(self):
        invalid = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        invalid["region"]["boundary_safety_distance_m"] = 1.6
        invalid_config = self.root / "impossible.json"
        invalid_config.write_text(
            json.dumps(invalid, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "scene_placement_failed"):
            build(invalid_config, output, REPOSITORY_ROOT)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
