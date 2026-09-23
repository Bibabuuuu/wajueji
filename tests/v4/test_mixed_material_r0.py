import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "v4"))

from mixed_material_r0 import (  # noqa: E402
    build_gauge,
    build_scene_manifest,
    mass_properties,
    matrix_quaternion_wxyz,
    normalized_prototypes,
    proper_axis_rotations,
    quaternion_matrix_wxyz,
)


PROTOTYPE_IDS = (
    "angular_tetra", "angular_bipyramid", "angular_octa", "angular_block")


def config():
    return {
        "rocks": {
            "density_kg_m3": 2600.0,
            "prototype_ids": list(PROTOTYPE_IDS),
            "visual_collision_policy": "same_normalized_closed_convex_mesh",
            "fracture_enabled": False,
        },
        "gauge": {"equivalent_diameters_m": [0.27, 0.33, 0.59949]},
        "scenes": [{
            "scene_id": "test-scene", "seed": 42, "size_bucket": "main_training",
            "equivalent_diameters_m": [0.12, 0.18, 0.24, 0.27],
            "grid_origin_xy_m": [4.0, -1.0], "grid_columns": 2,
            "grid_spacing_m": 0.8, "ground_clearance_m": 0.02,
        }],
    }


class RockPrototypeTests(unittest.TestCase):
    def test_prototypes_are_closed_convex_com_centered_unit_diameter(self):
        prototypes = normalized_prototypes(PROTOTYPE_IDS)
        for prototype in prototypes.values():
            self.assertTrue(prototype["closed_two_manifold"])
            self.assertTrue(prototype["convex"])
            self.assertAlmostEqual(prototype["equivalent_diameter_m"], 1.0, places=12)
            properties = mass_properties(prototype["vertices"], prototype["faces"])
            np.testing.assert_allclose(properties["center"], 0.0, atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(
                properties["inertia_unit_density"]) > 0.0))

    def test_mass_and_inertia_follow_scale_laws(self):
        prototype = normalized_prototypes(PROTOTYPE_IDS)["angular_octa"]
        density, scale = 2600.0, 0.27
        mass = density * prototype["unit_volume_m3"] * scale ** 3
        inertia = density * prototype["unit_inertia_m5"] * scale ** 5
        self.assertAlmostEqual(mass, density * math.pi / 6.0 * scale ** 3)
        self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.0))


class GaugeTests(unittest.TestCase):
    def test_axis_orientation_set_is_complete_and_proper(self):
        rotations = proper_axis_rotations()
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(value.ravel()) for value in rotations}), 24)
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            np.testing.assert_allclose(
                quaternion_matrix_wxyz(matrix_quaternion_wxyz(rotation)),
                rotation, atol=1e-12)

    def test_gauge_keeps_geometry_only_scope_and_pose_results(self):
        prototypes = normalized_prototypes(PROTOTYPE_IDS)
        bucket = {
            "mouth_endpoints_xz_m": [[0.021459, -0.157745],
                                     [0.213659, -1.023365]],
            "inner_clear_width_m": 0.59949,
            "mouth_chord_m": 0.8867010907853898,
        }
        gauge = build_gauge(bucket, config(), prototypes)
        self.assertEqual(len(gauge["prototype_pose_results"]), 4 * 3 * 24)
        self.assertTrue(gauge["sphere_gauges"][0]["passes"])
        self.assertTrue(gauge["sphere_gauges"][1]["passes"])
        self.assertTrue(gauge["sphere_gauges"][2]["passes"])
        self.assertIn("does not prove", gauge["scope_limit"])


class SceneManifestTests(unittest.TestCase):
    def test_manifest_is_deterministic_recomputable_and_conservatively_clear(self):
        prototypes = normalized_prototypes(PROTOTYPE_IDS)
        first = build_scene_manifest(config(), prototypes)
        second = build_scene_manifest(copy.deepcopy(config()), prototypes)
        self.assertEqual(first, second)
        scene = first["scenes"][0]
        self.assertEqual(scene["initial_overlap_check"],
                         "passed_conservative_bounding_spheres")
        self.assertGreater(scene["minimum_bounding_sphere_pair_clearance_m"], 0.0)
        self.assertLessEqual(scene["equivalent_diameter_d95_m"], 0.27)
        for rock in scene["rocks"]:
            expected = (rock["density_kg_m3"] * math.pi / 6.0
                        * rock["equivalent_diameter_m"] ** 3)
            self.assertAlmostEqual(rock["mass_kg"], expected)


if __name__ == "__main__":
    unittest.main()
