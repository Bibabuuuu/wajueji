import unittest

import numpy as np

from planning.rear_lower_prepoint import (
    ground_completed_terrain,
    mechanism_and_truck_clearance,
    proxy_surface_samples,
    sampled_surface_clearance,
    oriented_terrain_separation,
)


class RearLowerPrepointTest(unittest.TestCase):
    def test_sloped_link_box_refines_world_aabb_false_collision(self):
        angle = np.pi / 4
        rotation = np.array([[np.cos(angle), 0, -np.sin(angle)],
                             [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        local = np.array([[x, y, z] for x in (0., 4.) for y in (-.05, .05)
                          for z in (-.05, .05)])
        corners = local @ rotation.T + [0., 0., 1.]
        terrain = dict(origin_xy_m=[2., -.5], resolution_m=1., height_m=np.array([[1.5]]))
        self.assertLess(corners[:, 2].min() - 1.5, 0.)
        self.assertGreater(oriented_terrain_separation(corners, rotation, terrain), .02)
        terrain['height_m'][:] = 4.
        self.assertEqual(oriented_terrain_separation(corners, rotation, terrain), 0.)

    def test_ground_completion_preserves_material_and_fills_unknown(self):
        source = {
            "valid": True,
            "height_m": np.asarray([[np.nan, 0.7], [-0.1, 0.2]]),
            "point_count": np.asarray([[0, 3], [2, 4]]),
        }
        completed = ground_completed_terrain(source)
        np.testing.assert_allclose(
            completed["height_m"], [[0.0, 0.7], [0.0, 0.2]])
        np.testing.assert_array_equal(
            completed["point_count"], [[1, 3], [2, 4]])
        self.assertTrue(
            completed["ground_completion"]["material_observation_unchanged"])

    def test_proxy_sampler_covers_box_face_interiors(self):
        proxy = {
            "shell_boxes": [{
                "half_size_m": [0.05, 0.05, 0.05],
                "position_bucket_m": [0.0, 0.0, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }],
            "tooth_boxes": [],
            "side_plate_meshes": [],
        }
        samples = proxy_surface_samples(proxy, maximum_spacing_m=0.025)
        self.assertGreater(len(samples), 8)
        self.assertTrue(np.any(np.all(np.isclose(
            samples, [0.05, 0.0, 0.0]), axis=1)))

    def test_clearance_uses_cell_max_at_each_proxy_sample(self):
        terrain = {
            "valid": True,
            "origin_xy_m": [0.0, 0.0],
            "resolution_m": 1.0,
            "height_m": np.asarray([[0.2, 0.8]]),
            "point_count": np.asarray([[1, 1]]),
        }
        result = sampled_surface_clearance(np.asarray([
            [0.5, 0.5, 0.7],
            [1.5, 0.5, 1.1],
        ]), terrain)
        self.assertAlmostEqual(result["clearance_m"], 0.3)
        self.assertAlmostEqual(result["cell_surface_z_m"], 0.8)

    def test_clearance_rejects_unknown_or_outside_domain(self):
        terrain = {
            "valid": True,
            "origin_xy_m": [0.0, 0.0],
            "resolution_m": 1.0,
            "height_m": np.asarray([[np.nan]]),
            "point_count": np.asarray([[0]]),
        }
        with self.assertRaisesRegex(ValueError, "clearance_contract"):
            sampled_surface_clearance(np.asarray([[0.5, 0.5, 1.0]]), terrain)
        complete = ground_completed_terrain(terrain)
        with self.assertRaisesRegex(ValueError, "outside_observed_roi"):
            sampled_surface_clearance(np.asarray([[1.5, 0.5, 1.0]]), complete)

    def test_mechanism_and_truck_clearance_excludes_fixed_ground_base(self):
        geometries = [{
            "mesh": "caterpillar", "body": "base",
            "body_local_vertices": np.asarray([[0.0, 0.0, 0.0]]),
        }, {
            "mesh": "arm", "body": "arm",
            "body_local_vertices": np.asarray([
                [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]),
        }]
        bodies = {"base": np.eye(4), "arm": np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])}
        terrain = {
            "valid": True,
            "origin_xy_m": [-1.0, -1.0],
            "resolution_m": 1.0,
            "height_m": np.asarray([[np.nan, np.nan], [np.nan, 0.1]]),
            "point_count": np.asarray([[0, 0], [0, 1]]),
        }
        truck = np.asarray([[2.0, 2.0, 0.0], [3.0, 3.0, 1.0]])
        result = mechanism_and_truck_clearance(
            geometries, bodies, terrain, truck)
        self.assertAlmostEqual(
            result["mechanism_terrain_clearance_m"], 0.4)
        self.assertGreater(
            result["static_truck_aabb_separation_m"], 2.0)


if __name__ == "__main__":
    unittest.main()
