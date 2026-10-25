import unittest

import numpy as np

from planning.rear_lower_face import (
    observed_outward_profile,
    select_rear_lower_face,
)


def terrain(profile):
    height = np.full((30, 38), np.nan)
    count = np.zeros((30, 38), dtype=int)
    origin = np.asarray([3.5, -1.5])
    # The 0.60 m mouth centred on y=-0.05 includes both edge cells.
    for iy in range(11, 18):
        for x, z in profile.items():
            ix = int(np.floor((x - origin[0]) / 0.1))
            height[iy, ix] = z
            count[iy, ix] = 10
    return {
        "valid": True,
        "height_m": height,
        "point_count": count,
        "origin_xy_m": origin.tolist(),
        "resolution_m": 0.1,
    }


class RearLowerFaceTest(unittest.TestCase):
    def setUp(self):
        self.surface = terrain({
            5.65: 1.375,
            5.75: 1.201,
            5.85: 0.650,
            5.95: 0.160,
        })
        self.strip = {
            "entry_xy_m": [5.65, -0.05],
            "surface_z_m": 1.375,
        }

    def test_selects_observed_rear_face_near_forty_percent_height(self):
        selected = select_rear_lower_face(self.surface, self.strip)
        self.assertEqual(
            selected["source"], "rear_lower_face_from_observed_profile")
        self.assertAlmostEqual(selected["contact_xy_m"][0], 5.85, places=2)
        self.assertAlmostEqual(selected["contact_surface_z_m"], 0.650)
        self.assertAlmostEqual(
            selected["normalized_height_fraction"],
            (0.650 - 0.160) / (1.375 - 0.160))
        self.assertEqual(
            selected["prepoint_status"],
            "not_selected_requires_separate_clearance_plan")

    def test_profile_stops_before_unknown_space(self):
        profile = observed_outward_profile(self.surface, [5.65, -0.05])
        self.assertEqual(len(profile), 4)
        self.assertAlmostEqual(profile[-1]["surface_z_m"], 0.160)

    def test_rejects_flat_surface_without_a_rear_face(self):
        flat = terrain({
            5.65: 1.3,
            5.75: 1.3,
            5.85: 1.3,
            5.95: 1.3,
        })
        with self.assertRaisesRegex(ValueError, "drop_too_small"):
            select_rear_lower_face(flat, self.strip)

    def test_rejects_unknown_width_slice(self):
        broken = self.surface
        broken["height_m"][15, 23] = np.nan
        broken["point_count"][15, 23] = 0
        with self.assertRaisesRegex(ValueError, "profile_too_short"):
            select_rear_lower_face(broken, self.strip)


if __name__ == "__main__":
    unittest.main()
