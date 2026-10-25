import copy
import unittest

import numpy as np

from mixed_material_r1 import _convex_prism, outward_side_plate_faces


class BucketSideWindingTests(unittest.TestCase):
    def plate(self, negative=False, reverse_polygon=False):
        polygon = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        if reverse_polygon:
            polygon.reverse()
        return _convex_prism(polygon, -.3 if negative else .3,
                             -.35 if negative else .35, "test_side")

    def test_both_extrusions_and_polygon_orders_have_outward_normals(self):
        for negative in (False, True):
            for reverse in (False, True):
                side = self.plate(negative, reverse)
                v = np.asarray(side["vertices_bucket_m"])
                f = np.asarray(side["faces"])
                triangles = v[f]
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                # Independent geometric oracle: the centre is behind every
                # outward face, an offset along each face normal is outside.
                signed = np.einsum("ij,ij->i", normals,
                                   triangles.mean(axis=1) - v.mean(axis=0))
                self.assertTrue(np.all(signed > 0))
                self.assertAlmostEqual(outward_side_plate_faces(side)[1][
                    "runtime_signed_volume_m3"], .05)

    def test_legacy_inward_plate_is_fixed_without_mutating_source(self):
        side = self.plate(True)
        side["faces"] = np.asarray(side["faces"])[:, ::-1].tolist()
        frozen = copy.deepcopy(side)
        fixed, audit = outward_side_plate_faces(side)
        self.assertEqual(side, frozen)
        self.assertTrue(audit["flipped"])
        self.assertLess(audit["source_signed_volume_m3"], 0)
        self.assertNotEqual(audit["source_mesh_sha256"], audit["runtime_mesh_sha256"])
        np.testing.assert_array_equal(np.sort(fixed, axis=1),
                                      np.sort(side["faces"], axis=1))
        side["faces"] = fixed.tolist()
        second, second_audit = outward_side_plate_faces(side)
        np.testing.assert_array_equal(second, fixed)
        self.assertFalse(second_audit["flipped"])
        self.assertEqual(second_audit["source_mesh_sha256"],
                         second_audit["runtime_mesh_sha256"])

    def test_open_or_locally_reversed_plate_is_rejected(self):
        for corruption in ("open", "reversed"):
            side = self.plate()
            if corruption == "open":
                side["faces"].pop()
            else:
                side["faces"][0].reverse()
            with self.assertRaisesRegex(ValueError, "not_closed_consistently"):
                outward_side_plate_faces(side)

    def test_flat_plate_is_rejected(self):
        side = self.plate()
        for vertex in side["vertices_bucket_m"]:
            vertex[1] = .3
        with self.assertRaisesRegex(ValueError, "degenerate_side_plate_volume"):
            outward_side_plate_faces(side)


if __name__ == "__main__":
    unittest.main()
