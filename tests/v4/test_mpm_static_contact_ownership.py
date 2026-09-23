import unittest
import numpy as np
import newton
import warp as wp
from newton.solvers import SolverImplicitMPM
from mpm_static_contact_ownership import StaticContactOwnership


class StaticOwnershipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        builder = newton.ModelBuilder()
        # Reproduce the actual ID mismatch: 48 bodies, 65 dynamic shapes,
        # five static truck shapes and ground, but only 49 colliders.
        for i in range(48):
            body = builder.add_body(mass=1.0)
            for _ in range(18 if i == 47 else 1):
                builder.add_shape_box(body, hx=.03, hy=.03, hz=.03)
        cls.shapes = {}
        for i in range(5):
            cls.shapes[f"truck{i}"] = builder.add_shape_box(
                -1, xform=wp.transform(wp.vec3(2*i, 0, 2), wp.quat_identity()),
                hx=.3, hy=.3, hz=.03)
        cls.shapes["ground"] = builder.add_ground_plane()
        cls.model = builder.finalize(device="cpu")
        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        sand.add_particle(pos=(0, 0, 3), vel=(0, 0, 0), mass=1, radius=.01)
        cls.solver = SolverImplicitMPM(sand.finalize(device="cpu"),
                                      config=SolverImplicitMPM.Config())
        cls.solver.setup_collider(model=cls.model)

    def test_all_five_truck_faces_and_ground_are_counted(self):
        ownership = StaticContactOwnership(self.solver, self.model, self.shapes)
        self.assertEqual(ownership.manifest["collider_count"], 49)
        self.assertEqual(list(self.shapes.values()), list(range(65, 71)))
        positions = np.asarray([[2*i, 0, 2.031] for i in range(5)] +
                               [[20, 20, .001], [0, 0, 2.031], [0, 0, 2.031]])
        counts, unknown = ownership.count(positions, np.array([0]*6 + [48, 0]),
                                         np.array([True]*7 + [False]), self.model.body_q)
        self.assertEqual(counts, dict.fromkeys(self.shapes, 1))
        self.assertEqual(unknown, 0)
        np.testing.assert_array_equal(self.solver.collider_body_index.numpy(),
                                      np.arange(-1, 48))

    def test_missing_and_duplicate_static_owners_are_rejected(self):
        missing = dict(self.shapes)
        del missing["ground"]
        for mapping in (missing, {**self.shapes, "duplicate": self.shapes["ground"]}):
            with self.assertRaisesRegex(ValueError, "incomplete_or_duplicate"):
                StaticContactOwnership(self.solver, self.model, mapping)

    def test_query_failure_is_unmapped_not_silently_ground(self):
        ownership = StaticContactOwnership(self.solver, self.model, self.shapes)
        counts, unknown = ownership.count(np.array([[0, 0, 10000]]), np.array([0]),
                                         np.array([True]), self.model.body_q)
        self.assertEqual(sum(counts.values()), 0)
        self.assertEqual(unknown, 1)


if __name__ == "__main__":
    unittest.main()
