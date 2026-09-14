import sys
import unittest
from pathlib import Path
import numpy as np
from planning.scene_clearance import SceneClearance

EVIDENCE = Path(__file__).resolve().parents[1]/'evidence/manual-force-tracking-2026-09-13'
sys.path.insert(0, str(EVIDENCE))
from manual_scene import bucket_clearance
from audit_mesh_distance import tests as primitive_tests


class SceneNarrowPhaseTest(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]]])
        self.geom = {'mesh':'bucket', 'body':'bucket', 'body_local_vertices':self.a[0],
                     'faces':np.array([[0,1,2]])}
        self.bodies = {'bucket':np.eye(4)}
        self.truck = np.array([[[.8,.8,1.],[1.8,.8,1.],[.8,1.8,1.]]])

    def scene(self, triangles, callback=None, geometry=None):
        return SceneClearance(geometry or [self.geom], triangles, [[3,3,0],[4,4,2]],
                              bucket_narrow_phase=callback)

    def test_triangle_primitives(self):
        self.assertEqual(primitive_tests(), 7)

    def test_default_broad_rejection_remains(self):
        with self.assertRaisesRegex(ValueError, 'broad_overlap'):
            self.scene(self.truck).check(self.bodies, 'MANUAL')
        self.scene(self.truck, bucket_clearance).check(self.bodies, 'MANUAL')

    def test_actual_intersection_and_twenty_mm_margin_reject(self):
        for b in [self.a, self.a+[0,0,.019]]:
            with self.assertRaisesRegex(ValueError, 'surface_clearance'):
                self.scene(b, bucket_clearance).check(self.bodies, 'MANUAL')

    def test_other_mechanism_checks_still_run_after_bucket_refinement(self):
        arm = {'mesh':'arm','body':'arm','body_local_vertices':self.truck[0],
               'faces':np.array([[0,1,2]])}
        with self.assertRaisesRegex(ValueError, 'mechanism_truck'):
            self.scene(self.truck, bucket_clearance, [self.geom, arm]).check(
                {**self.bodies, 'arm':np.eye(4)}, 'MANUAL')

if __name__ == '__main__':
    unittest.main()
