import unittest
import numpy as np
from contact_phase_geometry import SolidBucketGap, ground_connected_mask, triangle_distances


class GeometryTest(unittest.TestCase):
    def cube(self):
        # Synthetic convex solid, independent of frozen excavation dimensions.
        v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],float)
        f=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
           [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        proxy=dict(shell_boxes=[dict(half_size_m=[1,1,1],position_bucket_m=[100,0,0],quaternion_wxyz=[1,0,0,0])],
                   tooth_boxes=[],side_plate_meshes=[dict(vertices_bucket_m=v,faces=f)])
        return SolidBucketGap(proxy)

    def test_triangle_face_edge_and_vertex(self):
        tri=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        p=np.array([[.2,.2,.003],[.5,-.003,0],[-.003,-.004,0],[.5,.5,0]])
        np.testing.assert_allclose(triangle_distances(p,tri),[.003,.003,.005,0],atol=1e-12)

    def test_solid_interior_and_side_edge(self):
        obs=self.cube()
        for point,expected in (([0,0,0],0),([1.003,0,0],.003),([1.003,1.004,0],.005),([101.003,0,0],.003)):
            self.assertAlmostEqual(obs.minimum_gap([point],np.eye(4)),expected,12)
        self.assertIsNone(obs.minimum_gap(np.empty((0,3)),np.eye(4)))

    def test_rotation_translation_invariance(self):
        obs=self.cube();pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[4,5,6]
        q=np.array([[1.003,0,0]])
        self.assertAlmostEqual(obs.minimum_gap(q@pose[:3,:3].T+pose[:3,3],pose),.003,12)
        pose[0,0]=2
        with self.assertRaises(ValueError):obs.minimum_gap(q,pose)

    def test_convex_triangle_distance_matches_analytic_cube(self):
        obs=self.cube()
        rng=np.random.default_rng(146)
        for point in rng.uniform(-2,2,size=(100,3)):
            expected=float(np.linalg.norm(np.maximum(np.abs(point)-1,0)))
            self.assertAlmostEqual(obs.minimum_gap([point],np.eye(4)),expected,11)

    def test_ground_connectivity_excludes_detached_payload(self):
        q=np.array([[0,0,z] for z in np.arange(.01,.5,.02)]+[[2,0,2]],float)
        mask=ground_connected_mask(q)
        self.assertTrue(mask[:-1].all());self.assertFalse(mask[-1])
        self.assertFalse(ground_connected_mask(q+np.array([0,0,3])).any())
        self.assertIsNone(self.cube().observe(q,np.eye(4))['remaining_soil_gap_m'])


if __name__=='__main__':unittest.main()
