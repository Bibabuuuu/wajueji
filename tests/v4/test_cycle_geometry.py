import unittest
import numpy as np
from layered_47_cycle_geometry import triangles_intersect_box, oriented_box_separation
from layered_47_cycle_selection import rank_workspace_candidates


class CycleGeometryTest(unittest.TestCase):
    def test_triangle_crosses_box_even_with_all_vertices_outside(self):
        triangle=np.array([[[-2.,0,0],[2,0,0],[0,2,0]]])
        self.assertTrue(triangles_intersect_box(triangle,np.full(3,-.5),np.full(3,.5),0))

    def test_overlapping_bounding_boxes_can_have_separated_triangles(self):
        triangle=np.array([[[.4,2,0],[2,.4,0],[2,2,0]]])
        self.assertFalse(triangles_intersect_box(triangle,np.full(3,-.5),np.full(3,.5),0))

    def test_margin_is_retained(self):
        triangle=np.array([[[.51,-.3,-.3],[.51,.3,-.3],[.51,0,.3]]])
        self.assertFalse(triangles_intersect_box(triangle,np.full(3,-.5),np.full(3,.5),0))
        self.assertTrue(triangles_intersect_box(triangle,np.full(3,-.5),np.full(3,.5),.02))

    def test_future_strips_are_lateral_and_nonoverlapping(self):
        height=np.full((30,38),.1);count=np.full_like(height,10,dtype=int)
        for iy in range(4,26):
            height[iy,2:22]=1.375
            height[iy,22]=1.201;height[iy,23]=.650;height[iy,24]=.160
        terrain=dict(valid=True,height_m=height,point_count=count,origin_xy_m=[3.5,-1.5],resolution_m=.1)
        candidates=rank_workspace_candidates(terrain,3)
        winner=candidates[0]
        self.assertEqual(winner['future_strip_count'],2)
        points=np.array([winner['entry_xy_m'],*winner['future_entry_xy_m']])
        radial=points[0]/np.linalg.norm(points[0]);lateral=np.array([-radial[1],radial[0]])
        for i in range(3):
            for j in range(i):self.assertGreaterEqual(abs((points[i]-points[j])@lateral),.6)
        # Re-observation with the positive-y stock removed must change the pair.
        changed=height.copy();changed[15:]=.1
        after=rank_workspace_candidates(dict(terrain,height_m=changed),2)
        self.assertLess(after[0]['entry_xy_m'][1],0)
        self.assertFalse(np.array_equal(changed,height))
