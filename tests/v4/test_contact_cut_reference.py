import unittest
import numpy as np
from contact_cut_reference import GroundSoilCutReference
from contact_geometric_feedback import bounded_velocity


class CutReferenceTest(unittest.TestCase):
    def reference(self, height=1.4, shift=0.):
        ref=GroundSoilCutReference.__new__(GroundSoilCutReference)
        xyz=np.meshgrid(np.linspace(-1,1,61),np.linspace(-.5,.5,31),np.linspace(0,height,41),indexing='ij')
        ref.points=np.stack(xyz,axis=-1).reshape(-1,3);ref.points[:,2]+=shift
        ref.root=np.array([0.,0.,-.1]);ref.width=.6;ref.vertical_pitch=.2
        return ref

    def plan(self, ref):return ref.plan(np.eye(4),np.array([.5,0.,1.]),0.,np.array([1.,0.]),.5,.03)

    def test_lower_soil_changes_depth_instead_of_fixed_drop(self):
        first=self.plan(self.reference());lower=self.plan(self.reference(height=1.1))
        self.assertGreater(first['tip_target_z_m']-lower['tip_target_z_m'],.25)

    def test_vertical_translation_equivariance(self):
        a=self.plan(self.reference());b=self.plan(self.reference(shift=.3))
        self.assertAlmostEqual(b['tip_target_z_m']-a['tip_target_z_m'],.3)

    def test_yaw_rotation_equivariance(self):
        ref=self.reference();base=self.plan(ref)
        angle=.7;c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        ref.points=ref.points@rot.T;pose=np.eye(4);pose[:3,:3]=rot
        result=ref.plan(pose,rot@np.array([.5,0,1.]),0.,rot[:2,0],.5,.03)
        self.assertAlmostEqual(result['tip_target_z_m'],base['tip_target_z_m'])
        self.assertTrue(np.allclose(result['root_target_xy_m'],rot[:2,:2]@base['root_target_xy_m']))

    def test_full_scoop_budget_requires_explicit_mode(self):
        from run_contact_geometric_probe import probe_budget
        with self.assertRaises(ValueError):probe_budget(2400,True)
        self.assertEqual(probe_budget(2400,False,True),900)
        with self.assertRaises(ValueError):probe_budget(2401,False,True)

    def test_sparse_air_outlier_does_not_define_surface(self):
        ref=self.reference();a=self.plan(ref);ref.points=np.vstack([ref.points,[.45,0,100.]])
        b=self.plan(ref);self.assertAlmostEqual(a['surface_z_m'],b['surface_z_m'])

    def test_unknown_width_rejected(self):
        ref=self.reference();ref.points=ref.points[ref.points[:,1]>.1]
        with self.assertRaisesRegex(ValueError,'unknown'):self.plan(ref)

    def test_soil_too_shallow_rejected(self):
        with self.assertRaisesRegex(ValueError,'ground_infeasible'):self.plan(self.reference(height=.4))

    def test_pitch_barrier_keeps_other_axes_moving(self):
        action=bounded_velocity(np.eye(4),[-.5,0,-.4,.8],[0,0,1,0],-1.,pitch_upper=.05)
        self.assertLessEqual(action[3],.0500001);self.assertLess(action[0],-.49);self.assertLess(action[2],-.39)

if __name__=='__main__':unittest.main()
