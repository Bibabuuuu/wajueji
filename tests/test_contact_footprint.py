import unittest
import numpy as np
from planning.contact_footprint import cut_opportunity, target_mouth_markers


def terrain(height):
    return dict(valid=True,height_m=height,point_count=np.ones_like(height,int),
                origin_xy_m=[3.5,-1.5],resolution_m=.1)


class ContactFootprintTest(unittest.TestCase):
    def setUp(self):
        self.contact=dict(contact_xy_m=[5.7,0.],contact_surface_z_m=1.)

    def test_narrow_high_ridge_loses_to_intact_full_width(self):
        ridge=np.full((30,38),.3);ridge[14:16]=3.
        intact=np.full_like(ridge,1.2)
        sparse=cut_opportunity(terrain(ridge),self.contact)
        full=cut_opportunity(terrain(intact),self.contact)
        self.assertGreater(full['score_m3'],sparse['score_m3'])
        self.assertEqual(sparse['weakest_third_equivalent_volume_m3'],0)

    def test_weak_side_penalty_distinguishes_equal_total_volume(self):
        asymmetric=np.full((30,38),.75);asymmetric[15:]=1.15
        symmetric=np.full_like(asymmetric,.95)
        a=cut_opportunity(terrain(asymmetric),self.contact)
        b=cut_opportunity(terrain(symmetric),self.contact)
        self.assertAlmostEqual(a['observed_cut_band_volume_m3'],b['observed_cut_band_volume_m3'])
        self.assertLess(a['score_m3'],b['score_m3'])

    def test_height_translation_does_not_change_opportunity(self):
        a=cut_opportunity(terrain(np.full((30,38),1.2)),self.contact)
        b=cut_opportunity(terrain(np.full((30,38),2.2)),dict(self.contact,contact_surface_z_m=2.))
        self.assertAlmostEqual(a['score_m3'],b['score_m3'])

    def test_unknown_or_outside_is_rejected(self):
        t=terrain(np.full((30,38),1.2));t['point_count'][14,18]=0
        with self.assertRaisesRegex(ValueError,'unknown'):cut_opportunity(t,self.contact)
        with self.assertRaisesRegex(ValueError,'outside'):
            cut_opportunity(t,dict(self.contact,contact_xy_m=[8.,0.]))

    def test_marker_span_rotates_about_center_and_has_full_width(self):
        for xy in ([5.6,.7],[5.6,-.7]):
            m=target_mouth_markers(dict(entry_xy_m=xy,surface_z_m=1.))
            delta=m['side_positive']-m['side_negative']
            self.assertAlmostEqual(np.linalg.norm(delta),.6)
            self.assertAlmostEqual(delta[:2]@np.asarray(xy),0)
            np.testing.assert_allclose((m['side_positive']+m['side_negative'])/2,m['center'])
            self.assertEqual(m['center'][2],1.)


if __name__=='__main__': unittest.main()
