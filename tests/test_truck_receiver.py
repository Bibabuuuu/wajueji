import unittest
import numpy as np
from terrain.truck_receiver import TruckReceiver, ReceiverDisplay, OUTSIDE, CAVITY, BOUNDARY, ABOVE_RIM


def box():
    return {'schema':'truck_receiver_triangles_v1','local_to_world':np.eye(4).tolist(),
        'boundary_band_m':.01,'rim_stations':[[0,0,1,1],[1,0,1,1]],
        'triangles':{
            'floor':[[[0,0,0],[1,0,0],[1,1,0]],[[0,0,0],[1,1,0],[0,1,0]]],
            'left':[[[0,0,0],[1,0,0],[1,0,1]],[[0,0,0],[1,0,1],[0,0,1]]],
            'right':[[[0,1,0],[1,1,0],[1,1,1]],[[0,1,0],[1,1,1],[0,1,1]]],
        }}


class ReceiverTests(unittest.TestCase):
    def test_core_steel_floor_and_airborne_separate(self):
        r = TruckReceiver(box())
        p = [[.5,.5,.5],[.5,.005,.5],[.5,-.02,.5],[.5,.5,-.02],
             [.5,.5,3.],[2,.5,.5],[.5,.5,.005]]
        np.testing.assert_array_equal(r.classify(p), [CAVITY,BOUNDARY,OUTSIDE,OUTSIDE,ABOVE_RIM,OUTSIDE,BOUNDARY])

    def test_rotated_translated_receiver_invariance(self):
        r = TruckReceiver(box()); p = np.array([[.5,.5,.5],[.5,.5,2.],[.5,-.1,.5]])
        t = np.array([[0,-1,0,8],[1,0,0,4],[0,0,1,2],[0,0,0,1]], float)
        np.testing.assert_array_equal(r.classify(p@t[:3,:3].T+t[:3,3],t),r.classify(p))

    def test_sloped_floor_rejects_points_below_steel(self):
        c = box()
        for name in c['triangles']:
            c['triangles'][name] = (np.array(c['triangles'][name]) +
                np.array(c['triangles'][name])[:,:,[0]]*np.array([0,0,.2])).tolist()
        c['rim_stations'][1][3] = 1.2
        r = TruckReceiver(c)
        np.testing.assert_array_equal(r.classify([[.5,.5,.05],[.5,.5,.3]]),[OUTSIDE,CAVITY])

    def test_invalid_pose_and_points_rejected(self):
        r = TruckReceiver(box())
        bad = np.eye(4); bad[0,0] = 2
        for pose in (bad, np.zeros((4,4)), [[1]]):
            with self.assertRaises(ValueError): r.classify([[.5,.5,.5]],pose)
        with self.assertRaises(ValueError): r.classify([[float('nan'),0,0]])

    def test_inward_rear_overhang_is_a_ceiling_not_a_floor(self):
        c = box()
        c['triangles']['ceiling'] = [[[0,0,.75],[.25,0,.75],[.25,1,.75]],
                                    [[0,0,.75],[.25,1,.75],[0,1,.75]]]
        r = TruckReceiver(c)
        np.testing.assert_array_equal(r.classify([[.1,.5,.6],[.1,.5,.9],[.5,.5,.9]]),[CAVITY,OUTSIDE,CAVITY])

    def test_display_mass_partition_and_stale_values_not_drawn(self):
        d = ReceiverDisplay(TruckReceiver(box()),[2,3,5],stale_after_s=.5)
        self.assertFalse(d.status(0)['valid'])
        p = [[.5,.5,.5],[.5,.005,.5],[.5,.5,2.]]
        row = d.observe(p,0,10,other_owner_mask=np.zeros(3,bool))
        self.assertEqual(row['core_region_kg'],2.)
        self.assertEqual(row['boundary_band_kg'],3.)
        self.assertEqual(row['above_rim_unverified_kg'],5.)
        self.assertEqual(row['partition_residual_kg'],0.)
        class UI:
            def __init__(self): self.lines=[]
            def text(self,s): self.lines.append(s)
        ui = UI(); d.draw_ui(ui,10.25)
        self.assertTrue(any('2.00 kg' in s for s in ui.lines))
        ui = UI(); d.draw_ui(ui,11)
        self.assertTrue(any('stale_observation' in s for s in ui.lines))
        self.assertFalse(any('kg' in s for s in ui.lines))

    def test_same_time_does_not_fake_freshness_or_readvance(self):
        d = ReceiverDisplay(TruckReceiver(box()),[2],stale_after_s=.5)
        d.observe([[.5,.5,.5]],0,10,other_owner_mask=np.zeros(1,bool))
        r = d.observe([[.5,.5,3]],0,11,other_owner_mask=np.zeros(1,bool))
        self.assertEqual(r['core_region_kg'],2.)
        self.assertFalse(d.status(11)['valid'])
        self.assertEqual(d.observe([[.5,.5,3]],1,11,other_owner_mask=np.zeros(1,bool))['above_rim_unverified_kg'],2.)
        with self.assertRaises(ValueError): d.observe([[.5,.5,.5]],0,12,other_owner_mask=np.zeros(1,bool))
        with self.assertRaises(ValueError): d.observe([[.5,.5,.5]],2,9,other_owner_mask=np.zeros(1,bool))
        with self.assertRaises(ValueError): d.observe([],2,12,other_owner_mask=np.zeros(1,bool))

    def test_other_owner_not_double_counted_and_labels_read_only(self):
        d = ReceiverDisplay(TruckReceiver(box()),[2,3])
        r = d.observe([[.5,.5,3],[.5,.5,.5]],0,10,other_owner_mask=np.array([True,False]))
        self.assertEqual(r['other_owner_kg'],2.)
        self.assertEqual(r['above_rim_unverified_kg'],0.)
        self.assertEqual(r['core_region_kg'],3.)
        self.assertEqual(r['partition_residual_kg'],0.)
        with self.assertRaises(ValueError): d.last_labels[0]=0
        with self.assertRaises(ValueError): d.observe([[.5,.5,3],[.5,.5,.5]],1,11,other_owner_mask=[1,0])


if __name__ == '__main__': unittest.main()
