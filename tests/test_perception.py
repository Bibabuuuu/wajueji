import unittest
import numpy as np
from perception.core import Camera, Grid, StateBuilder, LatestSchedule, observation_fresh, look_at, noisy_depth, rigid_transform, terrain_from_points


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(3, 3, 2, 2, 1, 1, .1, 10)
        self.grid = Grid(-2, 2, -2, 2, 1)

    def arguments(self):
        return dict(episode='a', sequence=0, capture_sim_s=0, now_sim_s=0, capture_wall_s=1, now_wall_s=1,
                    depth=np.ones((3, 3)), world_from_camera=np.eye(4), bucket_pose=np.eye(4), joints=[0]*4,
                    wrench=[0]*6, truck_prior={'frame':'world'})

    def test_projection_and_world_transform(self):
        pose=np.eye(4);pose[:3,3]=[4,5,6]
        points,valid=self.camera.points(np.full((3,3),2.),pose)
        np.testing.assert_allclose(points[4],[4,5,8]);np.testing.assert_allclose(points[0],[3,4,8])
        self.assertTrue(valid.all())

    def test_camera_orientation(self):
        pose=look_at([5,-5,5],[0,0,0]);rigid_transform(pose)
        np.testing.assert_allclose(pose[:3,2],np.array([-1,1,-1])/np.sqrt(3))

    def test_unknown_and_half_open_grid_boundaries(self):
        h,n=terrain_from_points(np.array([[-2,-2,.3],[-2,-2,.5],[2,0,99],[0,2,99],[np.nan,0,0]]),self.grid)
        self.assertEqual(n.sum(),2);self.assertEqual(h[0,0],.5);self.assertEqual(np.isfinite(h).sum(),1)

    def test_noise_reproducible_and_dropout_unknown(self):
        d=np.full((3,3),2.);d[0,0]=np.nan
        np.testing.assert_equal(noisy_depth(d,self.camera,seed=3),noisy_depth(d,self.camera,seed=3))
        self.assertTrue(np.isnan(noisy_depth(d,self.camera,seed=3,dropout=1)).all())

    def test_self_filter_and_no_truth_mass(self):
        args=self.arguments();args['self_mask']=np.ones((3,3),bool)
        p,h,n=StateBuilder(self.camera,self.grid).build(**args)
        self.assertFalse(p['terrain']['valid']);self.assertTrue(np.isnan(h).all())
        self.assertIsNone(p['load']['mass_kg']);self.assertIsNone(p['load']['fill_ratio'])
        self.assertFalse(p['truck']['visual_localization_verified'])

    def test_staleness(self):
        args=self.arguments();args.update(now_sim_s=.31,now_wall_s=1.01)
        p,_,_=StateBuilder(self.camera,self.grid).build(**args)
        self.assertFalse(p['fresh']);self.assertFalse(p['terrain']['valid']);self.assertFalse(p['load']['valid'])
        args.update(now_sim_s=0,now_wall_s=1.31)
        self.assertFalse(StateBuilder(self.camera,self.grid).build(**args)[0]['fresh'])

    def test_order_reset_and_error_does_not_consume_sequence(self):
        b=StateBuilder(self.camera,self.grid);args=self.arguments();b.build(**args)
        with self.assertRaises(ValueError):b.build(**args)
        args.update(sequence=1,capture_sim_s=-1)
        with self.assertRaises(ValueError):b.build(**args)
        args.update(capture_sim_s=0);b.build(**args)
        args.update(episode='b',sequence=0)
        with self.assertRaises(ValueError):b.build(**args)
        b.reset();b.build(**args)

    def test_invalid_sensor_inputs(self):
        args=self.arguments();args['wrench']=[np.nan]*6
        with self.assertRaises(ValueError):StateBuilder(self.camera,self.grid).build(**args)
        with self.assertRaises(ValueError):Camera(fx=0)
        with self.assertRaises(ValueError):Grid(resolution=0)
        pose=np.eye(4);pose[0,0]=2
        with self.assertRaises(ValueError):rigid_transform(pose)
        p,valid=self.camera.points(np.array([[0,np.inf,20],[1,1,1],[1,1,1]]),np.eye(4))
        self.assertEqual(len(p),6);self.assertEqual(valid.sum(),6)

    def test_dual_view_union_and_no_history(self):
        args=self.arguments();first=np.full((3,3),np.nan);first[1,0]=1
        second=np.full((3,3),np.nan);second[1,2]=1
        args.update(depth=first,extra_views=[(self.camera,second,np.eye(4),None)])
        b=StateBuilder(self.camera,self.grid);p,h,n=b.build(**args)
        self.assertEqual(np.isfinite(h).sum(),2);self.assertEqual(n.sum(),2)
        self.assertEqual(p['terrain']['camera_count'],2)
        args.update(sequence=1,depth=np.full((3,3),np.nan),extra_views=[])
        p,h,n=b.build(**args);self.assertTrue(np.isnan(h).all());self.assertFalse(p['terrain']['valid'])

    def test_published_packet_expires_without_new_frames(self):
        p,_,_=StateBuilder(self.camera,self.grid).build(**self.arguments())
        self.assertTrue(observation_fresh(p,0,1.1))
        self.assertFalse(observation_fresh(p,0,1.31));self.assertFalse(observation_fresh(p,.31,1.1))
        self.assertFalse(observation_fresh(p,0,.9))

    def test_sampling_skips_missed_deadlines_no_backlog(self):
        s=LatestSchedule(.2);self.assertTrue(s.due(10));self.assertFalse(s.due(10.1))
        self.assertTrue(s.due(10.81));self.assertEqual(s.skipped,3)
        self.assertFalse(s.due(10.82));self.assertTrue(s.due(11.01))
        with self.assertRaises(ValueError):s.due(10)


if __name__=='__main__':unittest.main()
