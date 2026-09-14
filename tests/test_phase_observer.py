import copy
import unittest
import numpy as np
from planning.phase_observer import PhaseObserver,PhaseConfig,bucket_shell_points


def packet(seq=0,sim=1.,wall=100.,force=200.,near=True,episode='test'):
    pose=np.eye(4);pose[:3,3]=[.45,.45,.3] if near else [3.,3.,2.]
    return {'schema':'excavation_observation_v1','episode':episode,'sequence':seq,'frame':'world',
            'units':{'length':'m','force':'N','time':'s'},'fresh':True,
            'time':{'capture_sim_s':sim,'processed_sim_s':sim,'capture_monotonic_s':wall,
                    'processed_monotonic_s':wall,'expires_sim_s':sim+.3,'expires_monotonic_s':wall+.3},
            'proprioception':{'world_from_bucket':pose.tolist()},
            'terrain':{'valid':True,'height_m':np.full((9,9),.3).tolist(),'point_count':np.full((9,9),10).tolist(),
                       'shape_yx':[9,9],'origin_xy_m':[0.,0.],'resolution_m':.1},
            'load':{'valid':True,'sample_sim_s':sim,'sample_monotonic_s':wall,'wrench_world':[force,0,0,0,0,0]}}


class PhaseTests(unittest.TestCase):
    def setUp(self):self.observer=PhaseObserver([[0,0,0],[.03,0,0],[0,.03,0]])
    def apply(self,p,delta=0):return self.observer.update(p,p['time']['capture_sim_s']+delta,p['time']['capture_monotonic_s']+delta)

    def test_distinct_samples_confirm_and_hysteresis_releases(self):
        self.assertEqual(self.apply(packet())['stage'],'CONTACT_CANDIDATE')
        self.assertEqual(self.apply(packet(1,1.1,100.2))['stage'],'CONTACT_CONFIRMED')
        self.assertTrue(self.apply(packet(2,1.2,100.4,force=100))['contact_confirmed'])
        self.assertEqual(self.apply(packet(3,1.3,100.6,force=50))['stage'],'NEAR_SURFACE')

    def test_duplicate_and_paused_samples_cannot_confirm(self):
        p=packet();self.apply(p)
        for _ in range(5):self.assertFalse(self.apply(p,.01)['contact_confirmed'])
        paused=packet(1,1.,100.2);paused['load']=copy.deepcopy(p['load'])
        self.assertFalse(self.apply(paused)['contact_confirmed'])
        paused=packet(2,1.,100.4);paused['load']=copy.deepcopy(p['load'])
        self.assertEqual(self.apply(paused)['stage'],'CONTACT_UNKNOWN')

    def test_single_spike_does_not_accumulate(self):
        for i,force in enumerate([200,0,200,0,200]):self.assertFalse(self.apply(packet(i,1+i*.1,100+i*.2,force))['contact_confirmed'])

    def test_gap_expiry_and_episode_reset_confirmation(self):
        self.apply(packet());self.apply(packet(1,1.1,100.2))
        self.assertEqual(self.apply(packet(2,1.2,100.7))['stage'],'CONTACT_CANDIDATE')
        self.assertEqual(self.apply(packet(2,1.2,100.7),.31)['stage'],'WAIT_OBSERVATION')
        self.assertEqual(self.apply(packet(0,0.,101.,episode='new'))['stage'],'CONTACT_CANDIDATE')

    def test_out_of_order_and_wrong_units_are_rejected(self):
        self.apply(packet(3))
        self.assertEqual(self.apply(packet(2))['stage'],'WAIT_OBSERVATION')
        p=packet(4);p['units']['length']='cm'
        self.assertEqual(self.apply(p)['stage'],'WAIT_OBSERVATION')

    def test_missing_stale_future_force_remains_unknown(self):
        for mutate in [lambda p:p['load'].pop('sample_monotonic_s'),
                       lambda p:p['load'].update(sample_monotonic_s=99.),
                       lambda p:p['load'].update(sample_sim_s=2.),
                       lambda p:p['load'].update(valid=False)]:
            p=packet();mutate(p);self.assertEqual(self.apply(p)['stage'],'CONTACT_UNKNOWN')

    def test_force_far_from_visible_surface_is_unlocalized(self):
        self.assertEqual(self.apply(packet(near=False))['stage'],'CONTACT_UNLOCALIZED')

    def test_unknown_sparse_flat_and_rough_patches_are_not_candidates(self):
        for height,count in [(None,0),(.3,1),(0.,10)]:
            p=packet(near=False);p['terrain']['height_m']=[[height]*9 for _ in range(9)];p['terrain']['point_count']=[[count]*9 for _ in range(9)]
            self.assertIsNone(self.apply(p)['candidate'])
        p=packet(near=False);p['terrain']['height_m']=(np.indices((9,9)).sum(axis=0)%2*.5+.1).tolist()
        self.assertIsNone(self.apply(p)['candidate'])

    def test_candidates_exclude_entire_bucket_neighborhood(self):
        result=self.apply(packet())
        self.assertIsNotNone(result['candidate'])
        x,y,_=result['candidate']['surface_world_m']
        self.assertTrue(x<.2 or x>.7 or y<.2 or y>.7)
        self.assertFalse(result['candidate']['excavation_target_approved'])

    def test_readonly_no_commands_and_absolute_expiry(self):
        p=packet();before=copy.deepcopy(p);r=self.apply(p)
        self.assertEqual(p,before);self.assertIsNone(r['command']);self.assertFalse(r['autonomous_task_ready'])
        self.assertEqual(r['expires_monotonic_s'],100.3)

    def test_invalid_shapes_and_payload_fail_closed(self):
        for p in [None,{},packet()]:
            if p and 'terrain' in p:p['terrain']['point_count']=[[float('nan')]]
            self.assertEqual(self.observer.update(p,1.,100.)['stage'],'WAIT_OBSERVATION')
        with self.assertRaises(ValueError):PhaseConfig(contact_enter_n=10,contact_exit_n=20)

    def test_section_transform_contract(self):
        points=bucket_shell_points({'bucket':{'coordinate_frame':'bucket_body','polygon_xz_m':[[0,0],[1,0],[0,1]],'y_limits_m':[-.3,.3]}})
        self.assertEqual(points.shape,(9,3));self.assertAlmostEqual(points[:,1].min(),-.3)

    def test_motion_separate_from_contact(self):
        self.apply(packet())
        p=packet(1,1.1,100.2);p['proprioception']['world_from_bucket'][2][3]+=.01
        result=self.apply(p)
        self.assertEqual(result['motion']['vertical_motion'],'RISING')
        self.assertTrue(result['contact_confirmed'])
        self.assertTrue(result['motion']['contact_does_not_imply_excavation'])

    def test_motion_paused_gap_and_reset_are_unknown(self):
        self.apply(packet())
        for p in [packet(1,1.,100.2),packet(2,1.1,101.),packet(0,0.,102.,episode='new')]:
            self.assertEqual(self.apply(p)['motion']['vertical_motion'],'UNKNOWN')

    def test_motion_lowering_duplicate_and_deadband(self):
        self.apply(packet())
        p=packet(1,1.1,100.2);p['proprioception']['world_from_bucket'][2][3]-=.01
        result=self.apply(p);self.assertEqual(result['motion']['vertical_motion'],'LOWERING')
        self.assertEqual(self.apply(p)['motion'],result['motion'])
        p2=packet(2,1.2,100.4);p2['proprioception']=copy.deepcopy(p['proprioception'])
        self.assertEqual(self.apply(p2)['motion']['vertical_motion'],'VERTICALLY_STATIONARY')

if __name__=='__main__':unittest.main()
