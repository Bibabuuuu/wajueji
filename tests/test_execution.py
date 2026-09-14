import copy,unittest
import numpy as np
from planning.execution import ApproachExecutor


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.now=[0.,10.];self.q=np.zeros(19);self.commits=[]
        self.path=np.zeros((4,19));self.path[:,0]=np.arange(4)*.001
        self.p={'schema':'excavation_observation_v1','episode':'e','fresh':True,'frame':'world',
                'units':{'length':'m','time':'s'},'time':{'capture_sim_s':0.,'expires_sim_s':.3,'capture_monotonic_s':10.,'expires_monotonic_s':10.3},
                'terrain':{'valid':True,'origin_xy_m':[0,0],'resolution_m':.1,'height_m':[[0.]]},'proprioception':{'joints':self.q.tolist()}}
        self.e=ApproachExecutor(self.path,[0,1,2,3],[.12,.1,.1,.1],.01,([0,0],.1,[1,1]))
        self.check=lambda q,old,p:{}
        self.e.arm(self.p,self.q,'p',clock=lambda:tuple(self.now),check_pose=self.check)
    def step(self,**kw):
        def advance(q,v,dt):q[:4]+=v*dt;return q
        return self.e.step(self.p,self.q,clock=lambda:tuple(self.now),advance=advance,
                           check_pose=kw.pop('check_pose',self.check),prepare=lambda q,v:lambda:self.commits.append(q.copy()),**kw)
    def test_completed_path_and_hold(self):
        for _ in range(3):
            self.q,r=self.step();self.assertTrue(r['consumed']);self.now[0]+=.01
        self.q,r=self.step();self.assertEqual(r['reason'],'path_complete');self.assertEqual(len(self.commits),3)
        np.testing.assert_array_equal(self.q,self.path[-1])
    def test_expiry_during_geometry_never_commits(self):
        def slow(q,old,p):self.now[1]+=.31;return {}
        q,r=self.step(check_pose=slow)
        self.assertEqual(r['reason'],'observation_invalid_or_expired');self.assertEqual(self.commits,[])
        np.testing.assert_array_equal(q,self.q);self.assertEqual(self.e.cursor,0)
    def test_command_expiry_after_prepare_or_geometry(self):
        def slow(q,old,p):self.now[1]+=.051;return {}
        _,r=self.step(check_pose=slow);self.assertEqual(r['reason'],'command_expired_or_future');self.assertEqual(self.commits,[])
    def test_recovery_cannot_resume(self):
        self.now[1]=10.31;_,r=self.step();self.assertFalse(r['consumed'])
        self.p['time'].update(capture_monotonic_s=10.31,expires_monotonic_s=10.61)
        _,r=self.step();self.assertFalse(r['consumed']);self.assertEqual(self.commits,[])
    def test_obstacle_and_tracking_error_hold(self):
        def obstacle(q,old,p):raise ValueError('changed_terrain_obstacle')
        _,r=self.step(check_pose=obstacle);self.assertEqual(r['reason'],'changed_terrain_obstacle');self.assertEqual(self.commits,[])
        self.setUp();self.q[0]=.0001;_,r=self.step();self.assertEqual(r['reason'],'actual_path_tracking_error')
    def test_grid_and_sensor_pose_change_block(self):
        self.p['terrain']['resolution_m']=.2;_,r=self.step();self.assertEqual(r['reason'],'map_grid_changed')
        self.setUp();self.p['proprioception']['joints'][0]=.02;_,r=self.step();self.assertEqual(r['reason'],'sensor_pose_tracking_error')
    def test_fault_manual_pause_do_not_apply(self):
        for flag in ('fault','manual','paused'):
            self.setUp();_,r=self.step(**{flag:True});self.assertFalse(r['consumed']);self.assertEqual(self.commits,[])
    def test_arm_wrong_start_does_not_enable(self):
        q=self.q.copy();q[0]=1.
        with self.assertRaisesRegex(ValueError,'tracking'):self.e.arm(self.p,q,'p',clock=lambda:tuple(self.now),check_pose=self.check)
        self.assertFalse(self.e.gate.enabled)

if __name__=='__main__':unittest.main()
