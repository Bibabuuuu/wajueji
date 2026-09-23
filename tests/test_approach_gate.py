import copy,unittest
import numpy as np
from planning.approach import bucket_clearance,sampled_approach,boxes_overlap
from planning.command_gate import AutomaticIntentGate


def inputs():
    p={'schema':'excavation_observation_v1','episode':'e','fresh':True,'frame':'world','units':{'length':'m','time':'s'},
       'time':{'capture_sim_s':1.,'expires_sim_s':1.3,'capture_monotonic_s':10.,'expires_monotonic_s':10.3},
       'terrain':{'valid':True},'proprioception':{'joints':[0.]*19}}
    intent={'episode':'e','plan_id':'p','sequence':1,'created_monotonic_s':10.,'velocity':[.01,0.,0.,0.]}
    return p,intent


class GateTests(unittest.TestCase):
    def setUp(self):self.gate=AutomaticIntentGate([.12,.0225,.0225,.0225]);self.p,self.i=inputs()
    def choose(self,wall=10.,sim=1.,**kw):return self.gate.select(self.p,self.i,now_wall_s=wall,now_sim_s=sim,**kw)
    def test_default_disabled_then_explicit_binding(self):
        self.assertFalse(self.choose()['automatic_enabled']);self.gate.arm('e','p');self.assertTrue(self.choose()['automatic_enabled'])
    def test_observation_expiry_latches_even_after_recovery(self):
        self.gate.arm('e','p');self.i['created_monotonic_s']=10.31
        self.assertEqual(self.choose(wall=10.31)['automatic_velocity'],[0.]*4)
        self.p['time'].update(capture_monotonic_s=10.31,expires_monotonic_s=10.61)
        self.assertFalse(self.choose(wall=10.31)['automatic_enabled']);self.gate.arm('e','p')
        self.assertTrue(self.choose(wall=10.31)['automatic_enabled'])
    def test_stale_command_and_future_observation_rejected(self):
        self.gate.arm('e','p');self.assertFalse(self.choose(wall=10.06)['automatic_enabled'])
        self.gate.arm('e','p');self.assertFalse(self.choose(wall=9.99)['automatic_enabled'])
    def test_manual_pause_fault_take_priority_and_clear(self):
        for flag in ['manual','paused','fault']:
            self.gate.arm('e','p');r=self.choose(**{flag:True});self.assertEqual(r['automatic_velocity'],[0.]*4)
            self.assertFalse(self.choose()['automatic_enabled'])
    def test_mismatched_plan_episode_and_bad_velocity(self):
        for field,value in [('plan_id','other'),('episode','other'),('velocity',[1,0,0,0]),('velocity',[np.nan,0,0,0])]:
            self.p,self.i=inputs();self.i[field]=value;self.gate.arm('e','p');self.assertFalse(self.choose()['automatic_enabled'])
    def test_reordered_and_mutated_same_sequence_rejected(self):
        self.gate.arm('e','p');self.choose();self.i['velocity']=[0.,0.,0.,0.]
        self.assertFalse(self.choose()['automatic_enabled'])
    def test_no_input_mutation_or_publication(self):
        a,b=copy.deepcopy(self.p),copy.deepcopy(self.i);self.gate.arm('e','p');r=self.choose()
        self.assertEqual(self.p,a);self.assertEqual(self.i,b);self.assertFalse(r['published'])


class ApproachTests(unittest.TestCase):
    def setUp(self):
        self.terrain={'valid':True,'height_m':np.full((10,10),.3).tolist(),'point_count':np.full((10,10),5).tolist(),'origin_xy_m':[0,0],'resolution_m':.1}
        self.vertices=np.array([[.21,.21,.6],[.49,.49,.8]])
    def test_whole_bucket_and_unknown_corridor(self):
        self.assertGreater(bucket_clearance(self.vertices,self.terrain),.15)
        self.terrain['height_m'][3][3]=None
        with self.assertRaisesRegex(ValueError,'unknown'):bucket_clearance(self.vertices,self.terrain)
    def test_high_cell_between_vertices_is_detected(self):
        self.terrain['height_m'][3][3]=.7
        with self.assertRaisesRegex(ValueError,'clearance'):bucket_clearance(self.vertices,self.terrain)
    def test_outside_and_truck_overlap(self):
        with self.assertRaisesRegex(ValueError,'outside'):bucket_clearance(self.vertices+[-.5,0,0],self.terrain)
        self.assertTrue(boxes_overlap(self.vertices,self.vertices+.05));self.assertFalse(boxes_overlap(self.vertices,self.vertices+3))
    def path(self,check,advance=None,end=None):
        return sampled_approach([0,0],[.1,.1] if end is None else end,axes=[0,1],speed_limits=[.1,.1],
                                advance=advance or (lambda q,v,dt:q+v*dt),check_pose=check)
    def test_every_sample_checked_and_velocity_bounded(self):
        calls=[];r=self.path(lambda q,previous:calls.append(q.copy()))
        self.assertEqual(len(calls),r['steps']+1);self.assertLessEqual(np.abs(r['velocity']).max(),.1)
        np.testing.assert_allclose(r['qpos'][-1],[.1,.1]);self.assertFalse(r['execution_approved'])
    def test_collision_in_middle_blocks_path_with_good_endpoints(self):
        def check(q,previous):
            if .04<q[0]<.06:raise ValueError('middle_obstacle')
        with self.assertRaisesRegex(ValueError,'middle_obstacle'):self.path(check)
    def test_validation_substeps_check_midpoint_without_changing_command_rate(self):
        calls=[]
        result=sampled_approach([0,0],[.1,.1],axes=[0,1],speed_limits=[.1,.1],
                                advance=lambda q,v,dt:q+v*dt,
                                check_pose=lambda q,previous:calls.append(q.copy()),
                                dt=1/60,validation_substeps=2)
        self.assertEqual(len(calls),2*result['steps']+1)
        self.assertEqual(result['validation_substeps'],2)
        self.assertEqual(result['validation_dt_s'],1/120)
        self.assertEqual(len(result['velocity']),result['steps'])
        self.assertEqual(len(result['qpos']),result['steps']+1)
        np.testing.assert_allclose(result['qpos'][-1],[.1,.1])
    def test_validation_substeps_contract(self):
        for bad in (0,1.5,True):
            with self.assertRaisesRegex(ValueError,'invalid_path_input'):
                sampled_approach([0],[.1],axes=[0],speed_limits=[.1],
                                 advance=lambda q,v,dt:q+v*dt,
                                 check_pose=lambda q,previous:None,
                                 validation_substeps=bad)

if __name__=='__main__':unittest.main()
