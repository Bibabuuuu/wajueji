import unittest
import numpy as np
from planning.reference_tracking import ReferenceTracker

class ReferenceTrackingTest(unittest.TestCase):
    def make(self):
        p=np.zeros((121,19));p[:,9]=np.linspace(0,.02,121)
        return ReferenceTracker(p,[.12,.0225,.0225,.0675],target_bounds=np.tile([-1.,1.],(4,1)))
    def test_internal_target_is_controlled_not_measured_pose(self):
        t=self.make();q=t.path[0].copy();q[9]=-.002
        v=t.command(time_s=0,qpos=q,target=np.zeros(4),axis_velocity=np.zeros(4))
        self.assertGreater(v[1],.01);self.assertLess(v[1],.013)
    def test_lag_holds_reference_then_resumes(self):
        t=self.make();q=t.path[0].copy();q[9]=-.006
        v=t.command(time_s=0,qpos=q,target=np.zeros(4),axis_velocity=np.zeros(4))
        self.assertEqual(t.phase,0.);self.assertGreater(v[1],0.)
        t.command(time_s=t.dt,qpos=t.path[0],target=np.zeros(4),axis_velocity=np.zeros(4))
        self.assertGreater(t.phase,0.)
    def test_duplicate_time_latches_failure(self):
        t=self.make();args=dict(time_s=0,qpos=t.path[0],target=np.zeros(4),axis_velocity=np.zeros(4))
        t.command(**args)
        with self.assertRaisesRegex(ValueError,'clock_discontinuity'):t.command(**args)
        args['time_s']=t.dt
        with self.assertRaisesRegex(ValueError,'clock_discontinuity'):t.command(**args)
    def test_departure_rejected_without_advancing(self):
        t=self.make();q=t.path[0].copy();q[9]=-.02
        with self.assertRaisesRegex(ValueError,'corridor_exceeded'):
            t.command(time_s=0,qpos=q,target=np.zeros(4),axis_velocity=np.zeros(4))
        self.assertEqual(t.phase,0.)
    def test_nonfinite_and_speed_budget(self):
        t=self.make();p=t.path.copy();p[-1,9]=1.
        with self.assertRaisesRegex(ValueError,'speed_limit'):ReferenceTracker(p,t.limits,target_bounds=t.bounds)
        with self.assertRaisesRegex(ValueError,'invalid_tracking_state'):
            t.command(time_s=0,qpos=t.path[0],target=[0,np.nan,0,0],axis_velocity=np.zeros(4))
    def test_clock_end_does_not_mean_settled(self):
        t=self.make();t.phase=t.duration;q=t.path[-1].copy()
        t.command(time_s=0,qpos=q,target=q[t.axes],axis_velocity=[0,.01,0,0])
        self.assertFalse(t.done)
        t.command(time_s=t.dt,qpos=q,target=q[t.axes],axis_velocity=np.zeros(4))
        self.assertTrue(t.done)
    def test_compensation_is_bounded_at_target_margin(self):
        t=self.make();t.bounds[1]=[-.1,.001];q=t.path[0].copy();q[9]=-.004
        for step in range(80):
            t.command(time_s=step*t.dt,qpos=q,target=np.zeros(4),axis_velocity=np.zeros(4))
            self.assertLessEqual(t.reference()[9]+t.bias[1],.001+1e-12)
    def test_manual_endpoint_outside_margin_is_not_silently_clipped(self):
        # Original manual bucket endpoint is 0.855 m; current 8 mm margin
        # bounds its target to 0.852 m. Slowing this path cannot admit it.
        p=np.zeros((121,19));p[:,15]=np.linspace(.851,.855,121)
        bounds=np.tile([-1.,1.],(4,1));bounds[3]=[.378,.852]
        for path_dt in (1/120,1/60):
            with self.assertRaisesRegex(ValueError,'reference_target_bounds'):
                ReferenceTracker(p,[.12,.0225,.0225,.0675],target_bounds=bounds,path_dt=path_dt)
    def test_interior_reference_violation_is_rejected(self):
        p=np.zeros((121,19));p[:,15]=.852
        p[60,15]+=.0001
        bounds=np.tile([-1.,1.],(4,1));bounds[3]=[.378,.852]
        with self.assertRaisesRegex(ValueError,'reference_target_bounds'):
            ReferenceTracker(p,[.12,.0225,.0225,.0675],target_bounds=bounds)
if __name__=='__main__':unittest.main()
