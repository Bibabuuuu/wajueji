import unittest
import numpy as np
from manual_trajectory_alignment import ProgressAlignment, AXES
from manual_residual_sac_env import command_at, ManualResidualEnv
from types import SimpleNamespace
from pathlib import Path


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.q=np.zeros((100,19))
        self.q[:,12]=np.arange(100)*.0002
        self.poses=np.tile(np.eye(4),(100,1,1))
        self.poses[:,0,3]=np.arange(100)*.004

    def test_zero_timing_and_commands_are_preserved(self):
        matcher=ProgressAlignment(self.q,self.poses)
        commands=np.tile([-.0,.005,.01,.03],(99,1))
        for cursor in range(99):
            result=matcher.observe(self.q[cursor],self.poses[cursor],cursor)
            self.assertEqual(result['actual_progress'],cursor)
            self.assertTrue(result['trusted'])
            *_,request,pace=command_at(commands,cursor,np.zeros(5),
                                      reference_limit=result['reference_limit'])
            self.assertEqual(request.tobytes(),commands[cursor].tobytes())
            self.assertEqual(pace,1.)

    def test_lag_wait_blocks_both_feedforward_and_residual(self):
        matcher=ProgressAlignment(self.q,self.poses);matcher.measured=40
        result=matcher.observe(self.q[42],self.poses[42],48)
        self.assertEqual(result['actual_progress'],42)
        self.assertEqual(result['mode'],'wait')
        base,delta,request,pace=command_at(np.zeros((99,4)),48,np.ones(5),
                                          reference_limit=result['reference_limit'])
        self.assertEqual(pace,0)
        np.testing.assert_array_equal(base+delta+request,np.zeros(4))
        result=matcher.observe(self.q[43],self.poses[43],48)
        self.assertEqual(result['mode'],'follow')

    def test_wait_correction_is_directional_bounded_and_never_advances_reference(self):
        commands=np.zeros((99,4))
        error=np.array([.00001,.0016,-.0027,0.])
        base,delta,request,pace=command_at(commands,48,[1,1,-1,1,1],
                                          reference_limit=48,wait_target_error=error)
        self.assertEqual(pace,0);np.testing.assert_array_equal(base,np.zeros(4))
        np.testing.assert_allclose(delta,[.0006,.00225,-.00225,0.],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(delta,request)
        _,wrong,_,_=command_at(commands,48,[-1,-1,1,-1,1],
                               reference_limit=48,wait_target_error=error)
        np.testing.assert_array_equal(wrong,np.zeros(4))
        _,zero,_,_=command_at(commands,48,np.zeros(5),
                              reference_limit=48,wait_target_error=error)
        np.testing.assert_array_equal(zero,np.zeros(4))
        with self.assertRaises(ValueError):
            command_at(commands,48,np.ones(5),reference_limit=48,wait_target_error=[0,0,float('nan'),0])

    def test_reference_endpoint_can_correct_without_reading_past_commands(self):
        base,delta,request,pace=command_at(np.zeros((99,4)),99,np.ones(5),
                         reference_limit=99,wait_target_error=[0,.001,.002,0])
        self.assertEqual(pace,0)
        np.testing.assert_array_equal(base,np.zeros(4))
        np.testing.assert_allclose(request,[0,.00225,.00225,0],atol=1e-15,rtol=0)

    def test_overshoot_and_orientation_and_joint_disagreement_stop(self):
        for kind in ['position','rotation','joint']:
            matcher=ProgressAlignment(self.q,self.poses)
            q=self.q[0].copy();pose=self.poses[0].copy()
            if kind=='position':pose[2,3]=.3
            if kind=='rotation':pose[:3,:3]=np.diag([-1,-1,1])
            if kind=='joint':q[7]=.1
            result=matcher.observe(q,pose,0)
            self.assertFalse(result['trusted'])
            self.assertEqual(result['mode'],'outside')
            self.assertEqual(matcher.measured,0)

    def test_estimate_never_rewinds_and_repeated_pose_uses_clock_tie(self):
        matcher=ProgressAlignment(self.q,self.poses);matcher.measured=42
        self.assertEqual(matcher.observe(self.q[41],self.poses[41],48)['actual_progress'],42)
        q=np.zeros((100,19));poses=np.tile(np.eye(4),(100,1,1))
        matcher=ProgressAlignment(q,poses);matcher.measured=40
        self.assertEqual(matcher.observe(q[43],poses[43],43)['actual_progress'],43)
        with self.assertRaises(ValueError):matcher.observe(q[0],np.full((4,4),np.nan),0)

    def test_outside_corridor_finishes_before_any_physical_step(self):
        env=ManualResidualEnv.__new__(ManualResidualEnv)
        env.runtime=SimpleNamespace(frames=0)
        env.alignment=SimpleNamespace(measured=0)
        env.cursor=0.;env.hold=None;env.waiting_action_mode='signed_residual_v2'
        env.report=dict(status='running',last_alignment={'reference_limit':6.})
        env.directory=Path('unused-cpu-only-episode')
        env.budget=SimpleNamespace(check=lambda:None)
        env.collect=False;env.max_frames=1560
        env.previous_action=np.zeros(5)
        env.observation=lambda:np.zeros(221,np.float32)
        env.alignment_state=lambda stage:dict(trusted=False)
        def finish(status):env.report.update(status=status)
        env.finish=finish
        _,reward,done,truncated,info=env.step(np.ones(5))
        self.assertTrue(done);self.assertFalse(truncated)
        self.assertLessEqual(reward,-2)
        self.assertEqual(info['episode_summary']['status'],'alignment_stop')
        self.assertEqual(env.runtime.frames,0)


if __name__=='__main__':unittest.main()
