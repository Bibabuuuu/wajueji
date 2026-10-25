"""Real holding-step regression with zero-gap adhered material and CPU doubles."""
from types import SimpleNamespace
import unittest
import numpy as np
import test_manual_action_handoff as handoff
from manual_residual_sac_env import TERMINAL_SCORE_VERSION


class HoldScoreTests(unittest.TestCase):
    def held_env(self, *, invalid_sample=False):
        fixture=handoff.HandoffTests();self.addCleanup(fixture.doCleanups)
        env,requests=fixture.make_env('wait',['wait']*60,cursor=1170.,hold=0)
        env.report.update(scores=[dict(valid_estimate=True,fill_percent=150.)],
                          gap_checks=[dict(remaining_soil_gap_m=0.)],success=False)
        env.runtime.assembly=SimpleNamespace(assemble=lambda *a,**k:(None,{'bucket':np.eye(4)}))
        env.runtime.fixture=SimpleNamespace(state=SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda:np.zeros((1,3)))))
        env.geometry=SimpleNamespace(observe=lambda *a:dict(remaining_soil_gap_m=0.))
        def measure():
            env.report['scores'].append(dict(valid_estimate=not invalid_sample,
                                             fill_percent=None if invalid_sample else 145. if env.hold==15 else 148.))
            env.report['gap_checks'].append(dict(remaining_soil_gap_m=0.))
        env.measure=measure
        return env,requests

    def test_zero_gap_material_does_not_fail_completed_half_second_hold(self):
        env,requests=self.held_env()
        for i in range(10):
            _,reward,done,truncated,info=env.step(np.ones(5))
            self.assertEqual(done,i==9);self.assertFalse(truncated)
            self.assertEqual(info['terminal_score_version'],TERMINAL_SCORE_VERSION)
        self.assertEqual(env.hold,30);self.assertEqual(env.runtime.frames,30)
        self.assertEqual(env.report['status'],'completed');self.assertTrue(env.report['success'])
        self.assertEqual(env.report['retained_fill_fraction'],1.45)
        self.assertGreater(info['raw_reward'],1.449)
        np.testing.assert_array_equal(requests,np.zeros((30,4)))
        self.assertTrue(all(x['remaining_soil_gap_m']==0 for x in env.report['gap_checks']))

    def test_invalid_volume_still_fails_despite_removed_gap_gate(self):
        env,_=self.held_env(invalid_sample=True)
        for _ in range(10):_,reward,done,truncated,info=env.step(np.zeros(5))
        self.assertTrue(done);self.assertFalse(truncated)
        self.assertEqual(env.report['status'],'invalid_endpoint');self.assertFalse(env.report['success'])
        self.assertIsNone(env.report['retained_fill_fraction']);self.assertLessEqual(info['raw_reward'],-2)

    def test_endpoint_pose_must_be_ready_before_holding(self):
        fixture=handoff.HandoffTests();self.addCleanup(fixture.doCleanups)
        env,_=fixture.make_env('wait',['wait']*6,cursor=1170.,hold=None)
        env.step(np.r_[-np.ones(4),0.])
        self.assertIsNone(env.hold);self.assertEqual(env.report['status'],'running')


if __name__=='__main__':unittest.main()
