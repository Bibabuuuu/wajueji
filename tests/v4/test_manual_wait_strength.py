import unittest
import numpy as np
from manual_residual_sac_env import command_at, LIMITS_SI


class WaitStrengthTests(unittest.TestCase):
    def test_stop_neutral_full_and_both_error_signs(self):
        commands=np.zeros((2,4));error=np.array([.02,-.01,.01,-.02])
        for action,strength in [(-1,0),(0,.5),(1,1)]:
            a=np.r_[np.full(4,action),0.]
            base,delta,request,pace=command_at(commands,0,a,reference_limit=0,
                wait_target_error=error,waiting_action_mode='error_strength_v1')
            np.testing.assert_array_equal(base,np.zeros(4));self.assertEqual(pace,0)
            np.testing.assert_allclose(delta,np.sign(error)*LIMITS_SI*.1*strength,rtol=0,atol=1e-16)
            np.testing.assert_array_equal(request,delta)

    def test_remaining_error_caps_and_zero_error(self):
        error=np.array([0,1e-10,-1e-9,.02])
        delta=command_at(np.zeros((2,4)),0,np.ones(5),reference_limit=0,
            wait_target_error=error,waiting_action_mode='error_strength_v1')[1]
        self.assertEqual(delta[0],0)
        self.assertTrue(np.all(abs(delta)/60<=abs(error)+1e-16))
        self.assertTrue(np.all(abs(delta)<=LIMITS_SI*.1+1e-16))
        self.assertTrue(np.all(delta*error>=0))

    def test_follow_identical_and_switches_only_on_wait(self):
        rng=np.random.default_rng(42);commands=rng.uniform(-1,1,(10,4))*LIMITS_SI
        for _ in range(100):
            a=rng.uniform(-1,1,5);cursor=rng.uniform(0,8)
            legacy=command_at(commands,cursor,a,reference_limit=cursor+1.2)
            directed=command_at(commands,cursor,a,reference_limit=cursor+1.2,waiting_action_mode='error_strength_v1')
            for x,y in zip(legacy,directed):np.testing.assert_array_equal(x,y)
        for limit in [0,1,0,1]:
            result=command_at(commands,0,np.zeros(5),reference_limit=limit,
                wait_target_error=np.ones(4),waiting_action_mode='error_strength_v1')
            if limit==0:self.assertTrue(np.all(result[1]>0));self.assertEqual(result[3],0)
            else:np.testing.assert_array_equal(result[1],np.zeros(4));self.assertEqual(result[3],1)

    def test_endpoint_wait_and_invalid_mode(self):
        result=command_at(np.zeros((2,4)),2,np.zeros(5),reference_limit=2,
            wait_target_error=np.ones(4),waiting_action_mode='error_strength_v1')
        self.assertEqual(result[3],0);self.assertTrue(np.all(result[1]>0))
        with self.assertRaises(ValueError):command_at(np.zeros((2,4)),0,np.zeros(5),waiting_action_mode='unknown')


if __name__=='__main__':unittest.main()
