import unittest
import numpy as np
from manual_residual_sac_env import command_at, retained_score, LIMITS_SI


class ResidualPaceTests(unittest.TestCase):
    def test_zero_command_and_original_pace_exact(self):
        commands=np.array([[-0.,.01,-.02,.06],[.03,0,0,-.04]])
        base,delta,request,pace=command_at(commands,0,np.zeros(5))
        self.assertEqual(request.tobytes(),commands[0].tobytes())
        self.assertEqual(pace,1.)
        np.testing.assert_array_equal(delta,np.zeros(4))

    def test_fractional_reference_and_pace_bounds(self):
        commands=np.array([[0.,0,0,0],[.1,.02,-.02,.06]])
        base,delta,request,pace=command_at(commands,.5,[0,0,0,0,0])
        np.testing.assert_array_equal(base,[.05,.01,-.01,.03])
        self.assertEqual(pace,1.)
        self.assertEqual(command_at(commands,.5,[0,0,0,0,-1])[3],.5)
        self.assertEqual(command_at(commands,.5,[0,0,0,0,1])[3],1.5)
        for cursor,action in [(-1,np.zeros(5)),(2,np.zeros(5)),(0,np.ones(4)),(0,[0,0,0,0,float('nan')])]:
            with self.assertRaises(ValueError):command_at(commands,cursor,action)

    def test_reward_uses_lowest_hold_score_and_preserves_invalidity(self):
        valid=[dict(valid_estimate=True,fill_percent=x) for x in [150.,145.,148.]]
        self.assertEqual(retained_score(valid),1.45)
        valid[1]=dict(valid_estimate=False,fill_percent=None)
        self.assertIsNone(retained_score(valid))
        self.assertIsNone(retained_score([]))

    def test_slow_retiming_preserves_reference_displacement_without_early_command_blending(self):
        commands=np.array([[.06,.01,-.01,.03],[-.06,-.01,.01,-.03]])
        first=command_at(commands,0,[0,0,0,0,-1])
        second=command_at(commands,first[3],[0,0,0,0,-1])
        np.testing.assert_array_equal(first[0]+second[0],commands[0])
        np.testing.assert_array_equal(first[0],commands[0]*.5)

    def test_fast_retiming_respects_speed_limit_and_remaining_reference(self):
        saturated=np.array([LIMITS_SI,LIMITS_SI])
        base,_,_,pace=command_at(saturated,0,[0,0,0,0,1])
        self.assertEqual(pace,1.)
        np.testing.assert_array_equal(base,LIMITS_SI)
        slow=saturated*.25
        base,_,_,pace=command_at(slow,0,[0,0,0,0,1])
        self.assertEqual(pace,1.5)
        np.testing.assert_array_equal(base,slow[0]*1.5)
        base,_,_,pace=command_at(slow,1.75,[0,0,0,0,1])
        self.assertEqual(pace,.25)
        np.testing.assert_array_equal(base,slow[-1]*.25)


if __name__=='__main__':unittest.main()
