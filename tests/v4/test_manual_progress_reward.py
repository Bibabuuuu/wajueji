import unittest
from manual_residual_sac_env import progress_reward

class ProgressRewardTests(unittest.TestCase):
    def test_progress_then_success_or_failure_keeps_original_total(self):
        for terminal in [1.45,-2.]:
            raw=[-.001,-.001,-.001,terminal]
            phases=[0.,.1,.7,1.,1.]
            shaped=[progress_reward(x,phases[i],phases[i+1],i==3) for i,x in enumerate(raw)]
            self.assertAlmostEqual(sum(raw),sum(shaped),places=12)
            self.assertAlmostEqual(shaped[-1],terminal-1.,places=12)

    def test_wait_does_not_gain_and_timeout_claws_back(self):
        self.assertAlmostEqual(progress_reward(-.001,.4,.4,False),-.001,places=12)
        self.assertEqual(progress_reward(-2.,.4,.9,True),-2.4)

    def test_unfinished_suffix_keeps_remaining_potential(self):
        self.assertAlmostEqual(progress_reward(-.001,0.,.2,False),.199)
