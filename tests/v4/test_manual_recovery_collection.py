import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from collect_manual_recovery import recovery_action,save_transitions


class RecoveryCollectionTests(unittest.TestCase):
    def test_only_waiting_uses_goal_direction_and_keeps_neutral_pace(self):
        error=[.1,-.1,0.,.2]
        np.testing.assert_array_equal(recovery_action('wait',error),[1,-1,0,1,0])
        np.testing.assert_array_equal(recovery_action('wait_endpoint',error),[1,-1,0,1,0])
        np.testing.assert_array_equal(recovery_action('follow',error),np.zeros(5))
        np.testing.assert_array_equal(recovery_action('wait',error,True),np.zeros(5))
        with self.assertRaises(ValueError):recovery_action('wait',[float('nan')]*4)

    def test_export_preserves_terminal_and_reward_identity(self):
        a=np.zeros(221,np.float32);b=a.copy();b[-2]=.25;c=b.copy()
        transitions=[(a,b,np.zeros(5,np.float32),.24,False,{'raw_reward':-.01}),
                     (b,c,np.ones(5,np.float32),-2.25,True,{'raw_reward':-2.})]
        rows=[dict(frame_before=0,frame_after=3,policy_source='manual_follow',terminated=False,truncated=False),
              dict(frame_before=3,frame_after=6,policy_source='fixed_correction',terminated=False,truncated=True)]
        with tempfile.TemporaryDirectory() as directory:
            result=save_transitions(SimpleNamespace(transitions=transitions),rows,Path(directory)/'data.npz')
            self.assertTrue(result['complete']);self.assertEqual(result['terminal_count'],1)
            self.assertAlmostEqual(result['total_reward_difference'],0.,places=12)
            with np.load(Path(directory)/'data.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['dones'],[False,True])
                np.testing.assert_array_equal(data['truncated'],[False,True])
                self.assertEqual(data['observations'].shape,(2,221))


if __name__=='__main__':unittest.main()
