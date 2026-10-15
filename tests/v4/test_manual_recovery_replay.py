import unittest
import numpy as np
from prepare_manual_recovery_replay import select_old_indices


class SelectionTests(unittest.TestCase):
    def test_baseline_failures_unique_and_deterministic(self):
        dones=np.zeros(11084);dones[[399,700,1200,11000]]=1
        chosen=select_old_indices(dones)
        self.assertEqual(len(chosen),1350)
        self.assertEqual(len(set(chosen)),1350)
        self.assertTrue(set(range(400)).issubset(chosen))
        self.assertTrue(set(np.flatnonzero(dones)).issubset(chosen))
        np.testing.assert_array_equal(chosen,select_old_indices(dones))

    def test_unqualified_baseline_rejected(self):
        with self.assertRaises(ValueError):select_old_indices(np.zeros(2000))

    def test_no_silent_terminal_dropping(self):
        with self.assertRaises(ValueError):select_old_indices(np.ones(2000))


if __name__=='__main__':unittest.main()
