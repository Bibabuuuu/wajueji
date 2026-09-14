import unittest
import numpy as np
from planning.reachability import search_endpoint


class ReachabilityTests(unittest.TestCase):
    def solve(self,target,guard=lambda q:None,seeds=None):
        return search_endpoint(target,seeds if seeds is not None else [[.2,.2,.2,9.]],
            point_world=lambda q:q[:3],closure=lambda q:np.array([q[0]-q[1]]),
            variable_indices=[0,1,2],limits=[[0,1]]*3,check_pose=guard)

    def test_constraint_solution_preserves_fixed_base(self):
        seed=np.array([.2,.2,.2,9.]);result=self.solve([.6,.6,.4],seeds=[seed])
        self.assertEqual(result['status'],'endpoint_witness');self.assertEqual(result['qpos'][3],9.)
        np.testing.assert_array_equal(seed,[.2,.2,.2,9.]);self.assertIsNone(result['command']);self.assertFalse(result['path_checked'])

    def test_position_alone_does_not_override_closure(self):
        result=self.solve([.8,.1,.4]);self.assertEqual(result['status'],'no_witness_found')
        self.assertFalse(result['unreachable_proven'])

    def test_guard_rejection_and_outside_ranges_cannot_pass(self):
        def reject(q):raise ValueError('collision')
        self.assertEqual(self.solve([.6,.6,.4],reject)['status'],'no_witness_found')
        self.assertEqual(self.solve([4.,4.,4.])['status'],'no_witness_found')

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):self.solve([np.nan,0,0])

if __name__=='__main__':unittest.main()
