import unittest
import numpy as np
from static_rocks import static_rock_ids

def box(low, high):
    return np.array([low, high], float)

class StaticRockTests(unittest.TestCase):
    def test_ground_soil_and_supported_stack_retained(self):
        rocks=dict(ground=box([0,0,.05],[1,1,.5]),
                   stacked=box([0,0,.51],[1,1,1]),
                   soil=box([2,0,.9],[3,1,1.2]),
                   airborne=box([4,0,3],[5,1,3.3]))
        static,_=static_rock_ids(np.array([[2.5,.5,.89]]),rocks,.1)
        self.assertEqual(static,{'ground','stacked','soil'})

    def test_airborne_cluster_not_static(self):
        rocks=dict(a=box([4,0,3],[5,1,3.3]),b=box([4,0,3.31],[5,1,3.6]))
        self.assertFalse(static_rock_ids(np.array([[0,0,0]]),rocks,.1)[0])

    def test_unobserved_ground_is_rejected(self):
        with self.assertRaises(ValueError):
            static_rock_ids(np.empty((0,3)),{},.1)

if __name__=='__main__':unittest.main()
