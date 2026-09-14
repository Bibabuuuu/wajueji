import unittest
import numpy as np
from work_envelope import WorkEnvelope

class DomainTests(unittest.TestCase):
    def setUp(self):self.d=WorkEnvelope(np.array([[[1.13,1.13,0],[6.06,6.383,2.613],[3,3,1]]]))
    def test_real_unloading_points_inside(self):
        self.assertFalse(self.d.outside(np.array([[3.658,4.0053,3.2754],[3.681,4.0093,3.2989]])).any())
    def test_old_box_remains(self):
        self.assertFalse(self.d.outside(np.array([[1,-4,-.05],[9,4,5]])).any())
    def test_domain_is_union_not_large_rectangle(self):
        self.assertTrue(self.d.outside(np.array([[8,5,1],[3,6.4,1],[.9,0,1],[4,4.5,5.01],[4,4.5,-.051]])).all())
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.d.outside(np.array([[3,np.nan,1]]))
if __name__=='__main__':unittest.main()
