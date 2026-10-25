import unittest
import numpy as np
from fit_response import free_body,fit

class FitTests(unittest.TestCase):
    def test_known_rigid_body_is_recovered(self):
        B=np.array([[.03,.001,0],[.001,.02,.002],[0,.002,.04]])
        C=free_body(.001,B,[.2,-.4,.1]);r=fit(C)
        self.assertLess(r['relative_frobenius_error'],1e-8)
        self.assertTrue(r['physical_parameters'])
    def test_constrained_translations_cannot_be_hidden(self):
        C=np.diag([1.,.001,.2,.1,.03,.07]);r=fit(C)
        self.assertGreater(max(r['relative_column_errors']),.3)
    def test_reference_point_transform_preserves_work(self):
        C=free_body(.002,np.diag([.01,.02,.03]),[1.,-.3,.4])
        self.assertTrue(np.allclose(C,C.T))
        self.assertGreater(np.linalg.eigvalsh(C).min(),0)
    def test_negative_inverse_mass_not_admitted(self):
        r=fit(free_body(-.002,np.diag([.01,.02,.03]),[0,0,0]))
        self.assertFalse(r['physical_parameters'])
if __name__=='__main__':unittest.main()
