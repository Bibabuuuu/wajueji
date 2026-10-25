import unittest
import numpy as np
from fit_response import free_body
from fit_physical import physical_fit

class PhysicalFitTests(unittest.TestCase):
    def test_known_positive_rigid_body_admitted(self):
        C=free_body(.002,np.diag([.003,.004,.005]),[.2,-.1,.3])
        r=physical_fit(C)['best']
        self.assertTrue(r['passed'])
        self.assertLess(r['relative_frobenius_error'],1e-5)
        self.assertGreater(r['mass_kg'],0)
        self.assertGreaterEqual(min(r['inverse_inertia_eigenvalues']),0)
    def test_no_negative_inertia_in_any_candidate(self):
        C=np.diag([.02,.0001,.03,.001,.002,.0001])
        for r in physical_fit(C)['candidates']:
            self.assertGreater(r['mass_kg'],0)
            self.assertGreaterEqual(min(r['inverse_inertia_eigenvalues']),-1e-12)
if __name__=='__main__':unittest.main()
