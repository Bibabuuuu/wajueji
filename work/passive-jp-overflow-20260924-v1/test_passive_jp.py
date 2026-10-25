"""CPU-only scope/lifecycle checks; physical equivalence needs a separate run."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
import warp as wp
from passive_jp_candidate import installed_passive_jp, validate_scope
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import hardening_law


@wp.kernel
def evaluate_hardening(jp: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = hardening_law(jp[i], 0.0)


def state(values):
    return NS(mpm=NS(particle_Jp=wp.array(values,dtype=wp.float32,device='cpu')))


class CandidateTest(unittest.TestCase):
    def setUp(self):
        self.values = np.array([.25, 1., 1e30, 3e38],dtype=np.float32)
        self.source = state(self.values)
        self.rates = wp.zeros(4,dtype=wp.float32,device='cpu')
        self.solver = NS(_mpm_model=NS(material_parameters=NS(hardening=self.rates)))
        self.seen = []
        def original(a,b,*_):
            self.seen.append(a.mpm.particle_Jp.numpy().copy())
            b.mpm.particle_Jp.fill_(2.)
            return 'updated'
        self.original = original
        self.solver._update_particles = original

    def check_alias(self,alias):
        target=self.source if alias else state([9.]*4)
        with installed_passive_jp(self.solver,self.source) as log:
            for _ in range(3):
                self.assertEqual(self.solver._update_particles(self.source,target,1/60,None,None),'updated')
                np.testing.assert_array_equal(self.source.mpm.particle_Jp.numpy(),self.values)
                np.testing.assert_array_equal(target.mpm.particle_Jp.numpy(),self.values)
            self.assertEqual(log['updates'],3)
        for a in self.seen:np.testing.assert_array_equal(a,np.ones(4))
        self.assertIs(self.solver._update_particles,self.original)

    def test_aliasing(self):self.check_alias(True)
    def test_distinct_states(self):self.check_alias(False)

    def test_exception_preserves_source_and_restores_method(self):
        def broken(a,b,*_):
            b.mpm.particle_Jp.fill_(float('inf'))
            raise RuntimeError('original failure')
        self.solver._update_particles=broken
        target=state([1.]*4)
        with self.assertRaisesRegex(RuntimeError,'original failure'):
            with installed_passive_jp(self.solver,self.source):
                self.solver._update_particles(self.source,target,1/60,None,None)
        np.testing.assert_array_equal(self.source.mpm.particle_Jp.numpy(),self.values)
        self.assertIs(self.solver._update_particles,broken)

    def test_scope_nonzero_and_nonfinite_hardening(self):
        for rates in ([0,0,.01,0],[0,0,-.01,0],[0,0,np.nan,0]):
            with self.assertRaisesRegex(ValueError,'exactly_zero'):validate_scope(rates,self.values)

    def test_bad_source_rejected_not_repaired(self):
        for values in ([1,1,np.inf,1],[1,1,np.nan,1],[1,1,0,1],[1,1,-1,1]):
            with self.assertRaisesRegex(ValueError,'finite_positive'):validate_scope([0]*4,values)

    def test_changed_scope_rejected_before_update(self):
        with installed_passive_jp(self.solver,self.source):
            self.rates.fill_(.1)
            with self.assertRaisesRegex(ValueError,'exactly_zero'):
                self.solver._update_particles(self.source,self.source,1/60,None,None)
        self.assertEqual(len(self.seen),0)

    def test_nested_install_rejected(self):
        with installed_passive_jp(self.solver,self.source):
            with self.assertRaisesRegex(RuntimeError,'already_installed'):
                with installed_passive_jp(self.solver,self.source):pass

    def test_native_zero_hardening_law_independent_of_Jp(self):
        out=wp.zeros(4,dtype=wp.float32,device='cpu')
        wp.launch(evaluate_hardening,dim=4,inputs=[self.source.mpm.particle_Jp,out],device='cpu')
        np.testing.assert_array_equal(out.numpy(),np.ones(4))

    def test_finite_float32_multiplication_can_overflow(self):
        with np.errstate(over='ignore'):
            value=np.float32(1.5878933e38)*np.float32(3.)
        self.assertFalse(np.isfinite(value))


if __name__=='__main__':unittest.main()
