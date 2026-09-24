import copy
import math
import unittest
from contact_transport_handoff import CarryHandoffGate
from gate import CONFIG, restore_with_tolerance

class TolerantGateTests(unittest.TestCase):
    def evaluate(self, mode):
        g=CarryHandoffGate(initial_soil_top_m=10.,source_sha256='a'*64,config=CONFIG)
        for f in range(181):
            z=1.+f*.0007+.004*math.sin(f*2*math.pi/15)
            if mode=='stationary':z=1.+.004*math.sin(f*2*math.pi/15)
            if mode=='descending':z=1.-f*.0007+.004*math.sin(f*2*math.pi/15)
            if mode=='large_drop' and f>=100:z-=.03
            r=g.observe_end(frame=f,closure_fraction=.98 if mode=='open' and f==90 else .99,
                origin_z_m=z,minimum_bucket_z_m=z-.5,contact_established=True)
        return g,r

    def test_mm_jitter_does_not_reset_held_closure(self):
        _,r=self.evaluate('rise');self.assertTrue(r['ready']);self.assertEqual(r['observed_span_seconds'],3.)
    def test_stationary_descent_and_large_drop_rejected(self):
        for mode in ('stationary','descending','large_drop'):
            with self.subTest(mode=mode):self.assertFalse(self.evaluate(mode)[1]['ready'])
    def test_actual_reopening_still_restarts_hold(self):
        self.assertFalse(self.evaluate('open')[1]['ready'])
    def test_history_preserved_without_mutation(self):
        g,_=self.evaluate('rise');snap=g.snapshot();original=copy.deepcopy(snap)
        restored=restore_with_tolerance(snap)
        self.assertEqual(snap,original);self.assertEqual(list(restored.rows),list(g.rows))

if __name__=='__main__':unittest.main()
