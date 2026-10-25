import copy
import unittest
from contact_transport_handoff import CarryHandoffGate


class HandoffTests(unittest.TestCase):
    def make(self):
        return CarryHandoffGate(initial_soil_top_m=1., source_sha256='a'*64)

    def row(self, f, **kw):
        return dict(frame=f, closure_fraction=.99, origin_z_m=2.+f*.001,
                    minimum_bucket_z_m=1.1+f*.001, contact_established=True, **kw)

    def test_exact_duration_and_next_decision(self):
        g=self.make()
        for f in range(180):
            self.assertFalse(g.observe_end(**self.row(f))['ready'])
        r=g.observe_end(**self.row(180))
        self.assertTrue(r['ready']);self.assertEqual(r['observed_span_seconds'],3.0)
        self.assertEqual(r['first_available_decision_frame'],181)

    def test_single_low_closure_invalidates_window(self):
        g=self.make()
        for f in range(181):
            row=self.row(f)
            if f==45:row['closure_fraction']=.98999
            r=g.observe_end(**row)
        self.assertFalse(r['closure_held']);self.assertFalse(r['ready'])

    def test_stationary_and_actual_descent_rejected(self):
        for dz in (0.,-.001):
            g=self.make()
            for f in range(181):
                row=self.row(f);row['origin_z_m']=2.+f*dz
                r=g.observe_end(**row)
            self.assertFalse(r['ready'])

    def test_bucket_bottom_not_origin_is_only_diagnostic(self):
        g=self.make()
        for f in range(181):
            row=self.row(f);row['minimum_bucket_z_m']=1.+f*.0005
            r=g.observe_end(**row)
        self.assertTrue(r['sustained_actual_rise']);self.assertTrue(r['ready']);self.assertLess(r['bucket_above_initial_soil_m'],.1)

    def test_small_jitter_allowed_but_stall_not(self):
        for stall in (False,True):
            g=self.make()
            for f in range(181):
                row=self.row(f)
                row['origin_z_m']=2.+min(f,40 if stall else 180)*.001+(-.0006 if f%2 else 0)
                r=g.observe_end(**row)
            self.assertEqual(r['ready'],not stall)

    def test_no_contact_does_not_qualify(self):
        g=self.make()
        for f in range(181):
            row=self.row(f);row['contact_established']=False;r=g.observe_end(**row)
        self.assertFalse(r['ready'])

    def test_bad_or_missing_observations_atomic(self):
        g=self.make();g.observe_end(**self.row(0));before=copy.deepcopy(g.snapshot())
        for row in (self.row(0),self.row(2),dict(self.row(1),origin_z_m=float('nan'))):
            with self.assertRaises(ValueError):g.observe_end(**row)
            self.assertEqual(g.snapshot(),before)

    def test_resume_same_history_and_readiness_not_latched(self):
        g=self.make()
        for f in range(181):g.observe_end(**self.row(f))
        h=CarryHandoffGate.restore(g.snapshot())
        row=dict(self.row(181),closure_fraction=.7)
        self.assertEqual(g.observe_end(**row),h.observe_end(**row))
        self.assertFalse(g.observe_end(**dict(self.row(182),closure_fraction=.7))['ready'])


if __name__=='__main__':unittest.main()
