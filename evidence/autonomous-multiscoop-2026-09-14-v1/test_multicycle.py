import unittest
import numpy as np

from planning.cycle import select_cut,validate_cut
from retention_observer import RetentionObserver
from planning_retry import replan_allowed


def terrain(height=1.2):
    h=np.full((30,38),height,float)
    return {'valid':True,'origin_xy_m':[3.5,-1.5],'resolution_m':.1,
            'height_m':h,'point_count':np.ones_like(h,int)}


class MultiCycleTests(unittest.TestCase):
    def test_reobservation_excludes_prior_entry(self):
        state=terrain();first=select_cut(state)
        second=select_cut(state,excluded_entry_xy=[first['entry_xy_m']])
        self.assertGreaterEqual(np.linalg.norm(np.asarray(first['entry_xy_m'])-second['entry_xy_m']),.58)

    def test_dispatch_recheck_rejects_unknown_corridor(self):
        state=terrain();selection=select_cut(state)
        self.assertTrue(validate_cut(state,selection)['valid'])
        state['point_count'][15,15]=0;state['height_m'][15,15]=np.nan
        with self.assertRaisesRegex(ValueError,'unknown_corridor'):validate_cut(state,selection)

    def test_acquisition_cohort_excludes_starting_residual(self):
        observer=RetentionObserver.__new__(RetentionObserver)
        observer.masses=np.array([2.,3.,5.]);observer.cohorts={
            'cycle-01-start':np.array([True,False,False]),
            'cycle-01-hold-end':np.array([True,True,False])}
        self.assertEqual(observer.derive_cohort('cycle-01-acquired','cycle-01-hold-end','cycle-01-start'),3.)
        np.testing.assert_array_equal(observer.cohorts['cycle-01-acquired'],[False,True,False])

    def test_dispatch_observation_can_trigger_bounded_replan(self):
        self.assertTrue(replan_allowed(ValueError('cut_selection_invalid:unknown_corridor'),2,1))
        self.assertTrue(replan_allowed(ValueError('cut_selection_invalid:surface_changed'),3,2))
        self.assertFalse(replan_allowed(ValueError('cut_selection_invalid:unknown_corridor'),1,1))
        self.assertFalse(replan_allowed(ValueError('cut_selection_invalid:unknown_corridor'),2,3))
        self.assertFalse(replan_allowed(ValueError('bucket_bank_clearance'),2,1))


if __name__=='__main__':unittest.main()
