import unittest
import numpy as np

from planning.cycle import select_cut,validate_cut


def terrain(height=1.2):
    h=np.full((30,38),height,float)
    return {'valid':True,'origin_xy_m':[3.5,-1.5],'resolution_m':.1,
            'height_m':h,'point_count':np.ones_like(h,int)}


class CutSelectionTests(unittest.TestCase):
    def test_previous_entry_is_excluded_without_cycle_index(self):
        state=terrain();first=select_cut(state)
        second=select_cut(state,excluded_entry_xy=[first['entry_xy_m']])
        self.assertGreaterEqual(np.linalg.norm(np.asarray(first['entry_xy_m'])-second['entry_xy_m']),.58)
        self.assertEqual(second['excluded_entry_count'],1)
        self.assertGreater(second['rejected_history_count'],0)

    def test_unknown_swept_corridor_is_not_a_candidate(self):
        state=terrain(.2)
        # Leave one raised cut patch near the nominal entry but make the deeper
        # swept corridor unknown. A short-patch-only selector would accept it.
        state['height_m'][12:18,16:22]=1.2
        state['point_count'][12:18,10:16]=0
        state['height_m'][12:18,10:16]=np.nan
        with self.assertRaisesRegex(ValueError,'no_observed_reachable_cut_strip'):
            select_cut(state)

    def test_invalid_history_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'invalid_cut_constraints'):
            select_cut(terrain(),excluded_entry_xy=[[np.nan,0.]])

    def test_dispatch_recheck_rejects_new_unknown(self):
        state=terrain();selection=select_cut(state)
        self.assertTrue(validate_cut(state,selection)['valid'])
        state['point_count'][15,15]=0;state['height_m'][15,15]=np.nan
        with self.assertRaisesRegex(ValueError,'cut_selection_invalid:unknown_corridor'):
            validate_cut(state,selection)


if __name__=='__main__':unittest.main()
