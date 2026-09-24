"""Causality and missing-data regressions for contact BC staging."""
import copy
import unittest
import numpy as np
from contact_bc_data import STATE_FIELDS, encode_row, terrain_arrays


def fixture():
    state={k:list(range(n)) for k,n in STATE_FIELDS.items()}
    physics=[dict(frame=f,before_state=copy.deepcopy(state),after_state={'future':1},
                  applied_command_si=[.06,0,0,0],requested_command_si=[.06,0,0,0],equivalent_drive_load=[float(f)]*4) for f in range(1,8)]
    terrain=dict(height_m=[[1.,None],[2.,3.]],shape_yx=[2,2],known_cell_count=3,origin_xy_m=[0.,0.],resolution_m=.1)
    observations={3:dict(terrain=copy.deepcopy(terrain)),6:dict(terrain=copy.deepcopy(terrain))}
    row=dict(frame=5,terrain_anchor_frame=3,terrain_age_frames=1,tensor_candidate=True,terrain_grid_available=True,
             previous_applied_action_si=[.06,0,0,0],recorded_goal=dict(entry_xy_m=[1.,2.],surface_z_m=3.),
             requested_action_normalized=[.5,0,0,0],applied_action_normalized=[.5,0,0,0])
    return row,physics,observations


class ContactBCDataTest(unittest.TestCase):
    def test_future_changes_do_not_change_current_features(self):
        row,p,o=fixture();before,_=encode_row(row,p,o,previous_frame=4,sequence_id=1)
        p[4]['after_state']={'leak':999};p[4]['equivalent_drive_load']=[999.]*4
        o[6]['terrain']['height_m']=[[999.]*2]*2
        after,_=encode_row(row,p,o,previous_frame=4,sequence_id=1)
        for key in before:np.testing.assert_array_equal(before[key],after[key])
        np.testing.assert_array_equal(after['previous_drive_load'],[4.]*4)

    def test_same_frame_terrain_is_rejected(self):
        row,p,o=fixture();row.update(frame=6,terrain_anchor_frame=6,terrain_age_frames=-1)
        with self.assertRaisesRegex(ValueError,'terrain_not_prior'):encode_row(row,p,o,previous_frame=5,sequence_id=1)

    def test_missing_initial_grid_cannot_be_encoded(self):
        with self.assertRaisesRegex(ValueError,'terrain_grid_missing'):terrain_arrays({'shape_yx':[2,2]})
        row,p,o=fixture();row.update(frame=1,terrain_anchor_frame=0,tensor_candidate=False,terrain_grid_available=False)
        with self.assertRaisesRegex(ValueError,'not_tensor_candidate'):encode_row(row,p,o,previous_frame=None,sequence_id=0)

    def test_control_gap_resets_history_but_keeps_actual_previous_action(self):
        row,p,o=fixture();value,sequence=encode_row(row,p,o,previous_frame=2,sequence_id=1)
        self.assertEqual(sequence,2);self.assertTrue(value['reset_learning_history']);self.assertFalse(value['previous_drive_load_valid'])
        np.testing.assert_array_equal(value['previous_drive_load'],[0.]*4)
        np.testing.assert_allclose(value['previous_applied_action_si'],[.06,0,0,0])

    def test_unknown_terrain_is_masked_not_known_zero(self):
        row,p,o=fixture();value,_=encode_row(row,p,o,previous_frame=4,sequence_id=1)
        self.assertFalse(value['terrain_known'][0,1]);self.assertEqual(value['terrain_height_m'][0,1],0)
        self.assertTrue(value['terrain_known'][0,0])

    def test_request_and_execution_remain_separate(self):
        row,p,o=fixture();p[4]['requested_command_si']=[.12,0,0,.0675];row['requested_action_normalized']=[1,0,0,1]
        value,_=encode_row(row,p,o,previous_frame=4,sequence_id=1)
        np.testing.assert_array_equal(value['requested_action_normalized'],[1,0,0,1])
        np.testing.assert_array_equal(value['applied_action_normalized'],[.5,0,0,0])

    def test_bad_actions_or_unknown_count_rejected(self):
        row,p,o=fixture();p[4]['applied_command_si'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'action_bounds'):encode_row(row,p,o,previous_frame=4,sequence_id=1)
        row,p,o=fixture();o[3]['terrain']['known_cell_count']=4
        with self.assertRaisesRegex(ValueError,'terrain_known_count'):encode_row(row,p,o,previous_frame=4,sequence_id=1)

if __name__=='__main__':unittest.main()
