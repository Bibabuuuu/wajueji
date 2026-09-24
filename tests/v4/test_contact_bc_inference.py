import unittest
import numpy as np
from contact_bc_inference import OnlineFeatures,structured_observation
from contact_bc_policy import encode_episode
from test_contact_bc_policy import data

class LiveInferenceTest(unittest.TestCase):
    def test_streaming_equals_batch_every_frame(self):
        a=data();a['sequence_id'][6:]=2;a['frame'][6:]+=1
        expected,_,digest=encode_episode(a)
        live=OnlineFeatures()
        for i in range(len(a['frame'])):
            x,actual=live.update({k:v[i] for k,v in a.items()})
            self.assertEqual(actual,digest);np.testing.assert_array_equal(x,expected[i])
    def test_probe_summary_json_serializable_after_numpy_actions(self):
        import json
        from run_contact_bc_probe import execution_summary
        events=[dict(command_projection_reason=None,applied_command_si=[0.,.01,0.,0.]),
                dict(command_projection_reason='stop',applied_command_si=[0.]*4)]
        summary=execution_summary(events,[.001,.002],[dict(hard_guard=True)])
        restored=json.loads(json.dumps(summary,allow_nan=False))
        self.assertEqual(restored['command_nonzero_frames'],1)
        self.assertEqual(restored['projection_frames'],1)
        self.assertTrue(restored['all_executed_frames_safe'])
        json.dumps(execution_summary([],[],[]),allow_nan=False)

    def test_repeated_frame_rejected(self):
        a=data();row={k:v[0] for k,v in a.items()};live=OnlineFeatures();live.update(row)
        with self.assertRaisesRegex(ValueError,'nonmonotonic'):live.update(row)
    def test_true_initial_grid_allowed_but_future_and_stale_rejected(self):
        from contact_bc_data import STATE_FIELDS
        state={k:[0.]*size for k,size in STATE_FIELDS.items()}
        terrain=dict(height_m=[[0.,None]],shape_yx=[1,2],known_cell_count=1,origin_xy_m=[0.,0.],resolution_m=.1)
        args=dict(frame=1,state=state,terrain=terrain,terrain_frame=0,goal=dict(entry_xy_m=[1.,2.],surface_z_m=0.),previous_action=[0.]*4,previous_load=None)
        x=structured_observation(**args);self.assertFalse(x['previous_drive_load_valid'])
        for change in (dict(terrain_frame=1),dict(frame=4)):
            with self.assertRaisesRegex(ValueError,'noncausal_or_stale'):structured_observation(**{**args,**change})

if __name__=='__main__':unittest.main()
