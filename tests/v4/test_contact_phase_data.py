import copy,unittest
import numpy as np
from contact_phase_data import align_phase_labels,PHASES


def trace():
    result=[]
    for n in range(1,10):
        old=PHASES[min((n-1)//2,4)];new=PHASES[min(n//2,4)]
        event=None if old==new else dict(transition=old+'_to_'+new,confirmation_frame_end=n,first_available_decision_frame=n+1)
        result.append(dict(frame=n,phase=dict(phase_before_observation=old,phase_for_next_decision=new,event=event,first_available_decision_frame=n+1)))
    return result


class PhaseDataTest(unittest.TestCase):
    def test_confirmation_action_uses_old_phase(self):
        labels=align_phase_labels(np.arange(1,10),trace())
        np.testing.assert_array_equal(labels['phase_id'],[0,0,1,1,2,2,3,3,4])
        np.testing.assert_array_equal(labels['latest_allowed_evidence_frame'],np.arange(9))
        np.testing.assert_array_equal(labels['phase_one_hot'].sum(axis=1),np.ones(9))
        self.assertEqual(labels['terminal_phase'].tolist(),[False]*8+[True])

    def test_action_gaps_do_not_borrow_wrong_row_or_modify_input(self):
        rows=trace();original=copy.deepcopy(rows);f=np.array([4,6,9])
        labels=align_phase_labels(f,rows)
        np.testing.assert_array_equal(labels['phase_id'],[1,2,4])
        self.assertEqual(rows,original);np.testing.assert_array_equal(f,[4,6,9])

    def test_reject_trace_gaps_future_and_skipped_transitions(self):
        variants=[]
        d=trace();del d[2];variants.append(d)
        d=trace();d[1]['phase']['event']['first_available_decision_frame']=2;variants.append(d)
        d=trace();d[1]['phase']['phase_for_next_decision']='curl_load';variants.append(d)
        for d in variants:
            with self.assertRaises(ValueError):align_phase_labels(np.array([1,2]),d)

    def test_reject_uncovered_and_invalid_action_frames(self):
        for f in (np.array([0]),np.array([1,1]),np.array([2,1]),np.array([10]),np.array([1.5])):
            with self.assertRaises(ValueError):align_phase_labels(f,trace())

if __name__=='__main__':unittest.main()
