import json
from pathlib import Path
import tempfile
import unittest

from build_contact_segments import (
    final_false_tail,
    first_confirmed,
    read_index_records,
    transient_contact_uncertainty,
)


class ConfirmationTests(unittest.TestCase):
    def test_first_confirmed_keeps_onset_and_confirmation_distinct(self):
        truth = {0: False, 1: True, 2: False, 3: True, 4: True}
        event = first_confirmed(
            list(truth), lambda step: truth[step], start_step=0,
            confirmation_ticks=2)
        self.assertEqual(event["onset_decision_step"], 3)
        self.assertEqual(event["confirmed_at_decision_step"], 4)

    def test_final_false_tail_ignores_transient_contact_loss(self):
        truth = {
            1: False, 2: True, 3: True, 4: False, 5: True,
            6: True, 7: False, 8: False, 9: False,
        }
        event = final_false_tail(list(truth), lambda step: truth[step])
        self.assertEqual(event["onset_decision_step"], 7)
        self.assertEqual(event["tail_rows"], 3)


class UncertaintyTests(unittest.TestCase):
    def test_transient_gap_is_masked_through_reentry_confirmation(self):
        truth = {
            1: True, 2: True, 3: False, 4: False,
            5: True, 6: True, 7: True, 8: False,
        }
        intervals = transient_contact_uncertainty(
            list(truth), lambda step: truth[step],
            contact_onset=1, off_soil_onset=8)
        self.assertEqual(intervals, [{
            "start_decision_step": 3,
            "end_decision_step": 6,
            "reason": "transient_contact_loss_or_unconfirmed_reentry",
        }])

    def test_unconfirmed_single_tick_reentry_stays_uncertain(self):
        truth = {1: True, 2: True, 3: False, 4: True, 5: False, 6: False}
        intervals = transient_contact_uncertainty(
            list(truth), lambda step: truth[step],
            contact_onset=1, off_soil_onset=5)
        self.assertEqual(intervals[0]["start_decision_step"], 3)
        self.assertEqual(intervals[0]["end_decision_step"], 4)


class ActorIndexBoundaryTests(unittest.TestCase):
    def test_holdout_row_is_rejected_before_episode_paths_are_constructed(self):
        record = {
            "schema": "excavation_v4_actor_data_index_record_v1",
            "episode_id": "human-demo-20260917-101448",
            "decision_step": 0,
            "subphase": "approach",
            "action": {
                "axis_order": ["swing", "boom", "arm", "bucket"],
                "normalized_velocity": [0.0, 0.0, 0.0, 0.0],
            },
            "axis_mask": [1.0, 1.0, 1.0, 1.0],
            "training_eligible": True,
            "holdout": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.ndjson"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe_actor_index"):
                read_index_records(path)


if __name__ == "__main__":
    unittest.main()
