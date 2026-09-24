import unittest
from collections import deque
import numpy as np
from tests.v4.test_contact_phase_bc_policy import example
from contact_phase_bc_policy import encode_episode
from contact_phase_bc_inference import OnlinePhaseFeatures, with_phase
from contact_phase_live import LivePhaseObserver


class PhaseInferenceTest(unittest.TestCase):
    def test_streaming_matches_batch_across_gaps_and_goal_mask(self):
        arrays = example()
        arrays['frame'][6:] += 2
        arrays['sequence_id'][6:] = 2
        arrays['latest_allowed_phase_evidence_frame'] = arrays['frame']-1
        for goal_valid in (True, False):
            expected, _, digest = encode_episode(arrays, goal_valid=goal_valid)
            live = OnlinePhaseFeatures(goal_valid=goal_valid)
            for i in range(len(arrays['frame'])):
                x, actual = live.update({k: v[i] for k, v in arrays.items()})
                self.assertEqual(actual, digest)
                np.testing.assert_array_equal(x, expected[i])

    def test_invalid_phase_does_not_consume_history_frame(self):
        arrays = example()
        row = {k: v[0] for k, v in arrays.items()}
        bad = dict(row, latest_allowed_phase_evidence_frame=row['frame'])
        live = OnlinePhaseFeatures()
        with self.assertRaisesRegex(ValueError, 'noncausal'):
            live.update(bad)
        self.assertIsNone(live.last_frame)
        live.update(row)
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'):
            live.update(row)

    def test_phase_injection_rejects_same_frame_and_unknown(self):
        row = dict(frame=np.int64(7))
        result = with_phase(row, 'curl_load', 6)
        self.assertEqual(result['phase_id'], 2)
        self.assertNotIn('phase_id', row)
        for phase, evidence in [('curl_load', 7), ('unknown', 6), ('lift', 6.0)]:
            with self.assertRaisesRegex(ValueError, 'noncausal'):
                with_phase(row, phase, evidence)

    def test_geometry_only_after_lift_and_timestamp_is_current_end(self):
        class Router:
            def phase_for_decision(self, frame): return self.phase
            def observe_end(self, **kwargs): self.received = kwargs; return kwargs
        class Geometry:
            def observe(self, particles, pose): return dict(remaining_soil_gap_m=.004)
        live = LivePhaseObserver.__new__(LivePhaseObserver)
        live.polygon = [[0., 0.], [0.1, -0.2]]
        live.router, live.geometry = Router(), Geometry()
        calls = []
        def particles(): calls.append(True); return np.zeros((1, 3))
        for phase in ('approach_contact', 'penetrate', 'curl_load', 'lift', 'off_soil'):
            live.router.phase = phase
            result = live.observe_end(frame=7, state={'mechanism_qpos': [0.]*19}, pose=np.eye(4),
                                      soil_force_n=12., rock_contact_substeps=0, particles=particles)
            self.assertEqual(result['phase']['gap_frame_end'], 7 if phase in ('lift', 'off_soil') else None)
        self.assertEqual(len(calls), 2)

    def test_probe_keeps_bounded_windows(self):
        from run_contact_phase_bc_probe import probe_budget
        self.assertEqual(probe_budget(120, False), 180)
        self.assertEqual(probe_budget(600, True), 240)
        with self.assertRaises(ValueError): probe_budget(601, True)


if __name__ == '__main__':
    unittest.main()
