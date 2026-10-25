"""Keyboard input must stay gated and retain the pre-gate operator action."""

import unittest
from types import SimpleNamespace

import numpy as np

from roadshow_d152_manual_worker import ManualWorker


class KeyboardWorkerTests(unittest.TestCase):
    def test_neutral_rearm_and_focus_gate(self):
        worker = ManualWorker(None, None, None)
        worker.bind(object(), None, {}, mode='zero', device='cuda:0')
        worker.keyboard.press('M')
        output, _ = worker.predict(None, None)
        self.assertTrue(np.array_equal(output, np.zeros(4)))
        self.assertEqual(worker.last_record['operator_action_normalized_before_gate'],
                         [0.0, 0.0, 0.0, 1.0])
        worker.keyboard.release('M')
        worker.predict(None, None)
        worker.keyboard.press('M')
        output, _ = worker.predict(None, None)
        self.assertEqual(output.tolist(), [0.0, 0.0, 0.0, 1.0])
        worker.focus = False
        output, _ = worker.predict(None, None)
        self.assertTrue(np.array_equal(output, np.zeros(4)))
        self.assertFalse(worker.last_record['operator_gate_armed'])

    def test_step_selection_is_latched_and_replaces_opposite_direction(self):
        worker = ManualWorker(None, None, None)
        worker.bind(SimpleNamespace(frames=394), SimpleNamespace(lift_intent=False),
                    {}, mode='zero', device='cuda:0', step_frames=15)
        worker.select_step_key('O')
        worker.select_step_key('M')
        self.assertEqual(worker.keyboard.action(window_focused=True),
                         [0.0, 0.0, 1.0, 1.0])
        worker.select_step_key('U')
        self.assertEqual(worker.keyboard.action(window_focused=True),
                         [0.0, 0.0, -1.0, 1.0])
        worker.select_step_key('U')
        self.assertEqual(worker.keyboard.action(window_focused=True),
                         [0.0, 0.0, 0.0, 1.0])

    def test_step_action_stays_fixed_for_horizon_and_records_one_decision(self):
        class FakeViewer:
            _paused = False

            def is_paused(self):
                return self._paused

            def is_running(self):
                return True

        runtime = SimpleNamespace(frames=394)
        worker = ManualWorker(None, None, None)
        worker.bind(runtime, SimpleNamespace(lift_intent=False), {},
                    mode='zero', device='cuda:0', step_frames=15)
        worker.mode = 'step_keyboard'
        worker.phase = 'curl_load'
        worker.viewer = FakeViewer()
        worker._draw = lambda: None
        worker.select_step_key('M')
        worker.begin_step()
        stamps = set()
        for frame in range(395, 410):
            runtime.frames = frame - 1
            action, _ = worker.predict(None, None)
            self.assertEqual(action.tolist(), [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(worker.last_record['operator_decision_frame'], 395)
            self.assertEqual(worker.last_record['operator_step_id'], 1)
            stamps.add(worker.last_record['operator_input_wall_monotonic_s'])
        self.assertEqual(worker.step_remaining, 0)
        self.assertEqual(len(stamps), 1)


if __name__ == '__main__':
    unittest.main()
