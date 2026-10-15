"""Direct proportional probe and old-ladder fallback behavior."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/v4'))
from roadshow_adaptive_lift_feedback_v2 import choose_preview_factor


def simulated_preview(dip_at_full, *, reject_estimate=False):
    calls = []

    def inspect(factor):
        calls.append(factor)
        delta = -dip_at_full * factor * factor
        failed_estimate = reject_estimate and factor not in (1., .5, .25, .125, 0.)
        result = dict(factor=factor, passed=delta >= -1e-6 and not failed_estimate,
                      error='estimate_rejected' if failed_estimate else None,
                      min_clearance_m=.1, min_body_delta_m=delta)
        return factor, result

    return inspect, calls


class DirectProportionalPreviewTest(unittest.TestCase):
    def test_severe_corner_dip_reaches_safe_fine_factor_in_two_previews(self):
        inspect, calls = simulated_preview(155e-6)
        command, result, factor, search = choose_preview_factor(inspect, .03)
        self.assertTrue(result['passed'])
        self.assertEqual(command, factor)
        self.assertTrue(.07 < factor < .09)
        self.assertEqual(calls, [1., factor])
        self.assertEqual(len(search), 2)

    def test_failed_estimate_falls_back_to_old_ladder(self):
        inspect, calls = simulated_preview(4.8e-6, reject_estimate=True)
        _, result, factor, search = choose_preview_factor(inspect, .03)
        self.assertEqual(factor, .25)
        self.assertTrue(result['passed'])
        self.assertTrue(.43 < calls[1] < .45)
        self.assertEqual(calls[2:], [.5, .25])
        self.assertEqual(len(search), 4)

    def test_original_full_speed_unaffected(self):
        inspect, calls = simulated_preview(.5e-6)
        _, result, factor, _ = choose_preview_factor(inspect, .03)
        self.assertTrue(result['passed'])
        self.assertEqual(factor, 1.)
        self.assertEqual(calls, [1.])


if __name__ == '__main__':
    unittest.main()
