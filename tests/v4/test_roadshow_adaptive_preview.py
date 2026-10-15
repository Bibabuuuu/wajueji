"""Guard regressions for the bounded roadshow preview search."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/v4'))
from roadshow_adaptive_lift_feedback import choose_preview_factor


class AdaptivePreviewTest(unittest.TestCase):
    def inspect(self, body_delta, *, fine_fails=False, ground_fails=False):
        calls = []

        def run(factor):
            calls.append(factor)
            error = 'preview_infeasible' if fine_fails and .25 < factor < .5 else None
            minimum = .02 if ground_fails and factor > .25 else .1
            delta = body_delta(factor)
            result = dict(factor=factor, passed=error is None and minimum >= .03-1e-6 and delta >= -1e-6,
                          error=error, min_clearance_m=minimum, min_body_delta_m=delta)
            return factor, result

        return run, calls

    def test_fine_factor_passes_before_old_quarter(self):
        run, calls = self.inspect(lambda factor: -4.8e-6 * factor * factor)
        command, safe, factor, search = choose_preview_factor(run, .03)
        self.assertTrue(safe['passed'])
        self.assertEqual(command, factor)
        self.assertTrue(.43 < factor < .45)
        self.assertEqual(calls, [1., .5, factor])
        self.assertEqual(len(search), 3)

    def test_failed_fine_factor_falls_back(self):
        run, calls = self.inspect(lambda factor: -4.8e-6 * factor * factor, fine_fails=True)
        command, safe, factor, search = choose_preview_factor(run, .03)
        self.assertEqual((command, factor), (.25, .25))
        self.assertTrue(safe['passed'])
        self.assertEqual(calls[0:2], [1., .5])
        self.assertEqual(calls[-1], .25)
        self.assertEqual(len(search), 4)

    def test_ground_only_failure_uses_original_fallback(self):
        run, calls = self.inspect(lambda factor: 0., ground_fails=True)
        _, safe, factor, _ = choose_preview_factor(run, .03)
        self.assertEqual(factor, .25)
        self.assertTrue(safe['passed'])
        self.assertEqual(calls, [1., .5, .25])

    def test_no_safe_factor_rejects(self):
        run, calls = self.inspect(lambda factor: 0., ground_fails=True)
        def always_fail(factor):
            _, result = run(factor)
            result['passed'] = False
            return factor, result
        with self.assertRaisesRegex(ValueError, 'feedback_ground_preview_infeasible'):
            choose_preview_factor(always_fail, .03)
        self.assertEqual(calls, [1., .5, .25, .125, 0.])


if __name__ == '__main__':
    unittest.main()
