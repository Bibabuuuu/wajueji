import unittest
from run_contact_geometric_probe import probe_budget


class ProbeBudgetTest(unittest.TestCase):
    def test_legacy_budget_is_unchanged(self):
        self.assertEqual(probe_budget(900, True), 360)
        with self.assertRaises(ValueError): probe_budget(901, True)
        with self.assertRaises(ValueError): probe_budget(121, False)

    def test_explicit_slow_handoff_is_bounded(self):
        self.assertEqual(probe_budget(1200, True, handoff_review_frames=1200), 360)
        with self.assertRaises(ValueError): probe_budget(1201, True, handoff_review_frames=1200)
        for invalid in (0, 2400, 1200.0, True):
            with self.assertRaises(ValueError): probe_budget(120, False, handoff_review_frames=invalid)
        self.assertEqual(probe_budget(2400, True, True, 1200), 900)
        with self.assertRaises(ValueError): probe_budget(2401, True, True, 1200)


if __name__ == '__main__': unittest.main()
