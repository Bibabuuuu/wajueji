import unittest
import numpy as np
from terrain.material_ledger import (MaterialLedger, exclusive_regions, SOURCE,
    BUCKET, TRUCK, UNASSIGNED, OUTSIDE, AMBIGUOUS)


class LedgerTests(unittest.TestCase):
    def test_reentry_and_cycle_net_do_not_double_credit_delivery(self):
        ledger = MaterialLedger([0, 1, 2], [2., 3., 5.])
        ledger.update(0, [0, 1, 2], [BUCKET, BUCKET, TRUCK])
        ledger.begin_cycle('first')
        ledger.update(1, [0, 1, 2], [TRUCK, UNASSIGNED, TRUCK])
        ledger.update(2, [0, 1, 2], [UNASSIGNED, UNASSIGNED, TRUCK])
        result = ledger.update(3, [0, 1, 2], [TRUCK, UNASSIGNED, UNASSIGNED])
        self.assertEqual(result['truck']['current_stock_kg'], 2.)
        self.assertEqual(result['truck']['sampled_entries_cumulative_kg'], 4.)
        self.assertEqual(result['truck']['sampled_exits_cumulative_kg'], 7.)
        self.assertEqual(result['truck']['first_observed_entries_kg'], 2.)
        cycle = ledger.end_cycle()
        self.assertEqual(cycle['truck_net_stock_change_kg'], -3.)
        self.assertEqual(cycle['first_observed_truck_entries_kg'], 2.)
        self.assertEqual(result['interval_balance_max_error_kg'], 0.)

    def test_mixed_material_mass_and_id_reordering(self):
        ids = np.array([9, 2, 7]); masses = np.array([2., 3., 5.])
        ledger = MaterialLedger(ids, masses, [20, 10, 20])
        masses[:] = 99; ids[:] = 99  # Caller cannot mutate initial accounting.
        ledger.update(0, [9, 2, 7], [SOURCE, SOURCE, SOURCE])
        result = ledger.update(1, [7, 9, 2], [TRUCK, BUCKET, TRUCK])
        self.assertEqual(result['mass_kg']['truck_receiver'], 8.)
        self.assertEqual(result['material_mass_kg']['20']['truck_receiver'], 5.)
        self.assertEqual(result['material_mass_kg']['10']['truck_receiver'], 3.)
        self.assertEqual(result['partition_residual_kg'], 0.)

    def test_bucket_loss_is_unassigned_not_truck_credit(self):
        ledger = MaterialLedger([1], [4.])
        ledger.update(0, [1], [BUCKET])
        result = ledger.update(1, [1], [UNASSIGNED])
        self.assertEqual(result['truck']['first_observed_entries_kg'], 0.)
        self.assertEqual(result['mass_kg']['unassigned'], 4.)

    def test_missing_duplicate_foreign_particles_rejected_without_advancing(self):
        ledger = MaterialLedger([1, 2], [2., 3.])
        ledger.update(0, [1, 2], [SOURCE, SOURCE])
        for ids in ([1], [1, 1], [1, 3]):
            with self.assertRaises(ValueError):
                ledger.update(1, ids, [TRUCK] * len(ids))
        result = ledger.update(1, [1, 2], [TRUCK, SOURCE])
        self.assertEqual(result['truck']['first_observed_entries_kg'], 2.)

    def test_time_and_region_validation(self):
        ledger = MaterialLedger([1], [1.])
        ledger.update(0, [1], [SOURCE])
        for time in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): ledger.update(time, [1], [TRUCK])
        for region in (-1, 6, 1.5):
            with self.assertRaises(ValueError): ledger.update(1, [1], [region])
        self.assertEqual(ledger.update(1, [1], [TRUCK])['truck']['first_observed_entries_kg'], 1.)

    def test_initial_cargo_not_new_and_cycle_ids_not_reused(self):
        ledger = MaterialLedger([1], [5.])
        with self.assertRaises(ValueError): ledger.begin_cycle('a')
        ledger.update(0, [1], [TRUCK])
        ledger.begin_cycle('a')
        with self.assertRaises(ValueError): ledger.begin_cycle('b')
        ledger.update(1, [1], [UNASSIGNED])
        ledger.end_cycle()
        with self.assertRaises(ValueError): ledger.begin_cycle('a')
        ledger.begin_cycle('b')
        r = ledger.update(2, [1], [TRUCK])
        self.assertEqual(r['truck']['first_observed_entries_kg'], 0.)
        self.assertEqual(ledger.end_cycle()['truck_net_stock_change_kg'], 5.)

    def test_overlap_and_outside_have_explicit_ownership(self):
        source = np.ones(5, bool)
        bucket = np.array([1, 1, 0, 0, 0], bool)
        truck = np.array([1, 0, 1, 0, 0], bool)
        outside = np.array([0, 0, 0, 1, 0], bool)
        np.testing.assert_array_equal(exclusive_regions(source, bucket, truck, outside),
                                      [AMBIGUOUS, BUCKET, TRUCK, OUTSIDE, SOURCE])
        with self.assertRaises(ValueError): exclusive_regions(source, bucket[:-1], truck, outside)

    def test_large_sample_gap_reported_not_hidden(self):
        ledger = MaterialLedger([1], [2.])
        ledger.update(0, [1], [BUCKET])
        result = ledger.update(20, [1], [TRUCK])
        self.assertEqual(result['max_sample_gap_s'], 20.)
        self.assertEqual(result['truck']['sampled_entries_cumulative_kg'], 2.)

    def test_invalid_mass_or_identity_at_creation(self):
        for ids, mass in [([1, 1], [1., 1.]), ([1], [0.]), ([1], [-2.]),
                          ([1], [float('nan')]), ([1], [float('inf')]), ([1.1], [1.])]:
            with self.assertRaises(ValueError): MaterialLedger(ids, mass)


if __name__ == '__main__':
    unittest.main()
