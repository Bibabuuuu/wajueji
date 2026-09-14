"""Read-only mass accounting for a fixed particle population.

Regions are geometric observations, not proof of support, load or settled cargo.
Transitions are observed between samples; excursions between samples are unknown.
"""
import numpy as np

REGIONS = ('source_region', 'bucket_cavity', 'truck_receiver', 'unassigned',
           'outside_domain', 'ambiguous')
SOURCE, BUCKET, TRUCK, UNASSIGNED, OUTSIDE, AMBIGUOUS = range(6)


def exclusive_regions(source, bucket, truck, outside):
    """Bucket/truck override source ROI; overlapping receivers stay ambiguous."""
    masks = [np.asarray(v) for v in (source, bucket, truck, outside)]
    if any(v.dtype != np.bool_ or v.ndim != 1 for v in masks):
        raise ValueError('expected_boolean_region_masks')
    if len({v.shape for v in masks}) != 1:
        raise ValueError('region_mask_length_mismatch')
    source, bucket, truck, outside = masks
    result = np.full(len(source), UNASSIGNED, dtype=np.int8)
    result[source] = SOURCE
    result[bucket] = BUCKET
    result[truck] = TRUCK
    result[bucket & truck] = AMBIGUOUS
    result[outside] = OUTSIDE
    return result


def _integers(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError('invalid_' + name)
    return a


class MaterialLedger:
    """One instance per run; masses/material IDs are immutable per particle ID.

    Call update for every available observation, then begin_cycle/end_cycle at
    known operator/controller boundaries. Never infer a cycle from bucket loss.
    """
    def __init__(self, particle_ids, masses_kg, material_ids=None):
        ids = _integers(particle_ids, 'particle_ids')
        mass = np.asarray(masses_kg, dtype=float)
        if not len(ids) or mass.shape != ids.shape or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError('invalid_particle_masses')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('duplicate_particle_ids')
        materials = np.zeros(len(ids), dtype=np.int64) if material_ids is None else _integers(material_ids, 'material_ids')
        if materials.shape != ids.shape:
            raise ValueError('material_id_length_mismatch')
        order = np.argsort(ids)
        self._ids = ids[order].copy()
        self._mass = mass[order].copy()
        self._materials = materials[order].copy()
        self._material_keys, self._material_index = np.unique(self._materials, return_inverse=True)
        self._previous = None
        self._time = None
        self._initial_stock = None
        self._stock = None
        self._transfers = np.zeros((len(REGIONS), len(REGIONS)))
        self._seen_truck = np.zeros(len(ids), dtype=bool)
        self._new_truck = np.zeros(len(ids), dtype=bool)
        self._initial_truck = 0.
        self._max_gap = 0.
        self._cycle = None
        self._used_cycles = set()

    def update(self, time_s, particle_ids, region_ids):
        time_s = float(time_s)
        if not np.isfinite(time_s) or time_s < 0 or (self._time is not None and time_s <= self._time):
            raise ValueError('non_increasing_or_invalid_time')
        ids = _integers(particle_ids, 'particle_ids')
        regions = _integers(region_ids, 'region_ids')
        if regions.shape != ids.shape or np.any(regions >= len(REGIONS)):
            raise ValueError('invalid_region_ids')
        order = np.argsort(ids)
        if not np.array_equal(ids[order], self._ids):
            raise ValueError('particle_population_changed')
        regions = regions[order].copy()
        stock = np.bincount(regions, weights=self._mass, minlength=len(REGIONS))
        transfers = np.zeros_like(self._transfers)
        if self._previous is not None:
            changed = self._previous != regions
            np.add.at(transfers, (self._previous[changed], regions[changed]), self._mass[changed])
            residual = stock - self._stock - (transfers.sum(0) - transfers.sum(1))
            self._max_gap = max(self._max_gap, time_s - self._time)
            first = (regions == TRUCK) & ~self._seen_truck
            self._new_truck |= first
        else:
            residual = np.zeros(len(REGIONS))
            self._initial_stock = stock.copy()
            self._initial_truck = float(stock[TRUCK])
        self._seen_truck |= regions == TRUCK
        self._transfers += transfers
        self._stock = stock
        self._previous = regions
        self._time = time_s
        material_stocks = np.zeros((len(self._material_keys), len(REGIONS)))
        np.add.at(material_stocks, (self._material_index, regions), self._mass)
        return {
            'time_s': time_s,
            'particle_count': len(self._ids),
            'mass_kg': dict(zip(REGIONS, map(float, stock))),
            'material_mass_kg': {str(k): dict(zip(REGIONS, map(float, row)))
                                 for k, row in zip(self._material_keys, material_stocks)},
            'partition_residual_kg': float(stock.sum() - self._mass.sum()),
            'interval_balance_max_error_kg': float(np.max(np.abs(residual))),
            'transitions_kg': self._edges(transfers),
            'truck': self._truck_summary(),
            'max_sample_gap_s': self._max_gap,
            'scope': 'sampled_geometric_occupancy_not_verified_delivery_or_bulk_volume',
        }

    @staticmethod
    def _edges(matrix):
        return [{'from': REGIONS[i], 'to': REGIONS[j], 'mass_kg': float(matrix[i, j])}
                for i, j in zip(*np.nonzero(matrix))]

    def _truck_summary(self):
        return {
            'initial_stock_kg': self._initial_truck,
            'current_stock_kg': float(self._stock[TRUCK]),
            'net_stock_change_kg': float(self._stock[TRUCK] - self._initial_truck),
            'sampled_entries_cumulative_kg': float(self._transfers[:, TRUCK].sum()),
            'sampled_exits_cumulative_kg': float(self._transfers[TRUCK, :].sum()),
            'first_observed_entries_kg': float(self._mass[self._new_truck].sum()),
            'first_entries_still_inside_kg': float(self._mass[self._new_truck & (self._previous == TRUCK)].sum()),
        }

    def begin_cycle(self, cycle_id):
        if self._time is None or self._cycle is not None:
            raise ValueError('cycle_requires_observation_and_no_active_cycle')
        if not isinstance(cycle_id, str) or not cycle_id or cycle_id in self._used_cycles:
            raise ValueError('invalid_or_duplicate_cycle_id')
        self._used_cycles.add(cycle_id)
        self._cycle = (cycle_id, self._time, self._truck_summary())

    def end_cycle(self):
        if self._cycle is None:
            raise ValueError('no_active_cycle')
        cycle_id, start, before = self._cycle
        after = self._truck_summary()
        result = {'cycle_id': cycle_id, 'start_time_s': start, 'end_time_s': self._time,
                  'truck_net_stock_change_kg': after['current_stock_kg'] - before['current_stock_kg'],
                  'sampled_truck_entries_kg': after['sampled_entries_cumulative_kg'] - before['sampled_entries_cumulative_kg'],
                  'sampled_truck_exits_kg': after['sampled_exits_cumulative_kg'] - before['sampled_exits_cumulative_kg'],
                  'first_observed_truck_entries_kg': after['first_observed_entries_kg'] - before['first_observed_entries_kg'],
                  'scope': 'net_change_in_receiver_region_during_explicit_cycle_not_delivery_success'}
        self._cycle = None
        return result
