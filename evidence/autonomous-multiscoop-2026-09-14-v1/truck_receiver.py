"""Read-only CAD receiver classification. No collision or material changes.

Core is geometric occupancy, not supported cargo. Above-rim candidates are never
included in core mass. Native CAD triangle IDs are versioned in the contract.
"""
import numpy as np

OUTSIDE, CAVITY, BOUNDARY, ABOVE_RIM, OTHER_OWNER = range(5)


def validate_pose(pose):
    t = np.asarray(pose, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('invalid_rigid_receiver_pose')
    return t


def surface_height(points, triangles, axes, output_axis, maximum):
    """Intersect rays normal to a coordinate plane with actual triangles."""
    result = np.full(len(points), -np.inf if maximum else np.inf)
    for triangle in triangles:
        a, b, c = triangle[:, axes]
        ab, ac = b-a, c-a
        det = ab[0]*ac[1]-ab[1]*ac[0]
        if abs(det) < 1e-12:
            continue
        d = points[:, axes]-a
        u = (d[:, 0]*ac[1]-d[:, 1]*ac[0])/det
        v = (ab[0]*d[:, 1]-ab[1]*d[:, 0])/det
        hit = (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
        height = triangle[0, output_axis]+u*(triangle[1, output_axis]-triangle[0, output_axis])+v*(triangle[2, output_axis]-triangle[0, output_axis])
        combine = np.maximum if maximum else np.minimum
        result[hit] = combine(result[hit], height[hit])
    return result


class TruckReceiver:
    def __init__(self, contract):
        if contract.get('schema') != 'truck_receiver_triangles_v1':
            raise ValueError('invalid_receiver_schema')
        self.pose = validate_pose(contract['local_to_world']).copy()
        self.band = float(contract['boundary_band_m'])
        if not np.isfinite(self.band) or self.band < 0:
            raise ValueError('invalid_boundary_band')
        self.groups = {}
        for key in ('floor', 'left', 'right'):
            a = np.asarray(contract['triangles'][key], float)
            if a.ndim != 3 or a.shape[1:] != (3, 3) or not len(a) or not np.isfinite(a).all():
                raise ValueError('invalid_receiver_triangles')
            self.groups[key] = a
        ceiling = np.asarray(contract['triangles'].get('ceiling', []), float)
        if ceiling.size == 0:
            ceiling = np.empty((0, 3, 3))
        if ceiling.ndim != 3 or ceiling.shape[1:] != (3, 3) or not np.isfinite(ceiling).all():
            raise ValueError('invalid_receiver_ceiling')
        self.groups['ceiling'] = ceiling
        self.rim = np.asarray(contract['rim_stations'], float)
        # Stations are [x, left_y, right_y, lower_rim_z]; above is unverified.
        if (self.rim.ndim != 2 or self.rim.shape[1] != 4 or len(self.rim) < 2
                or not np.isfinite(self.rim).all() or np.any(np.diff(self.rim[:, 0]) <= 0)
                or np.any(self.rim[:, 1] >= self.rim[:, 2])):
            raise ValueError('invalid_rim_stations')
        all_vertices = np.concatenate(list(self.groups.values())).reshape(-1, 3)
        self.low = all_vertices.min(0)-self.band
        self.high = all_vertices.max(0)+self.band

    def classify(self, positions, pose=None):
        q = np.asarray(positions, float)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError('invalid_particle_positions')
        t = self.pose if pose is None else validate_pose(pose)
        local = (q-t[:3, 3])@t[:3, :3]
        labels = np.full(len(q), OUTSIDE, np.int8)
        candidate = np.all((local >= self.low) & (local <= self.high), axis=1)
        p = local[candidate]
        left = surface_height(p, self.groups['left'], [0, 2], 1, True)
        right = surface_height(p, self.groups['right'], [0, 2], 1, False)
        floor = surface_height(p, self.groups['floor'], [0, 1], 2, True)
        ceiling = surface_height(p, self.groups['ceiling'], [0, 1], 2, False)
        hits = np.isfinite(left) & np.isfinite(right) & np.isfinite(floor)
        # Axis-coordinate clearance, not Euclidean distance to steel.
        clearance = np.minimum(np.minimum(np.minimum(p[:, 1]-left, right-p[:, 1]), p[:, 2]-floor), ceiling-p[:, 2])
        result = np.full(len(p), OUTSIDE, np.int8)
        result[hits & (clearance >= -self.band)] = BOUNDARY
        result[hits & (clearance > self.band)] = CAVITY
        labels[candidate] = result
        x = local[:, 0]
        opening = (x >= self.rim[0, 0]) & (x <= self.rim[-1, 0])
        ly = np.interp(x, self.rim[:, 0], self.rim[:, 1])
        ry = np.interp(x, self.rim[:, 0], self.rim[:, 2])
        rz = np.interp(x, self.rim[:, 0], self.rim[:, 3])
        above = opening & (local[:, 1] >= ly) & (local[:, 1] <= ry) & (local[:, 2] > rz)
        # Keep above-rim material separate even if an intersecting CAD surface
        # would otherwise include it; it can be flying or falling into the bed.
        labels[above] = ABOVE_RIM
        return labels


class ReceiverDisplay:
    """Synchronous read-only observation/GUI adapter; caller owns scheduling.

    At most one classification per advancing simulation time. Repeated times do
    not refresh freshness. Caller guarantees fixed particle array order/masses.
    """
    def __init__(self, receiver, masses_kg, stale_after_s=1.):
        self.receiver = receiver
        self.mass = np.asarray(masses_kg, float).copy()
        if self.mass.ndim != 1 or not len(self.mass) or not np.isfinite(self.mass).all() or np.any(self.mass <= 0):
            raise ValueError('invalid_display_masses')
        self.stale_after_s = float(stale_after_s)
        if not np.isfinite(self.stale_after_s) or self.stale_after_s <= 0:
            raise ValueError('invalid_stale_after')
        self.record = None
        self._observed_wall = None
        self.last_labels = None

    def observe(self, positions, sim_time_s, wall_time_s, pose=None, *, other_owner_mask):
        sim, wall = float(sim_time_s), float(wall_time_s)
        if (not np.isfinite([sim, wall]).all() or sim < 0
                or (self.record is not None and (sim < self.record['sim_time_s'] or wall < self._observed_wall))):
            raise ValueError('invalid_observation_time')
        q = np.asarray(positions)
        if q.shape != (len(self.mass), 3) or not np.isfinite(q).all():
            raise ValueError('invalid_observation_population')
        excluded = np.asarray(other_owner_mask)
        if excluded.dtype != np.bool_ or excluded.shape != self.mass.shape:
            raise ValueError('invalid_other_owner_mask')
        if pose is not None:
            validate_pose(pose)
        # Do not pretend an unclassified same-time state is a fresh observation.
        # Callers should simply keep displaying the cached record while paused.
        if self.record is not None and sim == self.record['sim_time_s']:
            return dict(self.record)
        labels = self.receiver.classify(q, pose)
        labels[excluded] = OTHER_OWNER
        labels.flags.writeable = False
        self.last_labels = labels
        stocks = np.bincount(labels, weights=self.mass, minlength=5)
        self.record = {'sim_time_s':sim, 'core_region_kg':float(stocks[CAVITY]),
            'boundary_band_kg':float(stocks[BOUNDARY]), 'above_rim_unverified_kg':float(stocks[ABOVE_RIM]),
            'outside_receiver_kg':float(stocks[OUTSIDE]),
            'other_owner_kg':float(stocks[OTHER_OWNER]),
            'partition_residual_kg':float(stocks.sum()-self.mass.sum()),
            'scope':'geometric_observation_not_verified_cargo_or_volume'}
        self._observed_wall = wall
        return dict(self.record)

    def status(self, wall_time_s):
        wall = float(wall_time_s)
        if not np.isfinite(wall) or (self._observed_wall is not None and wall < self._observed_wall):
            raise ValueError('invalid_status_time')
        if self.record is None:
            return {'valid':False, 'reason':'no_observation'}
        age = wall-self._observed_wall
        return {**self.record, 'age_s':age, 'valid':age <= self.stale_after_s,
                'reason':'fresh' if age <= self.stale_after_s else 'stale_observation'}

    def draw_ui(self, imgui, wall_time_s):
        s = self.status(wall_time_s)
        imgui.text('Truck receiver / geometric observation')
        if not s['valid']:
            imgui.text('No current cargo estimate: '+s['reason'])
            return
        imgui.text(f'Inside core: {s["core_region_kg"]:.2f} kg')
        imgui.text(f'Boundary uncertain: {s["boundary_band_kg"]:.2f} kg')
        imgui.text(f'Above rim, unverified: {s["above_rim_unverified_kg"]:.2f} kg')
        imgui.text('Support, heap volume and rated fill: not validated')
