"""Backend-independent camera geometry, terrain reconstruction and state contract."""
from dataclasses import dataclass
import math
import numpy as np


def rigid_transform(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('invalid_transform')
    if not np.allclose(value[3], [0, 0, 0, 1]) or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-6):
        raise ValueError('transform_not_rigid')
    return value.copy()


@dataclass(frozen=True)
class Camera:
    width: int = 320
    height: int = 240
    fx: float = 260.
    fy: float = 260.
    cx: float = 159.5
    cy: float = 119.5
    near: float = .2
    far: float = 20.

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or not all(math.isfinite(x) for x in (self.fx, self.fy, self.cx, self.cy, self.near, self.far)) or min(self.fx, self.fy) <= 0 or not 0 < self.near < self.far:
            raise ValueError('invalid_camera')

    def points(self, depth, world_from_camera):
        """Optical frame: +X right, +Y down, +Z forward; depth is optical Z."""
        depth = np.asarray(depth, dtype=float)
        if depth.shape != (self.height, self.width):
            raise ValueError('depth_shape')
        pose = rigid_transform(world_from_camera)
        valid = np.isfinite(depth) & (depth >= self.near) & (depth < self.far)
        v, u = np.nonzero(valid); z = depth[valid]
        optical = np.column_stack(((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z))
        return optical @ pose[:3, :3].T + pose[:3, 3], valid


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye; length = np.linalg.norm(forward)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError('invalid_camera_direction')
    forward /= length
    right = np.cross(forward, [0., 0., 1.]); length = np.linalg.norm(right)
    if length < 1e-9:
        right = np.array([1., 0., 0.])
    else:
        right /= length
    down = np.cross(forward, right)
    pose = np.eye(4); pose[:3, :3] = np.column_stack((right, down, forward)); pose[:3, 3] = eye
    return rigid_transform(pose)


@dataclass(frozen=True)
class Grid:
    xmin: float = 3.5
    xmax: float = 7.3
    ymin: float = -1.5
    ymax: float = 1.5
    resolution: float = .1

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.xmin, self.xmax, self.ymin, self.ymax, self.resolution)) or self.xmin >= self.xmax or self.ymin >= self.ymax or self.resolution <= 0:
            raise ValueError('invalid_grid')

    @property
    def shape(self):
        return (math.ceil((self.ymax-self.ymin)/self.resolution-1e-10), math.ceil((self.xmax-self.xmin)/self.resolution-1e-10))


def terrain_from_points(points, grid):
    """Observed top surface only. Unobserved cells remain NaN, never free space."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('point_shape')
    valid = np.isfinite(points).all(axis=1)
    valid &= (points[:, 0] >= grid.xmin) & (points[:, 0] < grid.xmax) & (points[:, 1] >= grid.ymin) & (points[:, 1] < grid.ymax)
    p = points[valid]
    nx = grid.shape[1]
    ix = np.floor((p[:, 0]-grid.xmin)/grid.resolution).astype(int)
    iy = np.floor((p[:, 1]-grid.ymin)/grid.resolution).astype(int)
    ids = iy*nx+ix; size = int(np.prod(grid.shape))
    height = np.full(size, -np.inf); np.maximum.at(height, ids, p[:, 2])
    count = np.bincount(ids, minlength=size)
    height[count == 0] = np.nan
    return height.reshape(grid.shape), count.reshape(grid.shape)


def noisy_depth(depth, camera, *, seed, sigma_m=.005, dropout=.02):
    if not math.isfinite(sigma_m) or sigma_m < 0 or not 0 <= dropout <= 1:
        raise ValueError('invalid_sensor_noise')
    output = np.asarray(depth, dtype=np.float32).copy()
    if output.shape != (camera.height, camera.width):
        raise ValueError('depth_shape')
    rng = np.random.default_rng(seed)
    valid = np.isfinite(output) & (output >= camera.near) & (output < camera.far)
    output[valid] += rng.normal(0, sigma_m, int(valid.sum()))
    valid &= rng.random(output.shape) >= dropout
    valid &= (output >= camera.near) & (output < camera.far)
    output[~valid] = np.nan
    return output


class StateBuilder:
    """Explicit sensor provenance and freshness, without simulator material state."""
    def __init__(self, camera, grid, max_age_sim_s=.3, max_age_wall_s=.3):
        if min(max_age_sim_s, max_age_wall_s) <= 0:
            raise ValueError('invalid_max_age')
        self.camera, self.grid = camera, grid
        self.max_age_sim_s, self.max_age_wall_s = max_age_sim_s, max_age_wall_s
        self._last = None

    def reset(self):
        self._last = None

    def build(self, *, episode, sequence, capture_sim_s, now_sim_s, capture_wall_s, now_wall_s,
              depth, world_from_camera, bucket_pose, joints, wrench, truck_prior, self_mask=None, replay=False,
              extra_views=()):
        times = (capture_sim_s, now_sim_s, capture_wall_s, now_wall_s)
        if not all(math.isfinite(v) for v in times) or min(times) < 0 or now_sim_s < capture_sim_s or now_wall_s < capture_wall_s:
            raise ValueError('invalid_timestamp')
        if not episode or sequence < 0 or int(sequence) != sequence:
            raise ValueError('invalid_sequence')
        if self._last and (episode != self._last[0] or sequence <= self._last[1] or capture_sim_s < self._last[2]):
            raise ValueError('out_of_order_or_episode_not_reset')
        joint_values = np.asarray(joints, dtype=float)
        if joint_values.ndim != 1 or not np.isfinite(joint_values).all():
            raise ValueError('invalid_joint_measurement')
        force = None if wrench is None else np.asarray(wrench, dtype=float)
        if force is not None and (force.shape != (6,) or not np.isfinite(force).all()):
            raise ValueError('invalid_wrench')
        depth = np.asarray(depth, dtype=float).copy()
        if self_mask is not None:
            mask = np.asarray(self_mask)
            if mask.shape != depth.shape or mask.dtype != bool:
                raise ValueError('invalid_self_mask')
            depth[mask] = np.nan
        points, valid = self.camera.points(depth, world_from_camera)
        view_points = [points]
        valid_pixels = [int(valid.sum())]
        for camera, view_depth, view_pose, view_mask in extra_views:
            view_depth = np.asarray(view_depth, dtype=float).copy()
            if view_mask is not None:
                view_mask = np.asarray(view_mask)
                if view_mask.shape != view_depth.shape or view_mask.dtype != bool:
                    raise ValueError('invalid_self_mask')
                view_depth[view_mask] = np.nan
            more, more_valid = camera.points(view_depth, view_pose)
            view_points.append(more); valid_pixels.append(int(more_valid.sum()))
        points = np.concatenate(view_points, axis=0)
        height, count = terrain_from_points(points, self.grid)
        age_sim, age_wall = now_sim_s-capture_sim_s, now_wall_s-capture_wall_s
        fresh = age_sim <= self.max_age_sim_s and age_wall <= self.max_age_wall_s
        observed = count > 0
        packet = {
            'schema': 'excavation_observation_v1', 'episode': episode, 'sequence': int(sequence),
            'frame': 'world', 'units': {'length': 'm', 'angle': 'rad', 'force': 'N', 'torque': 'N*m', 'time': 's'},
            'time': {'capture_sim_s': capture_sim_s, 'processed_sim_s': now_sim_s, 'capture_monotonic_s': capture_wall_s,
                     'processed_monotonic_s': now_wall_s, 'age_sim_s': age_sim, 'age_wall_s': age_wall,
                     'expires_sim_s': capture_sim_s+self.max_age_sim_s,
                     'expires_monotonic_s': capture_wall_s+self.max_age_wall_s},
            'fresh': bool(fresh), 'replay': bool(replay),
            'proprioception': {'source': 'ideal_virtual_all_joint_encoders_and_known_base_pose', 'joints': joint_values.tolist(),
                               'world_from_bucket': rigid_transform(bucket_pose).tolist(), 'extrinsics_calibrated': False},
            'terrain': {'source': 'depth_backprojection_and_cell_max', 'roi_is_prior': True,
                        'origin_xy_m': [self.grid.xmin, self.grid.ymin], 'resolution_m': self.grid.resolution,
                        'shape_yx': list(self.grid.shape), 'coverage': float(observed.mean()),
                        'valid': bool(fresh and observed.any()), 'unknown_cells': int((~observed).sum()),
                        'height_m': [[float(v) if np.isfinite(v) else None for v in row] for row in height],
                        'point_count': count.tolist(), 'history_fused': False, 'camera_count': len(view_points),
                        'spatial_views_fused': len(view_points)>1, 'collision_free_claim': False},
            'load': {'source': 'virtual_contact_wrench', 'wrench_world': None if force is None else force.tolist(),
                     'valid': bool(force is not None and fresh), 'mass_kg': None, 'fill_ratio': None,
                     'mass_estimator_status': 'not_implemented', 'physical_force_calibrated': False},
            'truck': {'source': 'configured_prior', 'visual_localization_verified': False, 'prior': truck_prior},
            'quality': {'valid_depth_pixels': sum(valid_pixels), 'valid_pixels_per_view': valid_pixels, 'no_return_is_unknown': True,
                        'material_class': 'unknown', 'autonomous_task_ready': False}}
        self._last = (episode, int(sequence), capture_sim_s)
        return packet, height, count


def observation_fresh(packet, now_sim_s, now_wall_s):
    """Consumers must recheck expiry; validity at publication does not persist."""
    if not all(math.isfinite(v) for v in (now_sim_s, now_wall_s)) or not packet.get('fresh',False):
        return False
    t=packet['time']
    return (t['capture_sim_s'] <= now_sim_s <= t['expires_sim_s'] and
            t['capture_monotonic_s'] <= now_wall_s <= t['expires_monotonic_s'])


class LatestSchedule:
    """Wall-clock sampling with no catch-up queue; skipped deadlines are counted."""
    def __init__(self, period_s=.2):
        if not math.isfinite(period_s) or period_s <= 0:
            raise ValueError('invalid_period')
        self.period=period_s; self.next=None; self.skipped=0; self.last=None

    def due(self, now):
        if not math.isfinite(now) or now < 0 or (self.last is not None and now < self.last):
            raise ValueError('invalid_clock')
        self.last=now
        if self.next is None:
            self.next=now+self.period
            return True
        if now < self.next:
            return False
        slots=math.floor((now-self.next)/self.period)+1
        self.skipped+=slots-1; self.next+=slots*self.period
        return True
