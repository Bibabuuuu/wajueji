"""Explicit frame adapter for the D-133 moving-bucket candidate.

Native projection remains unattributed. Only additional velocity impulses are
routed; the actual-motion correction is carried into the next rigid step.
"""
import copy

import numpy as np
import warp as wp

from moving_bucket_sweep_candidate import MovingBucketSweep
from run_layered_47_excavation_preflight import _bucket_transform


class MovingBucketSweepRuntime:
    def __init__(self, fixture, bridge, proxy, friction):
        self.fixture = fixture
        self.bridge = bridge
        mesh = fixture.solver._mpm_model._collider_meshes[
            int(fixture.bucket_collider_ids[0])]
        self.sweep = MovingBucketSweep(
            mesh, fixture.state, fixture.particle_mass, friction, proxy=proxy)
        self.previous = wp.empty_like(fixture.state.particle_q)
        self.after_mpm = wp.empty_like(fixture.state.particle_q)
        self.pending = None
        self.generated = np.zeros(3)
        self.routed = np.zeros(3)
        self.old_pose = None
        self.current = None

    def _correct(self, previous, old_pose, new_pose):
        state = self.bridge.state
        result = self.sweep.apply(
            self.fixture.state, previous, old_pose, new_pose,
            state['bucket_linear_velocity_world_m_s'],
            state['bucket_angular_velocity_world_rad_s'])
        result['point_world_m'] = list(wp.transform_get_translation(new_pose))
        self.generated += np.asarray(result['reaction_impulse_ns'])
        return result

    def step_mpm(self, original_step, dt, origins):
        wp.copy(self.previous, self.fixture.state.particle_q)
        self.old_pose = _bucket_transform(self.bridge.state)
        wrench, particles = original_step(dt, origins)
        self.current = self._correct(self.previous, self.old_pose, self.old_pose)
        increments = [self.current]
        if self.pending is not None:
            increments.append(self.pending)
        for item in increments:
            impulse = np.asarray(item['reaction_impulse_ns'])
            angular = np.asarray(item['reaction_angular_impulse_nms'])
            shift = np.asarray(item['point_world_m']) - np.asarray(origins['bucket'])
            wrench['bucket']['force'] += impulse / dt
            wrench['bucket']['torque'] += (angular + np.cross(shift, impulse)) / dt
            self.routed += impulse
        self.pending = None
        wp.copy(self.after_mpm, self.fixture.state.particle_q)
        return wrench, particles

    def finish_frame(self):
        if self.old_pose is None:
            raise RuntimeError('moving_bucket_sweep_frame_not_started')
        self.pending = self._correct(
            self.after_mpm, self.old_pose, _bucket_transform(self.bridge.state))
        self.old_pose = None
        balance = self.generated - self.routed - np.asarray(
            self.pending['reaction_impulse_ns'])
        finite = all(np.isfinite(item).all() for item in (
            self.generated, self.routed, balance,
            self.pending['reaction_angular_impulse_nms'],
            self.current['reaction_angular_impulse_nms']))
        return {
            'enabled': True,
            'mpm_correction': copy.deepcopy(self.current),
            'actual_motion_pending_next_frame': copy.deepcopy(self.pending),
            'incremental_impulse_balance_error_ns': balance.tolist(),
            'safe': bool(finite and np.max(np.abs(balance)) <= 1.e-7),
            'native_projection_reaction_attributed': False,
            'exact_rotational_ccd': False,
        }

    def diagnostic_state(self):
        return {
            'pending_velocity_reaction': copy.deepcopy(self.pending),
            'total_generated_reaction_ns': self.generated.tolist(),
            'total_routed_reaction_ns': self.routed.tolist(),
            'resume_qualified': False,
        }
