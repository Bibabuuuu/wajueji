"""Opt-in coherent origin/endpoint contract for moving bucket contact.

The original D133 adapter is retained for frozen checkpoints. MPM advances
particles to the predicted end of the step; its sweep must use that same end
pose. After MuJoCo, only the predicted-to-actual residual motion remains.
"""
from copy import deepcopy
import numpy as np
import warp as wp

from moving_bucket_sweep_runtime import MovingBucketSweepRuntime
from run_layered_47_excavation_preflight import _bucket_transform

POLICY = 'bucket_origin_predicted_endpoint_v1'


def restored_controller(controller, bridge_state, previous_policy):
    """Convert only the old derived COM velocity; keep physical state intact."""
    result = deepcopy(controller)
    if previous_policy == POLICY:
        return result
    if previous_policy is not None:
        raise ValueError('unknown_bucket_motion_policy')
    offset = np.asarray(bridge_state['bucket_inertial_offset_world_m'], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('invalid_bucket_inertial_offset')
    result['bucket_linear_velocity_world_m_s'] = (
        np.asarray(controller['bucket_linear_velocity_world_m_s']) -
        np.cross(controller['bucket_angular_velocity_world_rad_s'], offset)).tolist()
    return result


def predict_pose(pose, linear, angular, dt):
    omega = np.asarray(angular, dtype=float)
    speed = float(np.linalg.norm(omega))
    spin = (wp.quat_identity() if speed == 0 else
            wp.quat_from_axis_angle(wp.vec3(*(omega / speed)), speed * dt))
    return wp.transform(
        wp.transform_get_translation(pose) + wp.vec3(*(np.asarray(linear) * dt)),
        wp.mul(spin, wp.transform_get_rotation(pose)))


class ConsistentBucketSweepRuntime(MovingBucketSweepRuntime):
    def step_mpm(self, original_step, dt, origins):
        wp.copy(self.previous, self.fixture.state.particle_q)
        state = self.bridge.state
        start = _bucket_transform(state)
        predicted = predict_pose(start, state['bucket_linear_velocity_world_m_s'],
                                 state['bucket_angular_velocity_world_rad_s'], dt)
        wrench, particles = original_step(dt, origins)
        self.current = self._correct(self.previous, start, predicted)
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
        # Base finish_frame sweeps this endpoint to the actual rigid endpoint.
        self.old_pose = predicted
        return wrench, particles
