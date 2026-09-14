"""Reference governor for the four-axis, velocity-to-hold-target interface.

The caller owns dynamics, fresh scene checks and failure cleanup. This class
never writes qpos and does not qualify acquisition, delivery or task success.
"""
import numpy as np


class ReferenceTracker:
    def __init__(self, path, speed_limits, *, target_bounds, initial_bias=None, path_dt=1/60, dt=1/120):
        self.path = np.array(path, float, copy=True)
        self.limits = np.asarray(speed_limits, float)
        self.axes = np.array([7, 9, 12, 15])
        self.path_dt, self.dt = float(path_dt), float(dt)
        self.bounds = np.asarray(target_bounds, float)
        if self.bounds.shape != (4, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError('invalid_target_bounds')
        if (self.path.ndim != 2 or self.path.shape[1] != 19 or len(self.path) < 2
                or not np.isfinite(self.path).all() or self.limits.shape != (4,)
                or not np.isfinite(self.limits).all() or np.any(self.limits <= 0)
                or not 0 < self.dt <= 1/120 or not np.isfinite(self.path_dt)
                or self.path_dt < self.dt):
            raise ValueError('invalid_reference_path')
        if np.any(abs(np.diff(self.path[:, self.axes], axis=0) / self.path_dt) > self.limits + 1e-9):
            raise ValueError('reference_speed_limit')
        # Reject an inadmissible spatial reference instead of silently tracking
        # a clipped path. Bias saturation remains valid for in-bounds references.
        driven_path = self.path[:, self.axes]
        if np.any(driven_path < self.bounds[:, 0]) or np.any(driven_path > self.bounds[:, 1]):
            raise ValueError('reference_target_bounds')
        self.phase = 0.
        self.duration = (len(self.path) - 1) * self.path_dt
        self.last_time = None
        self.failure = None
        self.done = False
        self.paused_steps = 0
        self.bias = np.zeros(4) if initial_bias is None else np.asarray(initial_bias, float).copy()
        if self.bias.shape != (4,) or not np.isfinite(self.bias).all():
            raise ValueError('invalid_reference_bias')
        self.bias_limit = np.array([.02, .02, .02, .02])
        self.bias = np.clip(self.bias, -self.bias_limit, self.bias_limit)
        # Engineering tracking budgets, NOT soil force thresholds or real-machine
        # safety certificates. Actual CAD checks are required on every step.
        self.follow_tolerance = np.array([.01, .005, .005, .005])
        self.finish_tolerance = np.array([.001, .001, .001, .001])
        self.finish_velocity = np.array([.005, .002, .002, .002])

    def reference(self, phase=None):
        x = min(self.duration, self.phase if phase is None else phase) / self.path_dt
        i = min(int(x), len(self.path) - 2)
        return self.path[i] + (x-i) * (self.path[i+1] - self.path[i])

    def command(self, *, time_s, qpos, target, axis_velocity):
        if self.failure:
            raise ValueError(self.failure)
        try:
            q = np.asarray(qpos, float)
            target = np.asarray(target, float)
            velocity = np.asarray(axis_velocity, float)
            if (q.shape != (19,) or target.shape != (4,) or velocity.shape != (4,)
                    or not np.isfinite(q).all() or not np.isfinite(target).all()
                    or not np.isfinite(velocity).all() or not np.isfinite(time_s)):
                raise ValueError('invalid_tracking_state')
            if self.last_time is not None and abs(time_s-self.last_time-self.dt) > 1e-8:
                raise ValueError('tracking_clock_discontinuity')
            self.last_time = time_s
            error = abs(q[self.axes] - self.reference()[self.axes])
            if np.any(error > 3*self.follow_tolerance):
                raise ValueError('tracking_corridor_exceeded')
            if np.any(error > self.follow_tolerance):
                self.paused_steps += 1
                if self.paused_steps*self.dt > 5.:
                    raise ValueError('tracking_recovery_timeout')
            else:
                self.paused_steps = 0
                self.phase = min(self.duration, self.phase+self.dt)
            finish_error = abs(q[self.axes]-self.path[-1, self.axes])
            if (self.phase >= self.duration and np.all(finish_error <= self.finish_tolerance)
                    and np.all(abs(velocity) <= self.finish_velocity)):
                self.done = True
            # Bounded integral correction removes static servo deflection under
            # sustained load. Freeze windup in the direction of rate saturation.
            reference = self.reference()[self.axes]
            correction = self.dt * (reference-q[self.axes])
            proposed_bias = np.clip(self.bias+correction, -self.bias_limit, self.bias_limit)
            desired = np.clip(reference+proposed_bias, self.bounds[:, 0], self.bounds[:, 1])
            raw = (desired-target)/self.dt
            blocked = (abs(raw) > self.limits) & (raw*correction > 0)
            proposed_bias = np.where(blocked, self.bias, proposed_bias)
            desired = np.clip(reference+proposed_bias, self.bounds[:, 0], self.bounds[:, 1])
            self.bias = desired-reference  # anti-windup at joint/target bounds
            return np.clip((desired-target)/self.dt, -self.limits, self.limits)
        except ValueError as error:
            self.failure = str(error)
            raise
