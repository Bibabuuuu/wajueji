"""Preserve loaded servo deflection across handoff, as in the pure-soil tracker.

This computes velocity commands only. It never resets native targets or qpos.
The existing slew, no-down, geometry and physical guards remain downstream.
"""
import numpy as np


class LoadAwareReference:
    limits = np.array([.12, .0225, .0225, .0225])

    def __init__(self, actual, target):
        self.bias = np.asarray(target, float) - np.asarray(actual, float)
        if self.bias.shape != (4,) or not np.isfinite(self.bias).all():
            raise ValueError('invalid_loaded_handoff')
        if np.any(abs(self.bias) > .02):
            raise ValueError('handoff_bias_outside_pure_soil_budget')

    def command(self, actual, target, desired, velocity, *, loaded):
        actual, target, desired, velocity = [np.asarray(x, float) for x in
                                             (actual, target, desired, velocity)]
        if any(x.shape != (4,) or not np.isfinite(x).all()
               for x in (actual, target, desired, velocity)):
            raise ValueError('invalid_loaded_tracking_state')
        correction = (desired - actual) / 60.
        if loaded:
            correction[3] = 0.
        proposed = np.clip(self.bias + correction, -.02, .02)
        raw = velocity + 4. * (desired + proposed - target)
        blocked = (abs(raw) > self.limits) & (raw * correction > 0)
        self.bias = np.where(blocked, self.bias, proposed)
        command = np.clip(velocity + 4. * (desired + self.bias - target),
                          -self.limits, self.limits)
        if loaded:
            command[3] = 0.
        return command
