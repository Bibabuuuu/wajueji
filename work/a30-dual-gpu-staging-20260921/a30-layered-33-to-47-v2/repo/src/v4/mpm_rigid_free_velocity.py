"""Free-velocity predictor for finite-mass MPM proxies of free rigid rocks.

MPM solves the contact impulse, but does not apply body gravity. Its proxy
must therefore see v_free = v_n + dt*g before solving that impulse. MuJoCo
still advances the original body state with gravity and J/dt exactly once.
This helper is not an articulated-body/contact predictor for the excavator.
"""

import numpy as np


def predict_free_linear_velocity(velocity, gravity, dt):
    velocity = np.asarray(velocity, dtype=np.float64)
    gravity = np.asarray(gravity, dtype=np.float64)
    if velocity.shape != (3,) or gravity.shape != (3,):
        raise ValueError("velocity_and_gravity_must_be_vec3")
    if not np.isfinite(dt) or dt <= 0 or not (
            np.isfinite(velocity).all() and np.isfinite(gravity).all()):
        raise ValueError("nonfinite_or_nonpositive_predictor_input")
    return velocity + float(dt) * gravity
