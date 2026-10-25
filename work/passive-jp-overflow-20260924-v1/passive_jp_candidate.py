"""Process-local diagnostic candidate for exactly zero-hardening materials.

The Newton hardening law returns 1 for h=0 independently of Jp. This hook
freezes that unused history at its finite checkpoint value; it does not clamp
strength, change stress/elastic updates, or repair an invalid source. Not for
nonzero hardening or for later switching this trajectory to a hardening model.
"""
from contextlib import contextmanager
import numpy as np
import warp as wp


def validate_scope(hardening, history):
    hardening = np.asarray(hardening)
    history = np.asarray(history)
    if hardening.shape != history.shape or not history.size:
        raise ValueError('passive_jp_shape')
    if not np.isfinite(hardening).all() or np.any(hardening != 0):
        raise ValueError('passive_jp_requires_exactly_zero_hardening')
    if not np.isfinite(history).all() or np.any(history <= 0):
        raise ValueError('passive_jp_requires_finite_positive_source')


@contextmanager
def installed_passive_jp(solver, initial_state):
    """Wrap only this solver instance; leave global kernels and files intact.

    During _update_particles the unused scalar is neutral, so even temporary
    interpolation cannot overflow. Restore the exact input values on return,
    for both aliasing and distinct input/output states. The surrounding guard
    still observes the true preserved state after the call.
    """
    model = solver._mpm_model
    rates = model.material_parameters.hardening
    validate_scope(rates.numpy(), initial_state.mpm.particle_Jp.numpy())
    original = solver._update_particles
    if getattr(original, '_passive_jp_candidate', False):
        raise RuntimeError('passive_jp_already_installed')
    history = wp.empty_like(initial_state.mpm.particle_Jp)
    calls = dict(updates=0, policy='freeze_unused_history_at_previous_value',
                 hardening_required=0.0, source_history_rewritten=False)

    def update(state_in, state_out, dt, pic, scratch):
        previous = state_in.mpm.particle_Jp
        following = state_out.mpm.particle_Jp
        validate_scope(rates.numpy(), previous.numpy())
        wp.copy(history, previous)
        previous.fill_(1.0)
        try:
            result = original(state_in, state_out, dt, pic, scratch)
        finally:
            wp.copy(previous, history)
            if following is not previous:
                wp.copy(following, history)
        calls['updates'] += 1
        return result

    update._passive_jp_candidate = True
    solver._update_particles = update
    try:
        yield calls
    finally:
        solver._update_particles = original
