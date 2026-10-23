"""Four-axis manual baseline plus bounded SI residual, before runtime protection."""
import numpy as np

LIMITS_SI = np.array([.12, .0225, .0225, .0675])


def compare_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('manual_restore_field_set')
    exact = {k: bool(np.array_equal(actual[k], value)) for k, value in expected.items()}
    errors = {k: float(np.max(np.abs(actual[k].astype(float)-value.astype(float))))
              if value.dtype.kind in 'biufc' else None for k, value in expected.items()}
    return exact, errors


def compose_action(baseline_si, residual_normalized, residual_limits_si):
    baseline, residual, caps = [np.asarray(x, dtype=float) for x in
                                (baseline_si, residual_normalized, residual_limits_si)]
    if any(x.shape != (4,) or not np.isfinite(x).all() for x in (baseline, residual, caps)):
        raise ValueError('manual_residual_shape_or_finite')
    if np.any(abs(baseline) > LIMITS_SI + 1e-15):
        raise ValueError('manual_baseline_envelope')
    if np.any(abs(residual) > 1) or np.any(caps < 0) or np.any(caps > LIMITS_SI):
        raise ValueError('manual_residual_envelope')
    if np.any((caps == 0) & (residual != 0)):
        raise ValueError('manual_residual_axis_not_enabled')
    delta = residual * caps
    # Exact zero correction preserves the saved command, including signed zero.
    combined = baseline.copy() if not residual.any() else np.clip(baseline + delta, -LIMITS_SI, LIMITS_SI)
    return combined, delta
