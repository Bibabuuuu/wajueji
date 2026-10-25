"""Threshold-free observation of the four driven-axis external loads."""

from __future__ import annotations

import numpy as np


AXIS_LABELS = ("swing", "boom", "arm", "bucket")
AXIS_UNITS = ("N*m", "N", "N", "N")


def _finite_vector(state: dict, name: str, size: int) -> np.ndarray:
    value = np.asarray(state[name], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("drive_load_observation_" + name)
    return value


def observe_drive_load(
        state: dict, *, previous=None, marker_baseline=None) -> dict:
    """Return raw per-axis loads and deltas without declaring contact.

    The swing channel is a torque while the three cylinder channels are
    forces, so this deliberately does not form a mixed-unit vector norm or
    apply a guessed common threshold.
    """
    current = _finite_vector(state, "equivalent_drive_load", 4)
    prior = current if previous is None else np.asarray(
        previous, dtype=np.float64)
    baseline = current if marker_baseline is None else np.asarray(
        marker_baseline, dtype=np.float64)
    if (prior.shape != (4,) or baseline.shape != (4,)
            or not np.isfinite(prior).all()
            or not np.isfinite(baseline).all()):
        raise ValueError("drive_load_observation_reference")
    return {
        "drive_load_axis_labels": list(AXIS_LABELS),
        "drive_load_axis_units": list(AXIS_UNITS),
        "equivalent_drive_load": current.tolist(),
        "equivalent_drive_load_delta_from_previous": (
            current - prior).tolist(),
        "equivalent_drive_load_delta_from_phase_marker": (
            current - baseline).tolist(),
        "external_load_compensation": _finite_vector(
            state, "external_load_compensation", 4).tolist()
            if "external_load_compensation" in state else None,
        "external_load_compensation_applied": False,
        "unloaded_feedforward": _finite_vector(
            state, "unloaded_feedforward", 4).tolist(),
        "actuator_force": _finite_vector(
            state, "actuator_force", 12).tolist(),
        "load_contact_classification": None,
        "load_thresholds_applied": False,
    }
