"""Observation-only rear lower-face target selection.

This module does not plan or execute excavation motion.  It converts an
already selected observed strip into a separate contact target on the outward
face of the pile.  Non-contact handoff planning remains a later, independent
step so a face target cannot be mistaken for a safe prepoint.
"""

from __future__ import annotations

import math

import numpy as np


DEFAULT_TARGET_HEIGHT_FRACTION = 0.40
DEFAULT_HEIGHT_FRACTION_BAND = (0.30, 0.50)
DEFAULT_MAXIMUM_OUTWARD_DISTANCE_M = 0.50
DEFAULT_MINIMUM_FACE_DROP_M = 0.40


def _profile_inputs(terrain: dict, anchor_xy_m, bucket_width_m: float):
    height = np.asarray(terrain["height_m"], dtype=np.float64)
    count = np.asarray(terrain["point_count"], dtype=np.int64)
    origin = np.asarray(terrain["origin_xy_m"], dtype=np.float64)
    resolution = float(terrain["resolution_m"])
    anchor = np.asarray(anchor_xy_m, dtype=np.float64)
    if (not terrain.get("valid") or height.ndim != 2
            or count.shape != height.shape or origin.shape != (2,)
            or anchor.shape != (2,) or not np.isfinite(anchor).all()
            or not np.isfinite(resolution) or resolution <= 0.0
            or not math.isfinite(bucket_width_m) or bucket_width_m <= 0.0
            or np.linalg.norm(anchor) <= resolution):
        raise ValueError("rear_lower_face_input_contract")
    yy, xx = np.indices(height.shape)
    centers = np.stack((
        origin[0] + resolution * (xx + 0.5),
        origin[1] + resolution * (yy + 0.5)), axis=-1)
    radial = anchor / np.linalg.norm(anchor)
    return height, count, centers, resolution, anchor, radial


def observed_outward_profile(
        terrain: dict,
        anchor_xy_m,
        *,
        bucket_width_m: float = 0.60,
        maximum_outward_distance_m: float =
            DEFAULT_MAXIMUM_OUTWARD_DISTANCE_M) -> list[dict]:
    """Return width-median surface samples outward from an observed anchor."""
    if (not math.isfinite(maximum_outward_distance_m)
            or maximum_outward_distance_m <= 0.0):
        raise ValueError("rear_lower_face_distance_contract")
    height, count, centers, cell, anchor, radial = _profile_inputs(
        terrain, anchor_xy_m, bucket_width_m)
    delta = centers - anchor
    outward = delta[..., 0] * radial[0] + delta[..., 1] * radial[1]
    lateral = np.abs(
        delta[..., 0] * radial[1] - delta[..., 1] * radial[0])
    steps = np.arange(
        0.0, maximum_outward_distance_m + 0.25 * cell, cell)
    profile = []
    for step in steps:
        mask = (
            (np.abs(outward - step) <= 0.51 * cell)
            & (lateral <= bucket_width_m / 2.0 + 0.01 * cell))
        if not np.any(mask):
            break
        values = height[mask]
        counts = count[mask]
        if not np.isfinite(values).all() or np.any(counts < 1):
            break
        profile.append({
            "outward_offset_m": float(step),
            "center_xy_m": (anchor + step * radial).tolist(),
            "surface_z_m": float(np.median(values)),
            "minimum_surface_z_m": float(np.min(values)),
            "maximum_surface_z_m": float(np.max(values)),
            "cell_count": int(np.count_nonzero(mask)),
        })
    return profile


def select_rear_lower_face(
        terrain: dict,
        strip_selection: dict,
        *,
        bucket_width_m: float = 0.60,
        target_height_fraction: float = DEFAULT_TARGET_HEIGHT_FRACTION,
        height_fraction_band: tuple[float, float] =
            DEFAULT_HEIGHT_FRACTION_BAND,
        maximum_outward_distance_m: float =
            DEFAULT_MAXIMUM_OUTWARD_DISTANCE_M,
        minimum_face_drop_m: float =
            DEFAULT_MINIMUM_FACE_DROP_M) -> dict:
    """Select an observed rear-face contact target near 40% of pile height."""
    lower, upper = height_fraction_band
    if (not all(math.isfinite(value) for value in (
            target_height_fraction, lower, upper, minimum_face_drop_m))
            or not 0.0 < lower <= target_height_fraction <= upper < 1.0
            or minimum_face_drop_m <= 0.0):
        raise ValueError("rear_lower_face_fraction_contract")
    anchor = np.asarray(strip_selection.get("entry_xy_m"), dtype=np.float64)
    if (anchor.shape != (2,) or not np.isfinite(anchor).all()):
        raise ValueError("rear_lower_face_selection_contract")
    profile = observed_outward_profile(
        terrain, anchor, bucket_width_m=bucket_width_m,
        maximum_outward_distance_m=maximum_outward_distance_m)
    if len(profile) < 3:
        raise ValueError("rear_lower_face_profile_too_short")
    crest = float(profile[0]["surface_z_m"])
    base = min(float(sample["surface_z_m"]) for sample in profile[1:])
    face_drop = crest - base
    if face_drop < minimum_face_drop_m:
        raise ValueError("rear_lower_face_drop_too_small")
    candidates = []
    for sample in profile[1:]:
        fraction = (float(sample["surface_z_m"]) - base) / face_drop
        if lower <= fraction <= upper:
            candidates.append((
                abs(fraction - target_height_fraction),
                -float(sample["outward_offset_m"]),
                sample,
                fraction,
            ))
    if not candidates:
        raise ValueError("rear_lower_face_height_band_empty")
    _, _, selected, fraction = min(candidates, key=lambda item: item[:2])
    return {
        "source": "rear_lower_face_from_observed_profile",
        "contact_xy_m": selected["center_xy_m"],
        "contact_surface_z_m": selected["surface_z_m"],
        "outward_offset_m": selected["outward_offset_m"],
        "normalized_height_fraction": float(fraction),
        "target_height_fraction": float(target_height_fraction),
        "height_fraction_band": [float(lower), float(upper)],
        "crest_surface_z_m": crest,
        "base_surface_z_m": base,
        "face_drop_m": face_drop,
        "bucket_width_m": float(bucket_width_m),
        "strip_anchor_xy_m": anchor.tolist(),
        "profile": profile,
        "prepoint_status": "not_selected_requires_separate_clearance_plan",
    }
