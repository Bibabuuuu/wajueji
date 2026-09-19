"""Deterministic actor-vector contract for four-axis observations."""

from __future__ import annotations

import hashlib
import json
import math

from contracts import AXES, TERRAIN_SHAPE_YX, validate_observation


SCHEMA = "excavation_actor_vector_v1"


def feature_spec():
    """Return stable feature names and the indices normalized from train data."""
    names = []
    normalize = []

    def add(name, *, continuous=True):
        names.append(name)
        if continuous:
            normalize.append(len(names) - 1)

    height, width = TERRAIN_SHAPE_YX
    for y in range(height):
        for x in range(width):
            add(f"terrain.height_m[{y}][{x}]")
    for y in range(height):
        for x in range(width):
            add(f"terrain.known_mask[{y}][{x}]", continuous=False)
    add("terrain.valid", continuous=False)
    add("terrain.age_s")
    for index in range(3):
        add(f"target_world_m[{index}]")
    add("target_valid", continuous=False)
    for index in range(2):
        add(f"proprioception.swing_sin_cos[{index}]")
    for index in range(3):
        add(f"proprioception.cylinder_position_normalized[{index}]")
    for index in range(4):
        add(f"proprioception.axis_velocity_normalized[{index}]")
    for index in range(4):
        add(f"proprioception.axis_valid_mask[{index}]", continuous=False)
    for index in range(3):
        add(f"bucket.position_world_m[{index}]")
    for index in range(4):
        add(f"bucket.orientation_quaternion_wxyz[{index}]")
    for index in range(3):
        add(f"bucket.relative_to_target_m[{index}]")
    add("bucket.mouth_angle_deg")
    add("bucket.valid", continuous=False)
    for history in range(4):
        for component in range(6):
            add(f"wrench_history.values_world_si[{history}][{component}]")
    for history in range(4):
        add(f"wrench_history.age_s[{history}]")
    for history in range(4):
        add(f"wrench_history.valid_mask[{history}]", continuous=False)
    for axis in AXES:
        add(f"previous_action.normalized_velocity[{axis}]")
    add("elapsed_fraction")
    add("digging_interaction_observed", continuous=False)
    return tuple(names), tuple(normalize)


FEATURE_NAMES, NORMALIZED_INDICES = feature_spec()
FEATURE_COUNT = len(FEATURE_NAMES)
FEATURE_SPEC_SHA256 = hashlib.sha256(json.dumps({
    "schema": SCHEMA,
    "feature_names": FEATURE_NAMES,
    "normalized_indices": NORMALIZED_INDICES,
}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def encode_observation(value):
    """Validate and flatten one observation without using scoring-only truth."""
    validate_observation(value)
    output = []
    terrain = value["terrain"]
    for height_row, known_row in zip(
            terrain["height_m"], terrain["known_mask"]):
        output.extend(float(height) if known else 0.0
                      for height, known in zip(height_row, known_row))
    for known_row in terrain["known_mask"]:
        output.extend(1.0 if known else 0.0 for known in known_row)
    output.extend((1.0 if terrain["valid"] else 0.0, float(terrain["age_s"])))
    output.extend(float(item) for item in value["target_world_m"])
    output.append(1.0 if value["target_valid"] else 0.0)

    proprio = value["proprioception"]
    output.extend(float(item) for item in proprio["swing_sin_cos"])
    output.extend(float(item) for item in proprio["cylinder_position_normalized"])
    output.extend(float(item) for item in proprio["axis_velocity_normalized"])
    output.extend(1.0 if item else 0.0 for item in proprio["axis_valid_mask"])

    bucket = value["bucket"]
    output.extend(float(item) for item in bucket["position_world_m"])
    output.extend(float(item) for item in bucket["orientation_quaternion_wxyz"])
    output.extend(float(item) for item in bucket["relative_to_target_m"])
    output.append(float(bucket["mouth_angle_deg"]))
    output.append(1.0 if bucket["valid"] else 0.0)

    wrench = value["wrench_history"]
    for sample, valid in zip(wrench["values_world_si"], wrench["valid_mask"]):
        output.extend(float(item) for item in sample) if valid else output.extend([0.0] * 6)
    output.extend(float(item) for item in wrench["age_s"])
    output.extend(1.0 if item else 0.0 for item in wrench["valid_mask"])
    output.extend(float(item) for item in value["previous_action"]["normalized_velocity"])
    output.append(float(value["elapsed_fraction"]))
    output.append(1.0 if value["digging_interaction_observed"] else 0.0)

    if len(output) != FEATURE_COUNT:
        raise RuntimeError("actor_vector_feature_count_mismatch")
    if not all(math.isfinite(item) for item in output):
        raise RuntimeError("actor_vector_nonfinite")
    return tuple(output)


def feature_contract():
    return {
        "schema": SCHEMA,
        "feature_count": FEATURE_COUNT,
        "feature_spec_sha256": FEATURE_SPEC_SHA256,
        "normalization": (
            "fit mean/std on training episodes for continuous indices only; "
            "binary validity and known masks remain 0/1"),
        "unknown_terrain": "height is encoded as 0 only alongside a 0 known-mask bit",
        "excluded_actor_truth": [
            "medium_id", "fill_ratio_truth", "bucket_particle_count",
            "particle_id", "mpm_internal_plastic_state",
        ],
    }
