"""V4 actor vector with compact, episode-local kinematic history."""

from __future__ import annotations

import hashlib
import json
import math

from contracts import validate_observation_v3
from observation_vector_v3 import (
    FEATURE_NAMES as V3_FEATURE_NAMES,
    NORMALIZED_INDICES as V3_NORMALIZED_INDICES,
    encode_observation as encode_v3_observation,
)


SCHEMA = "excavation_actor_vector_v4"
HISTORY_TICKS = 3
COMPACT_VALUE_NAMES = (
    *(f"cylinder_position_normalized[{index}]" for index in range(3)),
    *(f"axis_velocity_normalized[{index}]" for index in range(4)),
    *(f"bucket_relative_to_target_m[{index}]" for index in range(3)),
    *(f"previous_action[{index}]" for index in range(4)),
    "bucket_clearance_m",
    "bucket_descent_rate_mps",
    "digging_interaction_observed",
)
CONTINUOUS_COMPACT_COUNT = len(COMPACT_VALUE_NAMES) - 1


def _history_feature_names():
    names = []
    normalize = list(V3_NORMALIZED_INDICES)
    for offset in range(1, HISTORY_TICKS + 1):
        for name in COMPACT_VALUE_NAMES:
            names.append(f"history_t_minus_{offset}.{name}")
            if name != "digging_interaction_observed":
                normalize.append(len(V3_FEATURE_NAMES) + len(names) - 1)
        names.append(f"history_t_minus_{offset}.valid")
    return tuple(names), tuple(normalize)


HISTORY_FEATURE_NAMES, NORMALIZED_INDICES = _history_feature_names()
FEATURE_NAMES = V3_FEATURE_NAMES + HISTORY_FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)
FEATURE_SPEC_SHA256 = hashlib.sha256(json.dumps({
    "schema": SCHEMA,
    "feature_names": FEATURE_NAMES,
    "normalized_indices": NORMALIZED_INDICES,
}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def compact_values(observation):
    """Extract one validated compact history snapshot."""
    validate_observation_v3(observation)
    proprio = observation["proprioception"]
    bucket = observation["bucket"]
    action = observation["previous_action"]["normalized_velocity"]
    safety = observation["safety_state"]
    values = (
        *(float(value) for value in
          proprio["cylinder_position_normalized"]),
        *(float(value) for value in
          proprio["axis_velocity_normalized"]),
        *(float(value) for value in bucket["relative_to_target_m"]),
        *(float(value) for value in action),
        float(safety["bucket_clearance_m"]),
        float(safety["bucket_descent_rate_mps"]),
        1.0 if observation["digging_interaction_observed"] else 0.0,
    )
    if len(values) != len(COMPACT_VALUE_NAMES):
        raise RuntimeError("compact_history_shape_mismatch")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("compact_history_nonfinite")
    return values


def encode_observation(value, prior_observations):
    """Encode current v3 observation plus three strictly prior observations.

    ``prior_observations`` is newest first. Missing history is represented by
    trailing ``None`` slots and an explicit validity bit; history never crosses
    an episode boundary.
    """
    if (not isinstance(prior_observations, (list, tuple))
            or len(prior_observations) != HISTORY_TICKS):
        raise ValueError("invalid_compact_history_slots")
    validate_observation_v3(value)
    current_step = int(value["decision_step"])
    output = list(encode_v3_observation(value))
    missing_seen = False
    for offset, prior in enumerate(prior_observations, start=1):
        if prior is None:
            missing_seen = True
            output.extend([0.0] * len(COMPACT_VALUE_NAMES))
            output.append(0.0)
            continue
        if missing_seen:
            raise ValueError("noncontiguous_compact_history")
        validate_observation_v3(prior)
        if int(prior["decision_step"]) != current_step - offset:
            raise ValueError("noncausal_compact_history_step")
        output.extend(compact_values(prior))
        output.append(1.0)
    if len(output) != FEATURE_COUNT:
        raise RuntimeError("actor_vector_v4_feature_count_mismatch")
    if not all(math.isfinite(item) for item in output):
        raise RuntimeError("actor_vector_v4_nonfinite")
    return tuple(output)


def feature_contract():
    return {
        "schema": SCHEMA,
        "feature_count": FEATURE_COUNT,
        "feature_spec_sha256": FEATURE_SPEC_SHA256,
        "parent_vector_schema": "excavation_actor_vector_v3",
        "history_ticks": HISTORY_TICKS,
        "history_duration_s_including_current": 0.2,
        "history_feature_names": list(HISTORY_FEATURE_NAMES),
        "episode_boundary": "zero_pad_with_explicit_valid_bit",
        "repeated_terrain_history": False,
        "excluded_actor_truth": [
            "medium_id", "fill_ratio_truth", "bucket_particle_count",
            "particle_id", "mpm_internal_plastic_state",
        ],
    }
