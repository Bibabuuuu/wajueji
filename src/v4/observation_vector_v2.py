"""Safety-augmented actor-vector contract for supervised takeover data."""

from __future__ import annotations

import hashlib
import json
import math

from contracts import OBSERVATION_SCHEMA, validate_observation_v2
from observation_vector import (
    FEATURE_NAMES as V1_FEATURE_NAMES,
    NORMALIZED_INDICES as V1_NORMALIZED_INDICES,
    encode_observation as encode_v1_observation,
)


SCHEMA = "excavation_actor_vector_v2"
SAFETY_FEATURE_NAMES = (
    "safety_state.bucket_clearance_m",
    "safety_state.bucket_descent_rate_mps",
    "safety_state.predicted_ground_ttc_s",
    "safety_state.predicted_ground_ttc_valid",
    "safety_state.bucket_end_stop_confirmed",
)
FEATURE_NAMES = V1_FEATURE_NAMES + SAFETY_FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)
NORMALIZED_INDICES = V1_NORMALIZED_INDICES + (
    len(V1_FEATURE_NAMES),
    len(V1_FEATURE_NAMES) + 1,
    len(V1_FEATURE_NAMES) + 2,
)
FEATURE_SPEC_SHA256 = hashlib.sha256(json.dumps({
    "schema": SCHEMA,
    "feature_names": FEATURE_NAMES,
    "normalized_indices": NORMALIZED_INDICES,
}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def encode_observation(value):
    """Validate and flatten a v2 observation; invalid TTC remains explicit."""
    validate_observation_v2(value)
    legacy = dict(value)
    safety = legacy.pop("safety_state")
    legacy["schema"] = OBSERVATION_SCHEMA
    output = list(encode_v1_observation(legacy))
    output.extend((
        float(safety["bucket_clearance_m"]),
        float(safety["bucket_descent_rate_mps"]),
        (float(safety["predicted_ground_ttc_s"])
         if safety["predicted_ground_ttc_valid"] else 0.0),
        1.0 if safety["predicted_ground_ttc_valid"] else 0.0,
        1.0 if safety["bucket_end_stop_confirmed"] else 0.0,
    ))
    if len(output) != FEATURE_COUNT:
        raise RuntimeError("actor_vector_v2_feature_count_mismatch")
    if not all(math.isfinite(item) for item in output):
        raise RuntimeError("actor_vector_v2_nonfinite")
    return tuple(output)


def feature_contract():
    return {
        "schema": SCHEMA,
        "feature_count": FEATURE_COUNT,
        "feature_spec_sha256": FEATURE_SPEC_SHA256,
        "parent_vector_schema": "excavation_actor_vector_v1",
        "added_safety_features": list(SAFETY_FEATURE_NAMES),
        "normalization": (
            "fit mean/std on training episodes for continuous indices only; "
            "validity and end-stop indicators remain 0/1"),
        "invalid_ttc": "encoded as value 0 only alongside a 0 validity bit",
        "excluded_actor_truth": [
            "medium_id", "fill_ratio_truth", "bucket_particle_count",
            "particle_id", "mpm_internal_plastic_state",
        ],
    }
