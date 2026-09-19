"""Fill-blind task-phase actor vector for supervised takeover data."""

from __future__ import annotations

import hashlib
import json
import math

from contracts import (OBSERVATION_SCHEMA_V2, TASK_PHASES,
                       validate_observation_v3)
from observation_vector_v2 import (
    FEATURE_NAMES as V2_FEATURE_NAMES,
    NORMALIZED_INDICES as V2_NORMALIZED_INDICES,
    encode_observation as encode_v2_observation,
)


SCHEMA = "excavation_actor_vector_v3"
TASK_PHASE_FEATURE_NAMES = tuple(
    f"task_phase.{phase}" for phase in TASK_PHASES)
FEATURE_NAMES = V2_FEATURE_NAMES + TASK_PHASE_FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)
NORMALIZED_INDICES = V2_NORMALIZED_INDICES
FEATURE_SPEC_SHA256 = hashlib.sha256(json.dumps({
    "schema": SCHEMA,
    "feature_names": FEATURE_NAMES,
    "normalized_indices": NORMALIZED_INDICES,
}, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def encode_observation(value):
    """Validate and flatten v3 with a one-hot phase that leaks no fill value."""
    validate_observation_v3(value)
    task_phase = value["task_phase"]
    legacy = dict(value)
    legacy.pop("task_phase")
    legacy["schema"] = OBSERVATION_SCHEMA_V2
    output = list(encode_v2_observation(legacy))
    output.extend(1.0 if task_phase == phase else 0.0 for phase in TASK_PHASES)
    if len(output) != FEATURE_COUNT:
        raise RuntimeError("actor_vector_v3_feature_count_mismatch")
    if not all(math.isfinite(item) for item in output):
        raise RuntimeError("actor_vector_v3_nonfinite")
    return tuple(output)


def feature_contract():
    return {
        "schema": SCHEMA,
        "feature_count": FEATURE_COUNT,
        "feature_spec_sha256": FEATURE_SPEC_SHA256,
        "parent_vector_schema": "excavation_actor_vector_v2",
        "added_task_phase_features": list(TASK_PHASE_FEATURE_NAMES),
        "task_phase_semantics": (
            "one-hot phase derived from the shared termination supervisor; "
            "LOAD_CORRECTION and EVALUATING collapse to carry so fill threshold "
            "and fill ratio are not exposed to the actor"),
        "excluded_actor_truth": [
            "medium_id", "fill_ratio_truth", "bucket_particle_count",
            "particle_id", "mpm_internal_plastic_state",
        ],
    }
