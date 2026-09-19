"""Strict zero-physics contracts for the four-axis excavation policy.

Parsing and validation in this module are side-effect free.  They must be run
before constructing a simulator or opening an episode output directory.
"""
from dataclasses import dataclass
import json
import math


ACTION_SCHEMA = "excavation_four_axis_action_v1"
OBSERVATION_SCHEMA = "excavation_four_axis_observation_v1"
OBSERVATION_SCHEMA_V2 = "excavation_four_axis_observation_v2"
OBSERVATION_SCHEMA_V3 = "excavation_four_axis_observation_v3"
TASK_PHASES = (
    "digging", "off_soil_confirming", "waiting_for_bucket_closure", "carry",
)
AXES = ("swing", "boom", "arm", "bucket")
ACTION_SPEED_LIMIT_SI = (0.12, 0.0225, 0.0225, 0.0675)
ACTION_SPEED_UNITS = ("rad/s", "m/s", "m/s", "m/s")
TERRAIN_SHAPE_YX = (30, 38)
WRENCH_HISTORY_LENGTH = 4


def _finite_number(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not_number:" + name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite:" + name)
    if low is not None and value < low:
        raise ValueError("below_range:" + name)
    if high is not None and value > high:
        raise ValueError("above_range:" + name)
    return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("invalid_integer:" + name)
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError("not_boolean:" + name)
    return value


def _exact_fields(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError("not_object:" + name)
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if missing:
        raise ValueError("missing_fields:%s:%s" % (name, ",".join(sorted(missing))))
    if unknown:
        raise ValueError("unknown_fields:%s:%s" % (name, ",".join(sorted(unknown))))


def _vector(value, length, name, *, low=None, high=None):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("invalid_shape:" + name)
    return tuple(_finite_number(item, "%s[%d]" % (name, index), low=low, high=high)
                 for index, item in enumerate(value))


def _bool_vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("invalid_shape:" + name)
    return tuple(_boolean(item, "%s[%d]" % (name, index))
                 for index, item in enumerate(value))


def _reject_duplicates(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_field:" + str(key))
        output[key] = value
    return output


def strict_json_loads(text):
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("nonfinite_json_constant:" + token)),
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid_json") from error


@dataclass(frozen=True)
class FourAxisAction:
    """Normalized simultaneous speed requests in the frozen axis order."""

    values: tuple

    @classmethod
    def parse(cls, value):
        _exact_fields(value, ("schema", "axis_order", "normalized_velocity"), "action")
        if value["schema"] != ACTION_SCHEMA:
            raise ValueError("unsupported_action_schema")
        if value["axis_order"] != list(AXES):
            raise ValueError("action_axis_order_changed")
        values = _vector(value["normalized_velocity"], 4, "normalized_velocity",
                         low=-1.0, high=1.0)
        return cls(values=values)

    @classmethod
    def loads(cls, text):
        return cls.parse(strict_json_loads(text))

    def as_dict(self):
        return {
            "schema": ACTION_SCHEMA,
            "axis_order": list(AXES),
            "normalized_velocity": list(self.values),
        }

    def physical_velocity(self):
        return tuple(value * limit for value, limit in zip(self.values, ACTION_SPEED_LIMIT_SI))


def _validate_terrain(value, sim_time_s):
    _exact_fields(value, (
        "height_m", "known_mask", "valid", "capture_sim_s", "age_s", "sequence",
    ), "terrain")
    valid = _boolean(value["valid"], "terrain.valid")
    capture = _finite_number(value["capture_sim_s"], "terrain.capture_sim_s", low=0.0)
    age = _finite_number(value["age_s"], "terrain.age_s", low=0.0)
    if capture > sim_time_s + 1e-9 or abs((sim_time_s - capture) - age) > 1e-8:
        raise ValueError("terrain_time_mismatch")
    _integer(value["sequence"], "terrain.sequence")
    heights = value["height_m"]
    known = value["known_mask"]
    if (not isinstance(heights, list) or len(heights) != TERRAIN_SHAPE_YX[0]
            or not isinstance(known, list) or len(known) != TERRAIN_SHAPE_YX[0]):
        raise ValueError("invalid_shape:terrain")
    known_count = 0
    for y, (height_row, known_row) in enumerate(zip(heights, known)):
        if (not isinstance(height_row, list) or len(height_row) != TERRAIN_SHAPE_YX[1]
                or not isinstance(known_row, list) or len(known_row) != TERRAIN_SHAPE_YX[1]):
            raise ValueError("invalid_shape:terrain_row")
        for x, (height, is_known) in enumerate(zip(height_row, known_row)):
            _boolean(is_known, "terrain.known_mask[%d][%d]" % (y, x))
            if is_known:
                _finite_number(height, "terrain.height_m[%d][%d]" % (y, x))
                known_count += 1
            elif height is not None:
                raise ValueError("unknown_terrain_must_be_null")
    if valid and known_count == 0:
        raise ValueError("terrain_validity_mismatch")


def _validate_proprioception(value):
    _exact_fields(value, (
        "swing_sin_cos", "cylinder_position_normalized", "axis_velocity_normalized",
        "axis_valid_mask",
    ), "proprioception")
    swing = _vector(value["swing_sin_cos"], 2, "swing_sin_cos", low=-1.0, high=1.0)
    if abs(sum(item * item for item in swing) - 1.0) > 1e-5:
        raise ValueError("swing_sin_cos_not_unit")
    # Measured state may exceed a nominal normalization scale immediately
    # before a separately reported physical hard failure. Do not clip or reject
    # finite evidence here; action requests remain strictly bounded.
    _vector(value["cylinder_position_normalized"], 3,
            "cylinder_position_normalized")
    _vector(value["axis_velocity_normalized"], 4,
            "axis_velocity_normalized")
    _bool_vector(value["axis_valid_mask"], 4, "axis_valid_mask")


def _validate_bucket(value):
    _exact_fields(value, (
        "position_world_m", "orientation_quaternion_wxyz", "relative_to_target_m",
        "mouth_angle_deg", "valid",
    ), "bucket")
    _vector(value["position_world_m"], 3, "bucket.position_world_m")
    quaternion = _vector(value["orientation_quaternion_wxyz"], 4,
                         "bucket.orientation_quaternion_wxyz")
    if abs(sum(item * item for item in quaternion) - 1.0) > 1e-5:
        raise ValueError("bucket_quaternion_not_unit")
    _vector(value["relative_to_target_m"], 3, "bucket.relative_to_target_m")
    _finite_number(value["mouth_angle_deg"], "bucket.mouth_angle_deg")
    _boolean(value["valid"], "bucket.valid")


def _validate_wrench_history(value):
    _exact_fields(value, ("values_world_si", "age_s", "valid_mask"), "wrench_history")
    samples = value["values_world_si"]
    ages = value["age_s"]
    valid = value["valid_mask"]
    if not isinstance(samples, list) or len(samples) != WRENCH_HISTORY_LENGTH:
        raise ValueError("invalid_shape:wrench_history")
    _bool_vector(valid, WRENCH_HISTORY_LENGTH, "wrench_history.valid_mask")
    parsed_ages = _vector(ages, WRENCH_HISTORY_LENGTH, "wrench_history.age_s", low=0.0)
    for index, (sample, is_valid) in enumerate(zip(samples, valid)):
        if is_valid:
            _vector(sample, 6, "wrench_history.values_world_si[%d]" % index)
        elif sample is not None:
            raise ValueError("invalid_wrench_sample_must_be_null")
    valid_ages = [age for age, is_valid in zip(parsed_ages, valid) if is_valid]
    if any(left > right + 1e-9 for left, right in zip(valid_ages, valid_ages[1:])):
        raise ValueError("wrench_history_not_newest_first")


def validate_observation(value):
    """Validate the actor-visible observation; return it unchanged on success."""
    _exact_fields(value, (
        "schema", "decision_step", "sim_time_s", "terrain", "target_world_m",
        "target_valid", "proprioception", "bucket", "wrench_history",
        "previous_action", "elapsed_fraction", "digging_interaction_observed",
    ), "observation")
    if value["schema"] != OBSERVATION_SCHEMA:
        raise ValueError("unsupported_observation_schema")
    _integer(value["decision_step"], "decision_step")
    sim_time_s = _finite_number(value["sim_time_s"], "sim_time_s", low=0.0, high=60.0)
    _validate_terrain(value["terrain"], sim_time_s)
    _vector(value["target_world_m"], 3, "target_world_m")
    _boolean(value["target_valid"], "target_valid")
    _validate_proprioception(value["proprioception"])
    _validate_bucket(value["bucket"])
    _validate_wrench_history(value["wrench_history"])
    FourAxisAction.parse(value["previous_action"])
    elapsed = _finite_number(value["elapsed_fraction"], "elapsed_fraction", low=0.0, high=1.0)
    if abs(elapsed - sim_time_s / 60.0) > 1e-8:
        raise ValueError("elapsed_fraction_mismatch")
    _boolean(value["digging_interaction_observed"], "digging_interaction_observed")
    return value


def _validate_safety_state(value):
    _exact_fields(value, (
        "bucket_clearance_m", "bucket_descent_rate_mps",
        "predicted_ground_ttc_s", "predicted_ground_ttc_valid",
        "bucket_end_stop_confirmed",
    ), "safety_state")
    _finite_number(value["bucket_clearance_m"], "safety_state.bucket_clearance_m")
    _finite_number(
        value["bucket_descent_rate_mps"],
        "safety_state.bucket_descent_rate_mps", low=0.0)
    ttc_valid = _boolean(
        value["predicted_ground_ttc_valid"],
        "safety_state.predicted_ground_ttc_valid")
    ttc = value["predicted_ground_ttc_s"]
    if ttc_valid:
        _finite_number(ttc, "safety_state.predicted_ground_ttc_s", low=0.0)
    elif ttc is not None:
        raise ValueError("invalid_ttc_must_be_null")
    _boolean(
        value["bucket_end_stop_confirmed"],
        "safety_state.bucket_end_stop_confirmed")


def validate_observation_v2(value):
    """Validate the safety-augmented actor observation without changing v1."""
    expected = (
        "schema", "decision_step", "sim_time_s", "terrain", "target_world_m",
        "target_valid", "proprioception", "bucket", "wrench_history",
        "previous_action", "elapsed_fraction", "digging_interaction_observed",
        "safety_state",
    )
    _exact_fields(value, expected, "observation")
    if value["schema"] != OBSERVATION_SCHEMA_V2:
        raise ValueError("unsupported_observation_schema")
    legacy = dict(value)
    legacy.pop("safety_state")
    legacy["schema"] = OBSERVATION_SCHEMA
    validate_observation(legacy)
    _validate_safety_state(value["safety_state"])
    return value


def validate_observation_v3(value):
    """Validate safety plus a fill-blind task phase without changing v1/v2."""
    expected = (
        "schema", "decision_step", "sim_time_s", "terrain", "target_world_m",
        "target_valid", "proprioception", "bucket", "wrench_history",
        "previous_action", "elapsed_fraction", "digging_interaction_observed",
        "safety_state", "task_phase",
    )
    _exact_fields(value, expected, "observation")
    if value["schema"] != OBSERVATION_SCHEMA_V3:
        raise ValueError("unsupported_observation_schema")
    legacy = dict(value)
    task_phase = legacy.pop("task_phase")
    legacy["schema"] = OBSERVATION_SCHEMA_V2
    validate_observation_v2(legacy)
    if not isinstance(task_phase, str) or task_phase not in TASK_PHASES:
        raise ValueError("unsupported_task_phase")
    return value
