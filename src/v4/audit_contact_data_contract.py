"""Read-only audit of the M1 contact-policy data and action contract.

The actor index is the only authority used to select training episodes.  This
tool deliberately accepts no holdout path and never walks the archive tree.
It validates causal timing, four-axis action expansion, observation chaining,
and the DAgger correction-to-policy-log correspondence without running physics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from contracts import (
    ACTION_SCHEMA,
    ACTION_SPEED_LIMIT_SI,
    AXES,
    OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMA_V2,
    OBSERVATION_SCHEMA_V3,
    FourAxisAction,
    validate_observation,
    validate_observation_v2,
    validate_observation_v3,
)
from runtime_adapter import (
    MECHANISM_DT_S,
    MECHANISM_STEPS_PER_ACTION,
    POLICY_DT_S,
    SOIL_STEPS_PER_ACTION,
    TERRAIN_ACTION_STRIDE,
)


INDEX_SCHEMA = "excavation_v4_actor_data_index_record_v1"
CORRECTION_SCHEMA = "excavation_dagger_correction_v1"
TOLERANCE = 1e-8
COMMAND_THRESHOLD = 0.1
RESPONSE_THRESHOLD = 1e-4
MAX_ISSUE_SAMPLES = 24
KINEMATIC_MIN_CYLINDER_SPEED_MPS = 0.005
KINEMATIC_MIN_JOINT_SPEED_RADPS = 0.001
KINEMATIC_LOOKUP_BINS = 12
TRAINING_EPISODE_ALLOWLIST = (
    "human-demo-20260917-101448",
    "human-demo-20260917-091339",
    "v4-demo-attempt-20260917-231940",
    "v4-demo-attempt-20260918-102031",
    "v4-demo-attempt-20260918-102212",
    "v4-demo-attempt-20260918-102352",
)
KINEMATIC_PAIRS = {
    "boom": ("chassis piston rod", "boom"),
    "arm": ("boom piston rod", "arm"),
    "bucket": ("arm piston rod", "bucket"),
}


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, tolerance=TOLERANCE):
    return abs(float(left) - float(right)) <= tolerance


def vector_close(left, right, tolerance=TOLERANCE):
    return len(left) == len(right) and all(
        close(a, b, tolerance) for a, b in zip(left, right)
    )


def add_issue(issues, code, **context):
    issues["count"] += 1
    issues["codes"][code] += 1
    if len(issues["samples"]) < MAX_ISSUE_SAMPLES:
        issues["samples"].append({"code": code, **context})


def new_issues():
    return {"count": 0, "codes": Counter(), "samples": []}


def finish_issues(issues):
    return {
        "count": issues["count"],
        "codes": dict(sorted(issues["codes"].items())),
        "samples": issues["samples"],
    }


def percentile(values, percentage):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percentage) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def joint_addresses_from_xml(path):
    """Derive MuJoCo qpos/dof addresses from XML joint declaration order."""
    qpos_address = 0
    dof_address = 0
    joints = {}
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("missing_worldbody")
    for element in worldbody.iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        joint_type = ("free" if element.tag == "freejoint"
                      else element.get("type", "hinge"))
        name = element.get("name")
        if name:
            if name in joints:
                raise ValueError("duplicate_joint_name:" + name)
            joints[name] = {
                "type": joint_type,
                "qpos_address": qpos_address,
                "dof_address": dof_address,
            }
        if joint_type == "free":
            qpos_address += 7
            dof_address += 6
        elif joint_type == "ball":
            qpos_address += 4
            dof_address += 3
        elif joint_type in ("hinge", "slide"):
            qpos_address += 1
            dof_address += 1
        else:
            raise ValueError("unsupported_joint_type:" + str(joint_type))
    required = {"chassis"}
    required.update(name for pair in KINEMATIC_PAIRS.values() for name in pair)
    missing = sorted(required - set(joints))
    if missing:
        raise ValueError("missing_model_joints:" + ",".join(missing))
    return {
        "nq_derived": qpos_address,
        "nv_derived": dof_address,
        "joints": {name: joints[name] for name in sorted(required)},
    }


def validate_actor_action(value):
    if value.get("axis_order") != list(AXES):
        raise ValueError("action_axis_order_changed")
    values = value.get("normalized_velocity")
    if not isinstance(values, list) or len(values) != len(AXES):
        raise ValueError("invalid_action_shape")
    output = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("invalid_action_number")
        number = float(item)
        if not math.isfinite(number) or number < -1.0 or number > 1.0:
            raise ValueError("action_out_of_range")
        output.append(number)
    return output


def load_actor_index(path):
    """Return the explicit non-holdout episode allowlist and index coverage."""
    issues = new_issues()
    groups = {}
    row_count = 0
    for line_number, row in iter_jsonl(path):
        row_count += 1
        episode_id = row.get("episode_id")
        if (not isinstance(episode_id, str) or not episode_id
                or Path(episode_id).name != episode_id):
            add_issue(issues, "unsafe_episode_id", line=line_number)
            continue
        if row.get("schema") != INDEX_SCHEMA:
            add_issue(issues, "index_schema", line=line_number,
                      episode_id=episode_id)
        if row.get("holdout") is not False:
            add_issue(issues, "holdout_row_forbidden", line=line_number,
                      episode_id=episode_id)
        if row.get("training_eligible") is not True:
            add_issue(issues, "ineligible_index_row", line=line_number,
                      episode_id=episode_id)
        if episode_id not in TRAINING_EPISODE_ALLOWLIST:
            add_issue(issues, "episode_not_in_explicit_allowlist",
                      line=line_number, episode_id=episode_id)
        try:
            values = validate_actor_action(row["action"])
        except (KeyError, TypeError, ValueError) as error:
            add_issue(issues, "index_action_contract", line=line_number,
                      episode_id=episode_id, detail=str(error))
            values = [0.0] * len(AXES)
        step = row.get("decision_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            add_issue(issues, "index_decision_step", line=line_number,
                      episode_id=episode_id)
            continue
        group = groups.setdefault(episode_id, {
            "steps": [], "phase_counts": Counter(),
            "axis_supervised_rows": [0] * len(AXES),
        })
        group["steps"].append(step)
        group["phase_counts"][row.get("subphase", "<missing>")] += 1
        mask = row.get("axis_mask")
        if not isinstance(mask, list) or len(mask) != len(AXES):
            add_issue(issues, "index_axis_mask", line=line_number,
                      episode_id=episode_id)
        else:
            for index, item in enumerate(mask):
                if item == 1.0:
                    group["axis_supervised_rows"][index] += 1
        del values

    coverage = {}
    for episode_id, group in groups.items():
        steps = group.pop("steps")
        expected = list(range(min(steps), max(steps) + 1)) if steps else []
        if steps != expected:
            add_issue(issues, "index_steps_not_contiguous", episode_id=episode_id)
        coverage[episode_id] = {
            "indexed_rows": len(steps),
            "first_decision_step": min(steps) if steps else None,
            "last_decision_step": max(steps) if steps else None,
            "phase_counts": dict(sorted(group["phase_counts"].items())),
            "axis_supervised_rows": dict(zip(AXES, group["axis_supervised_rows"])),
        }
    return {
        "row_count": row_count,
        "episode_ids": list(groups),
        "coverage": coverage,
        "issues": finish_issues(issues),
    }


def require_safe_actor_index(index):
    """Fail before constructing or opening any episode path."""
    if index["issues"]["count"]:
        raise ValueError("unsafe_actor_index:" + json.dumps(
            index["issues"]["codes"], sort_keys=True))
    selected = tuple(index["episode_ids"])
    if set(selected) != set(TRAINING_EPISODE_ALLOWLIST):
        raise ValueError("training_episode_allowlist_mismatch")


def validate_observation_by_schema(observation):
    validators = {
        OBSERVATION_SCHEMA: validate_observation,
        OBSERVATION_SCHEMA_V2: validate_observation_v2,
        OBSERVATION_SCHEMA_V3: validate_observation_v3,
    }
    schema = observation.get("schema")
    if schema not in validators:
        raise ValueError("unsupported_observation_schema:" + str(schema))
    validators[schema](observation)


def action_values(value):
    return list(FourAxisAction.parse(value).values)


def wrapped_angle_delta(next_sin_cos, current_sin_cos):
    current = math.atan2(current_sin_cos[0], current_sin_cos[1])
    following = math.atan2(next_sin_cos[0], next_sin_cos[1])
    return math.atan2(math.sin(following - current), math.cos(following - current))


def _new_response():
    return {
        axis: {
            "command_samples": 0,
            "measured_velocity_observable": 0,
            "measured_velocity_sign_agreement": 0,
            "position_delta_observable": 0,
            "position_delta_sign_agreement": 0,
            "single_axis_world_effect": {
                "negative": {"count": 0, "bucket_delta_sum_m": [0.0] * 3,
                             "mouth_delta_sum_deg": 0.0},
                "positive": {"count": 0, "bucket_delta_sum_m": [0.0] * 3,
                             "mouth_delta_sum_deg": 0.0},
            },
        }
        for axis in AXES
    }


def _record_response(response, observation, next_observation, action):
    measured = next_observation["proprioception"]["axis_velocity_normalized"]
    current_cylinders = observation["proprioception"]["cylinder_position_normalized"]
    next_cylinders = next_observation["proprioception"]["cylinder_position_normalized"]
    swing_delta = wrapped_angle_delta(
        next_observation["proprioception"]["swing_sin_cos"],
        observation["proprioception"]["swing_sin_cos"],
    )
    active = [index for index, value in enumerate(action)
              if abs(value) >= COMMAND_THRESHOLD]
    for index, (axis, command) in enumerate(zip(AXES, action)):
        if abs(command) < COMMAND_THRESHOLD:
            continue
        bucket = response[axis]
        bucket["command_samples"] += 1
        if abs(measured[index]) >= RESPONSE_THRESHOLD:
            bucket["measured_velocity_observable"] += 1
            bucket["measured_velocity_sign_agreement"] += int(
                command * measured[index] > 0.0)
        position_delta = (swing_delta if index == 0
                          else next_cylinders[index - 1] - current_cylinders[index - 1])
        if abs(position_delta) >= RESPONSE_THRESHOLD:
            bucket["position_delta_observable"] += 1
            bucket["position_delta_sign_agreement"] += int(
                command * position_delta > 0.0)
        if active == [index]:
            sign = "positive" if command > 0.0 else "negative"
            effect = bucket["single_axis_world_effect"][sign]
            effect["count"] += 1
            current_position = observation["bucket"]["position_world_m"]
            next_position = next_observation["bucket"]["position_world_m"]
            for coordinate in range(3):
                effect["bucket_delta_sum_m"][coordinate] += (
                    next_position[coordinate] - current_position[coordinate])
            effect["mouth_delta_sum_deg"] += (
                next_observation["bucket"]["mouth_angle_deg"]
                - observation["bucket"]["mouth_angle_deg"])


def _finish_response(response):
    for bucket in response.values():
        velocity_count = bucket["measured_velocity_observable"]
        position_count = bucket["position_delta_observable"]
        bucket["measured_velocity_sign_agreement_fraction"] = (
            bucket["measured_velocity_sign_agreement"] / velocity_count
            if velocity_count else None)
        bucket["position_delta_sign_agreement_fraction"] = (
            bucket["position_delta_sign_agreement"] / position_count
            if position_count else None)
        for effect in bucket["single_axis_world_effect"].values():
            count = effect.pop("count")
            sums = effect.pop("bucket_delta_sum_m")
            mouth_sum = effect.pop("mouth_delta_sum_deg")
            effect.update({
                "sample_count": count,
                "mean_bucket_delta_m_per_policy_step": (
                    [value / count for value in sums] if count else None),
                "mean_mouth_delta_deg_per_policy_step": (
                    mouth_sum / count if count else None),
            })
    return response


def analyze_policy_episode(path, indexed_steps=None):
    issues = new_issues()
    row_count = 0
    phase_counts = Counter()
    source_counts = Counter()
    action_sign_counts = {
        phase: {axis: Counter() for axis in AXES} for phase in ()
    }
    response = _new_response()
    previous_step = None
    previous_action = None
    previous_next_observation = None
    raw_action_present = 0
    indexed_rows_observed = 0

    for line_number, row in iter_jsonl(path):
        row_count += 1
        step = row.get("decision_step")
        if previous_step is not None and step != previous_step + 1:
            add_issue(issues, "policy_steps_not_contiguous", line=line_number,
                      decision_step=step, previous_decision_step=previous_step)
        if previous_step is None and step != 0:
            add_issue(issues, "policy_does_not_start_at_zero", line=line_number,
                      decision_step=step)
        previous_step = step
        phase = row.get("phase", "<missing>")
        source = row.get("action_source", "<missing>")
        phase_counts[phase] += 1
        source_counts[source] += 1
        if phase not in action_sign_counts:
            action_sign_counts[phase] = {axis: Counter() for axis in AXES}

        try:
            action = action_values(row["action"])
            if "raw_action" in row:
                raw_action_present += 1
                action_values(row["raw_action"])
        except (KeyError, TypeError, ValueError) as error:
            add_issue(issues, "policy_action_contract", line=line_number,
                      decision_step=step, detail=str(error))
            action = [0.0] * len(AXES)

        observation = row.get("observation")
        next_observation = row.get("next_observation")
        for label, value in (("observation", observation),
                             ("next_observation", next_observation)):
            try:
                validate_observation_by_schema(value)
            except (AttributeError, TypeError, ValueError) as error:
                add_issue(issues, label + "_contract", line=line_number,
                          decision_step=step, detail=str(error))
        if isinstance(observation, dict) and isinstance(next_observation, dict):
            if observation.get("decision_step") != step:
                add_issue(issues, "observation_step_mismatch", line=line_number,
                          decision_step=step)
            if next_observation.get("decision_step") != step + 1:
                add_issue(issues, "next_observation_step_mismatch", line=line_number,
                          decision_step=step)
            if previous_next_observation is not None and observation != previous_next_observation:
                add_issue(issues, "observation_chain_broken", line=line_number,
                          decision_step=step)
            previous_next_observation = next_observation
            expected_previous = ([0.0] * len(AXES)
                                 if previous_action is None else previous_action)
            observed_previous = observation.get("previous_action", {}).get(
                "normalized_velocity", [])
            next_previous = next_observation.get("previous_action", {}).get(
                "normalized_velocity", [])
            if not vector_close(observed_previous, expected_previous):
                add_issue(issues, "previous_action_causality", line=line_number,
                          decision_step=step)
            if not vector_close(next_previous, action):
                add_issue(issues, "next_previous_action_causality", line=line_number,
                          decision_step=step)
            if not close(next_observation.get("sim_time_s", -1.0)
                         - observation.get("sim_time_s", 0.0), POLICY_DT_S):
                add_issue(issues, "policy_dt", line=line_number,
                          decision_step=step)
            try:
                _record_response(response, observation, next_observation, action)
            except (KeyError, TypeError, ValueError) as error:
                add_issue(issues, "response_fields", line=line_number,
                          decision_step=step, detail=str(error))
        previous_action = action

        plan = row.get("plan", {})
        expected_physical = [value * limit for value, limit
                             in zip(action, ACTION_SPEED_LIMIT_SI)]
        if (plan.get("decision_step") != step
                or plan.get("action") != row.get("action")):
            add_issue(issues, "plan_identity", line=line_number,
                      decision_step=step)
        if isinstance(observation, dict) and (
                not close(plan.get("start_sim_s", -1.0),
                          observation.get("sim_time_s", -2.0))
                or not close(plan.get("end_sim_s", -1.0),
                             next_observation.get("sim_time_s", -2.0))):
            add_issue(issues, "plan_time", line=line_number, decision_step=step)
        commands = plan.get("mechanism_commands", [])
        if (len(commands) != MECHANISM_STEPS_PER_ACTION
                or plan.get("soil_steps") != SOIL_STEPS_PER_ACTION
                or plan.get("refresh_terrain_after")
                is not ((step + 1) % TERRAIN_ACTION_STRIDE == 0)):
            add_issue(issues, "plan_rate_expansion", line=line_number,
                      decision_step=step)
        for index, command in enumerate(commands):
            if (command.get("substep") != index + 1
                    or command.get("axis_order") != list(AXES)
                    or not close(command.get("dt_s", -1.0), MECHANISM_DT_S)
                    or not vector_close(command.get("physical_velocity_si", []),
                                        expected_physical)):
                add_issue(issues, "mechanism_command", line=line_number,
                          decision_step=step, substep=index + 1)
                break

        for axis, value in zip(AXES, action):
            sign = "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
            action_sign_counts[phase][axis][sign] += 1
        if indexed_steps is not None and step in indexed_steps:
            indexed_rows_observed += 1

    return {
        "row_count": row_count,
        "first_decision_step": 0 if row_count else None,
        "last_decision_step": previous_step,
        "indexed_rows_observed": indexed_rows_observed,
        "raw_action_field_rows": raw_action_present,
        "action_source_counts": dict(sorted(source_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "action_sign_counts_by_phase": {
            phase: {axis: dict(action_sign_counts[phase][axis]) for axis in AXES}
            for phase in sorted(action_sign_counts)
        },
        "empirical_one_step_response": _finish_response(response),
        "issues": finish_issues(issues),
    }


def analyze_corrections(correction_path, policy_path):
    issues = new_issues()
    corrections = {}
    raw_axis_sign_counts = {axis: Counter() for axis in AXES}
    applied_axis_sign_counts = {axis: Counter() for axis in AXES}
    zero_action_rows = 0
    projected_rows = 0
    takeover_event_steps = []
    command_segments = []
    segment_signs = None
    segment_start = None
    segment_end = None
    previous_step = None
    for line_number, row in iter_jsonl(correction_path):
        step = row.get("decision_step")
        if previous_step is not None and step != previous_step + 1:
            add_issue(issues, "correction_steps_not_contiguous", line=line_number,
                      decision_step=step)
        previous_step = step
        if row.get("schema") != CORRECTION_SCHEMA:
            add_issue(issues, "correction_schema", line=line_number,
                      decision_step=step)
        if row.get("observation", {}).get("decision_step") != step:
            add_issue(issues, "correction_observation_step", line=line_number,
                      decision_step=step)
        try:
            raw = action_values(row["raw_human_action"])
            applied = action_values(row["applied_action"])
        except (KeyError, TypeError, ValueError) as error:
            add_issue(issues, "correction_action_contract", line=line_number,
                      decision_step=step, detail=str(error))
            raw = applied = [0.0] * len(AXES)
        if all(value == 0.0 for value in raw):
            zero_action_rows += 1
        for axis, raw_value, applied_value in zip(AXES, raw, applied):
            raw_axis_sign_counts[axis][
                "positive" if raw_value > 0.0
                else "negative" if raw_value < 0.0 else "zero"] += 1
            applied_axis_sign_counts[axis][
                "positive" if applied_value > 0.0
                else "negative" if applied_value < 0.0 else "zero"] += 1
        signs = tuple(1 if value > 0.0 else -1 if value < 0.0 else 0
                      for value in raw)
        if signs != segment_signs:
            if segment_signs is not None:
                command_segments.append({
                    "first_decision_step": segment_start,
                    "last_decision_step": segment_end,
                    "row_count": segment_end - segment_start + 1,
                    "raw_action_sign": dict(zip(AXES, segment_signs)),
                })
            segment_signs = signs
            segment_start = step
        segment_end = step
        projection = row.get("end_stop_projection", {})
        projected_rows += int(projection.get("bucket_action_projected") is True)
        if row.get("takeover_first_nonzero_action") is True:
            takeover_event_steps.append(step)
            if not isinstance(row.get("takeover_handshake_event"), dict):
                add_issue(issues, "missing_takeover_event", line=line_number,
                          decision_step=step)
        if row.get("critic_eligible") is not False:
            add_issue(issues, "correction_critic_eligibility", line=line_number,
                      decision_step=step)
        if row.get("autonomous_success_eligible") is not False:
            add_issue(issues, "correction_success_eligibility", line=line_number,
                      decision_step=step)
        if step in corrections:
            add_issue(issues, "duplicate_correction_step", line=line_number,
                      decision_step=step)
        corrections[step] = row

    if segment_signs is not None:
        command_segments.append({
            "first_decision_step": segment_start,
            "last_decision_step": segment_end,
            "row_count": segment_end - segment_start + 1,
            "raw_action_sign": dict(zip(AXES, segment_signs)),
        })

    policy_source_counts = Counter()
    matched = 0
    for line_number, row in iter_jsonl(policy_path):
        policy_source_counts[row.get("action_source", "<missing>")] += 1
        step = row.get("decision_step")
        if step not in corrections:
            continue
        correction = corrections[step]
        matched += 1
        if row.get("action_source") != "human_correction":
            add_issue(issues, "correction_policy_source", line=line_number,
                      decision_step=step)
        comparisons = (
            ("raw_action", row.get("raw_action"), correction.get("raw_human_action")),
            ("applied_action", row.get("action"), correction.get("applied_action")),
            ("end_stop_projection", row.get("end_stop_projection"),
             correction.get("end_stop_projection")),
            ("observation", row.get("observation"), correction.get("observation")),
        )
        for label, policy_value, correction_value in comparisons:
            if policy_value != correction_value:
                add_issue(issues, "correction_policy_" + label,
                          line=line_number, decision_step=step)
    if matched != len(corrections):
        add_issue(issues, "correction_match_count", matched=matched,
                  correction_rows=len(corrections))

    first_step = min(corrections) if corrections else None
    last_step = max(corrections) if corrections else None
    return {
        "row_count": len(corrections),
        "first_decision_step": first_step,
        "last_decision_step": last_step,
        "zero_human_action_rows": zero_action_rows,
        "nonzero_human_action_rows": len(corrections) - zero_action_rows,
        "raw_action_sign_counts": {
            axis: dict(raw_axis_sign_counts[axis]) for axis in AXES
        },
        "applied_action_sign_counts": {
            axis: dict(applied_axis_sign_counts[axis]) for axis in AXES
        },
        "constant_sign_command_segments": command_segments,
        "constant_sign_command_segment_count": len(command_segments),
        "bucket_end_stop_projected_rows": projected_rows,
        "takeover_event_steps": takeover_event_steps,
        "matched_policy_rows": matched,
        "policy_action_source_counts": dict(sorted(policy_source_counts.items())),
        "issues": finish_issues(issues),
    }


def _new_kinematic_axis():
    return {
        "observed_cylinder_positions_m": [],
        "observed_joint_angles_rad": [],
        "observed_joint_velocities_radps": [],
        "mapping_samples": [],
        "ratio_sign_counts": Counter(),
    }


def _finish_kinematic_axis(axis, values):
    positions = values["observed_cylinder_positions_m"]
    angles = values["observed_joint_angles_rad"]
    joint_velocities = values["observed_joint_velocities_radps"]
    samples = values["mapping_samples"]
    ratios = [ratio for _, ratio in samples]
    absolute_ratios = [abs(value) for value in ratios]
    bins = []
    minimum = min(positions)
    maximum = max(positions)
    width = (maximum - minimum) / KINEMATIC_LOOKUP_BINS
    grouped = [[] for _ in range(KINEMATIC_LOOKUP_BINS)]
    if width > 0.0:
        for position, ratio in samples:
            index = min(KINEMATIC_LOOKUP_BINS - 1,
                        int((position - minimum) / width))
            grouped[index].append(ratio)
    for index, group in enumerate(grouped):
        lower = minimum + index * width
        upper = maximum if index == KINEMATIC_LOOKUP_BINS - 1 else lower + width
        bins.append({
            "cylinder_position_min_m": lower,
            "cylinder_position_max_m": upper,
            "sample_count": len(group),
            "d_joint_angle_d_cylinder_length_rad_per_m": {
                "p05": percentile(group, 5),
                "median": percentile(group, 50),
                "p95": percentile(group, 95),
            },
        })
    speed_limit = ACTION_SPEED_LIMIT_SI[list(AXES).index(axis)]
    sign_counts = values["ratio_sign_counts"]
    dominant = max(sign_counts.values()) if sign_counts else 0
    return {
        "semantics": (
            "actual relative joint angular velocity divided by actual cylinder "
            "linear velocity; internal MuJoCo coordinate sign"),
        "mapping_sample_filter": {
            "minimum_abs_cylinder_velocity_mps": KINEMATIC_MIN_CYLINDER_SPEED_MPS,
            "minimum_abs_joint_velocity_radps": KINEMATIC_MIN_JOINT_SPEED_RADPS,
        },
        "mapping_sample_count": len(samples),
        "observed_cylinder_position_range_m": [min(positions), max(positions)],
        "observed_joint_angle_range_deg": [
            math.degrees(min(angles)), math.degrees(max(angles))],
        "observed_abs_joint_velocity_radps": {
            "p95": percentile([abs(value) for value in joint_velocities], 95),
            "maximum": max(abs(value) for value in joint_velocities),
        },
        "d_joint_angle_d_cylinder_length_rad_per_m": {
            "p01": percentile(ratios, 1),
            "p05": percentile(ratios, 5),
            "median": percentile(ratios, 50),
            "p95": percentile(ratios, 95),
            "p99": percentile(ratios, 99),
        },
        "absolute_ratio_rad_per_m": {
            "p05": percentile(absolute_ratios, 5),
            "median": percentile(absolute_ratios, 50),
            "p95": percentile(absolute_ratios, 95),
        },
        "ratio_sign_counts": dict(sign_counts),
        "dominant_sign_fraction": dominant / len(samples) if samples else None,
        "implied_abs_joint_speed_at_cylinder_action_limit_radps": {
            "cylinder_action_limit_mps": speed_limit,
            "p05": percentile(absolute_ratios, 5) * speed_limit,
            "median": percentile(absolute_ratios, 50) * speed_limit,
            "p95": percentile(absolute_ratios, 95) * speed_limit,
        },
        "pose_dependent_lookup": bins,
    }


def analyze_kinematic_logs(training_root, episode_ids, mechanism_model):
    model = joint_addresses_from_xml(mechanism_model)
    issues = new_issues()
    aggregate = {axis: _new_kinematic_axis() for axis in KINEMATIC_PAIRS}
    episode_reports = {}
    maximum_required_qpos = max(
        model["joints"][name]["qpos_address"]
        for pair in KINEMATIC_PAIRS.values() for name in pair)
    maximum_required_dof = max(
        model["joints"][name]["dof_address"]
        for pair in KINEMATIC_PAIRS.values() for name in pair)

    for episode_id in episode_ids:
        path = Path(training_root) / episode_id / "mechanism-steps.ndjson"
        episode_issues = new_issues()
        row_count = 0
        previous_step = None
        previous_time = None
        maximum_closure_m = 0.0
        episode_samples = Counter()
        for line_number, row in iter_jsonl(path):
            row_count += 1
            step = row.get("mechanism_step")
            time_s = row.get("time_s")
            if previous_step is None and step != 1:
                add_issue(episode_issues, "mechanism_does_not_start_at_one",
                          line=line_number, mechanism_step=step)
            if previous_step is not None and step != previous_step + 1:
                add_issue(episode_issues, "mechanism_steps_not_contiguous",
                          line=line_number, mechanism_step=step)
            if previous_time is not None and not close(
                    float(time_s) - previous_time, MECHANISM_DT_S, 1e-7):
                add_issue(episode_issues, "mechanism_dt", line=line_number,
                          mechanism_step=step)
            previous_step = step
            previous_time = float(time_s)
            closure = float(row.get("closure_m", float("inf")))
            maximum_closure_m = max(maximum_closure_m, closure)
            if not math.isfinite(closure) or closure > 0.001 + TOLERANCE:
                add_issue(episode_issues, "mechanism_closure", line=line_number,
                          mechanism_step=step, closure_m=closure)
            qpos = row.get("qpos", [])
            qvel = row.get("qvel", [])
            if (len(qpos) <= maximum_required_qpos
                    or len(qvel) <= maximum_required_dof
                    or not all(isinstance(value, (int, float))
                               and not isinstance(value, bool)
                               and math.isfinite(float(value))
                               for value in qpos + qvel)):
                add_issue(episode_issues, "mechanism_state_shape_or_finite",
                          line=line_number, mechanism_step=step)
                continue
            for axis, (cylinder_name, joint_name) in KINEMATIC_PAIRS.items():
                cylinder = model["joints"][cylinder_name]
                joint = model["joints"][joint_name]
                cylinder_position = float(qpos[cylinder["qpos_address"]])
                joint_angle = float(qpos[joint["qpos_address"]])
                cylinder_velocity = float(qvel[cylinder["dof_address"]])
                joint_velocity = float(qvel[joint["dof_address"]])
                bucket = aggregate[axis]
                bucket["observed_cylinder_positions_m"].append(cylinder_position)
                bucket["observed_joint_angles_rad"].append(joint_angle)
                bucket["observed_joint_velocities_radps"].append(joint_velocity)
                if (abs(cylinder_velocity) >= KINEMATIC_MIN_CYLINDER_SPEED_MPS
                        and abs(joint_velocity) >= KINEMATIC_MIN_JOINT_SPEED_RADPS):
                    ratio = joint_velocity / cylinder_velocity
                    bucket["mapping_samples"].append((cylinder_position, ratio))
                    bucket["ratio_sign_counts"][
                        "positive" if ratio > 0.0 else "negative"] += 1
                    episode_samples[axis] += 1
        finished_episode_issues = finish_issues(episode_issues)
        for code, count in finished_episode_issues["codes"].items():
            issues["count"] += count
            issues["codes"][code] += count
        for sample in finished_episode_issues["samples"]:
            if len(issues["samples"]) < MAX_ISSUE_SAMPLES:
                issues["samples"].append({"episode_id": episode_id, **sample})
        episode_reports[episode_id] = {
            "row_count": row_count,
            "mapping_sample_counts": {
                axis: episode_samples[axis] for axis in KINEMATIC_PAIRS},
            "maximum_closure_m": maximum_closure_m,
            "source_sha256": sha256(path),
            "issues": finished_episode_issues,
        }

    axes = {
        axis: _finish_kinematic_axis(axis, values)
        for axis, values in aggregate.items()
    }
    return {
        "model_path": str(mechanism_model),
        "model_sha256": sha256(mechanism_model),
        "derived_model_dimensions": {
            "nq": model["nq_derived"], "nv": model["nv_derived"]},
        "joint_addresses": model["joints"],
        "swing_mapping": {
            "input": "swing angular velocity rad/s",
            "output": "chassis relative angular velocity rad/s",
            "mapping": "identity before lower-level tracking dynamics",
        },
        "cylinder_to_relative_joint_mapping": axes,
        "episode_reports": episode_reports,
        "issues": finish_issues(issues),
        "scope": {
            "physics_steps": 0,
            "source": "recorded training mechanism logs only",
            "holdout_files_opened": 0,
            "warning": (
                "lookup describes this MuJoCo mechanism only; official measured "
                "joint velocity remains a state outcome, not an operator command"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-index", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--mechanism-model", type=Path, required=True)
    parser.add_argument("--correction-log", type=Path, required=True)
    parser.add_argument("--correction-policy-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))

    index = load_actor_index(args.actor_index)
    require_safe_actor_index(index)
    episodes = {}
    for episode_id in index["episode_ids"]:
        policy_path = args.training_root / episode_id / "policy-steps.ndjson"
        indexed = index["coverage"][episode_id]
        indexed_steps = set(range(indexed["first_decision_step"],
                                  indexed["last_decision_step"] + 1))
        analysis = analyze_policy_episode(policy_path, indexed_steps=indexed_steps)
        analysis["source_sha256"] = sha256(policy_path)
        analysis["index_coverage"] = {
            **indexed,
            "source_rows_not_indexed": analysis["row_count"] - indexed["indexed_rows"],
        }
        episodes[episode_id] = analysis

    correction_policy = analyze_policy_episode(args.correction_policy_log)
    correction_policy["source_sha256"] = sha256(args.correction_policy_log)
    corrections = analyze_corrections(args.correction_log,
                                      args.correction_policy_log)
    corrections["source_sha256"] = sha256(args.correction_log)
    kinematics = analyze_kinematic_logs(
        args.training_root, index["episode_ids"], args.mechanism_model)

    checks = {
        "actor_index_contains_no_holdout_rows": (
            index["issues"]["codes"].get("holdout_row_forbidden", 0) == 0),
        "actor_index_contract_passed": index["issues"]["count"] == 0,
        "six_training_episodes_selected": len(episodes) == 6,
        "training_policy_contracts_passed": all(
            item["issues"]["count"] == 0 for item in episodes.values()),
        "indexed_rows_found_in_sources": all(
            item["indexed_rows_observed"] == item["index_coverage"]["indexed_rows"]
            for item in episodes.values()),
        "correction_policy_contract_passed": correction_policy["issues"]["count"] == 0,
        "corrections_match_policy_log": corrections["issues"]["count"] == 0,
        "one_takeover_handshake_event": len(corrections["takeover_event_steps"]) == 1,
        "kinematic_log_contracts_passed": kinematics["issues"]["count"] == 0,
        "kinematic_mapping_has_sufficient_samples": all(
            item["mapping_sample_count"] >= 100
            for item in kinematics[
                "cylinder_to_relative_joint_mapping"].values()),
        "kinematic_mapping_dominant_sign_ge_99_percent": all(
            item["dominant_sign_fraction"] >= 0.99
            for item in kinematics[
                "cylinder_to_relative_joint_mapping"].values()),
        "corrections_are_not_critic_or_success_evidence": (
            corrections["issues"]["codes"].get("correction_critic_eligibility", 0) == 0
            and corrections["issues"]["codes"].get(
                "correction_success_eligibility", 0) == 0),
        "holdout_files_opened_zero": True,
        "physics_steps_zero": True,
        "parameter_updates_zero": True,
    }
    report = {
        "schema": "excavation_contact_data_contract_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "frozen_action_contract": {
            "schema": ACTION_SCHEMA,
            "axis_order": list(AXES),
            "semantics": "simultaneous normalized velocity requests",
            "speed_limit_si": dict(zip(AXES, ACTION_SPEED_LIMIT_SI)),
            "speed_unit": {"swing": "rad/s", "boom": "m/s",
                           "arm": "m/s", "bucket": "m/s"},
            "policy_hz": round(1.0 / POLICY_DT_S),
            "mechanism_hz": round(1.0 / MECHANISM_DT_S),
            "soil_hz": round(SOIL_STEPS_PER_ACTION / POLICY_DT_S),
        },
        "actor_index": index,
        "training_episodes": episodes,
        "correction_policy_episode": correction_policy,
        "dagger_corrections": corrections,
        "model_kinematics": kinematics,
        "scope": {
            "generator_sha256": sha256(Path(__file__)),
            "training_episode_allowlist": list(TRAINING_EPISODE_ALLOWLIST),
            "input_paths": {
                "actor_index": str(args.actor_index),
                "training_root": str(args.training_root),
                "mechanism_model": str(args.mechanism_model),
                "correction_log": str(args.correction_log),
                "correction_policy_log": str(args.correction_policy_log),
            },
            "holdout_path_argument_supported": False,
            "holdout_files_opened": 0,
            "physics_steps": 0,
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "claim": "fixed-log contract and correspondence audit only",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.sha256").write_text(
        f"{sha256(report_path)}  ./report.json\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "checks": checks,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
