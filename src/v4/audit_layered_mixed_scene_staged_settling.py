#!/usr/bin/env python3
"""Audit the scaled staged-settling report before full-scale construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


CONFIG_SCHEMA = (
    "excavation_layered_mixed_scene_staged_settling_audit_config_v1")
REPORT_SCHEMA = "excavation_layered_mixed_scene_staged_settling_audit_report_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HISTORY_COMPONENTS = (
    "q", "qd", "elastic_strain", "transform", "qd_grad", "stress", "Jp")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError("staged_settling_number_required:" + name)
    return float(value)


def _nullable_finite_number(value: object, name: str,
                            allow_null: bool) -> float | None:
    if value is None and allow_null:
        return None
    return _finite_number(value, name)


def _check_nonfinite_failure(candidate: dict) -> tuple[bool, str | None]:
    failure = candidate.get("nonfinite_failure")
    if failure is None:
        return True, None
    if not isinstance(failure, dict) or set(failure) != {
            "detected", "stage_id", "global_frame", "local_frame", "phase",
            "fields", "solver_advance_stopped"}:
        return False, None
    stage_id = failure.get("stage_id")
    if (failure.get("detected") is not True or stage_id not in {
            "foundation", "lower", "middle", "top"} or
            isinstance(failure.get("global_frame"), bool) or
            not isinstance(failure.get("global_frame"), int) or
            failure["global_frame"] < 0 or
            isinstance(failure.get("local_frame"), bool) or
            not isinstance(failure.get("local_frame"), int) or
            failure["local_frame"] < 0 or
            failure.get("phase") not in {
                "pre_activation", "post_activation_pre_step",
                "post_coupled_step_pre_trace"} or
            failure.get("solver_advance_stopped") is not True):
        return False, None
    fields = failure.get("fields")
    if not isinstance(fields, list) or not fields:
        return False, None
    paths = []
    for item in fields:
        if not isinstance(item, dict) or set(item) != {
                "path", "value_class", "entity_type", "entity_id"}:
            return False, None
        if (not isinstance(item["path"], str) or not item["path"] or
                item["value_class"] not in {"nan", "positive_infinity",
                                             "negative_infinity"} or
                item["entity_type"] not in {
                    "particle", "rock", "solver_state", "aggregate"} or
                not isinstance(item["entity_id"], (str, int)) or
                isinstance(item["entity_id"], bool)):
            return False, None
        paths.append(item["path"])
    if len(paths) != len(set(paths)):
        return False, None
    return True, stage_id


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "pre_registered_independent_admission_contract"):
        raise ValueError("staged_settling_audit_config_schema_or_status")
    repository = Path(repository_root).resolve()
    expected_bindings = {
        "geometry_config", "scene_manifest", "particle_bank", "geometry_metrics"}
    bindings = config.get("frozen_inputs")
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise ValueError("staged_settling_audit_binding_set")
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("staged_settling_audit_binding_shape:" + name)
        target = repository / binding["path"]
        if not target.is_file() or sha256(target) != binding["sha256"]:
            raise ValueError("staged_settling_audit_binding_mismatch:" + name)
    if config.get("candidate_report_schema") != (
            "excavation_layered_mixed_scene_staged_settling_report_v1"):
        raise ValueError("staged_settling_audit_candidate_schema")
    expected = config.get("expected", {})
    if (expected.get("particle_count") != 72415 or
            expected.get("rock_count") != 22 or
            expected.get("reset_count") != 1 or
            expected.get("stages") != [
                {"id": "foundation", "active_particle_count": 17747,
                 "active_rock_count": 0, "inactive_rock_count": 22},
                {"id": "lower", "active_particle_count": 30977,
                 "active_rock_count": 4, "inactive_rock_count": 18},
                {"id": "middle", "active_particle_count": 50148,
                 "active_rock_count": 10, "inactive_rock_count": 12},
                {"id": "top", "active_particle_count": 72415,
                 "active_rock_count": 22, "inactive_rock_count": 0},
            ]):
        raise ValueError("staged_settling_audit_expected_inventory")
    if config.get("required_evidence") != {
            "implementation_bindings": ["execution_config", "execution_source"],
            "activation_history_components": list(HISTORY_COMPONENTS),
            "stage_timing_and_wall_clock": True,
            "inactive_mpm_transfer_material_contact_zero": True,
            "construction_controls_pre_registered": True,
            "stable_rock_mapping_hash": True,
            "top_mpm_and_rock_outliers": True,
            "nonfinite_failure_evidence": True}:
        raise ValueError("staged_settling_audit_required_evidence_drift")
    exact_thresholds = {
        "stage_max_duration_s": 6.0,
        "stage_min_consecutive_window_s": 0.5,
        "stage_max_rock_linear_speed_m_s": 0.01,
        "stage_max_rock_angular_speed_rad_s": 0.15,
        "stage_max_mpm_speed_p99_m_s": 0.01,
        "stage_max_mpm_speed_max_m_s": 0.15,
        "inactive_max_pose_drift_m": 1e-12,
        "hold_min_duration_s": 1.0,
        "hold_tail_window_s": 0.5,
        "hold_max_rock_displacement_m": 0.005,
        "hold_final_max_rock_linear_speed_m_s": 0.002,
        "hold_final_max_rock_angular_speed_rad_s": 0.03,
        "hold_tail_max_mpm_speed_p99_m_s": 0.01,
        "hold_tail_max_mpm_speed_max_m_s": 0.15,
        "hold_max_closure_error_m": 0.001,
        "hold_min_particle_z_m": -0.002,
    }
    if config.get("thresholds") != exact_thresholds:
        raise ValueError("staged_settling_audit_threshold_drift")
    if config.get("scope") != {
            "independent_report_audit_only": True,
            "physics_execution": False,
            "parameter_tuning": False,
            "fullscale_scene_qualification": False,
            "fullscale_scaleup_authorization_on_pass": True,
            "single_scoop_authorization_on_pass": False,
            "three_cycle_qualification": False}:
        raise ValueError("staged_settling_audit_scope_expanded")
    return config


def _check_bindings(config: dict, report: dict) -> bool:
    actual = report.get("input_bindings")
    return isinstance(actual, dict) and actual == config["frozen_inputs"]


def _sha256_string(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _check_implementation(candidate: dict) -> bool:
    implementation = candidate.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {
            "execution_config", "execution_source",
            "stable_rock_mapping_sha256"}:
        return False
    for name in ("execution_config", "execution_source"):
        binding = implementation[name]
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                or not isinstance(binding["path"], str) or not binding["path"]
                or not _sha256_string(binding["sha256"])):
            return False
    return _sha256_string(implementation["stable_rock_mapping_sha256"])


def _check_construction_controls(candidate: dict) -> bool:
    controls = candidate.get("construction_controls")
    if not isinstance(controls, dict) or set(controls) != {
            "pre_registered", "execution_config_sha256", "gravity_ramp_events",
            "temporary_damping_events", "unregistered_control_changes",
            "nominal_values_restored_before_hold"}:
        return False
    implementation = candidate.get("implementation", {})
    config_binding = implementation.get("execution_config", {})
    if (controls["pre_registered"] is not True or
            controls["execution_config_sha256"] != config_binding.get("sha256") or
            controls["unregistered_control_changes"] is not False or
            controls["nominal_values_restored_before_hold"] is not True):
        return False
    for group in ("gravity_ramp_events", "temporary_damping_events"):
        events = controls[group]
        if not isinstance(events, list):
            return False
        for event in events:
            if not isinstance(event, dict) or set(event) != {
                    "start_step", "end_step", "value"}:
                return False
            if (isinstance(event["start_step"], bool) or
                    not isinstance(event["start_step"], int) or
                    isinstance(event["end_step"], bool) or
                    not isinstance(event["end_step"], int) or
                    event["start_step"] < 0 or
                    event["end_step"] < event["start_step"]):
                return False
            try:
                _finite_number(event["value"], group + ".value")
            except ValueError:
                return False
    return True


def _check_outliers(candidate: dict) -> bool:
    outliers = candidate.get("outliers")
    if not isinstance(outliers, dict) or set(outliers) != {
            "top_mpm_speed_particles", "top_rock_displacements"}:
        return False
    mpm = outliers["top_mpm_speed_particles"]
    rocks = outliers["top_rock_displacements"]
    if not isinstance(mpm, list) or not mpm or not isinstance(rocks, list) or not rocks:
        return False
    for item in mpm:
        if (not isinstance(item, dict) or set(item) != {
                "original_particle_index", "position_world_m", "speed_m_s"}
                or isinstance(item["original_particle_index"], bool)
                or not isinstance(item["original_particle_index"], int)
                or item["original_particle_index"] < 0
                or not isinstance(item["position_world_m"], list)
                or len(item["position_world_m"]) != 3):
            return False
        try:
            [_finite_number(value, "outlier.mpm.position")
             for value in item["position_world_m"]]
            _finite_number(item["speed_m_s"], "outlier.mpm.speed")
        except ValueError:
            return False
    for item in rocks:
        if (not isinstance(item, dict) or set(item) != {
                "rock_id", "max_displacement_m"}
                or not isinstance(item["rock_id"], str) or not item["rock_id"]):
            return False
        try:
            _finite_number(item["max_displacement_m"], "outlier.rock.displacement")
        except ValueError:
            return False
    return True


def _audit_stage(stage: dict, expected: dict, previous_expected: dict | None,
                 limits: dict, nonfinite_failure: dict | None) -> dict:
    if not isinstance(stage, dict):
        raise ValueError("staged_settling_stage_shape")
    failure_bound = (
        isinstance(nonfinite_failure, dict) and
        nonfinite_failure.get("stage_id") == expected["id"])
    pre_activation_failure = (
        failure_bound and nonfinite_failure.get("phase") == "pre_activation")
    inventory_expected = (
        previous_expected if pre_activation_failure and previous_expected is not None
        else expected)
    inventory = (
        stage.get("id") == expected["id"] and
        stage.get("active_particle_count") ==
        inventory_expected["active_particle_count"] and
        stage.get("active_rock_count") ==
        inventory_expected["active_rock_count"] and
        stage.get("inactive_rock_count") ==
        inventory_expected["inactive_rock_count"])
    finite = stage.get("finite") is True
    allow_null = not finite
    nonfinite_failure_bound = failure_bound and not finite
    duration = _finite_number(stage.get("duration_s"), expected["id"] + ".duration")
    window = _finite_number(
        stage.get("consecutive_converged_window_s"), expected["id"] + ".window")
    linear = _nullable_finite_number(
        stage.get("max_rock_linear_speed_m_s"), expected["id"] + ".linear",
        allow_null)
    angular = _nullable_finite_number(
        stage.get("max_rock_angular_speed_rad_s"), expected["id"] + ".angular",
        allow_null)
    p99 = _nullable_finite_number(
        stage.get("mpm_speed_p99_m_s"), expected["id"] + ".p99",
        allow_null)
    maximum = _nullable_finite_number(
        stage.get("mpm_speed_max_m_s"), expected["id"] + ".max",
        allow_null)
    inactive_drift = _nullable_finite_number(
        stage.get("inactive_mujoco_max_pose_drift_m"),
        expected["id"] + ".inactive_drift", allow_null)
    inactive_contacts = (
        stage.get("inactive_mujoco_contact_count") == 0 and
        stage.get("inactive_newton_contact_sample_count") == 0)
    inactive_mpm = all(stage.get(name) == 0 for name in (
        "inactive_mpm_transfer_count", "inactive_mpm_material_count",
        "inactive_mpm_contact_sample_count"))
    start_time = _finite_number(
        stage.get("start_sim_time_s"), expected["id"] + ".start_time")
    end_time = _finite_number(
        stage.get("end_sim_time_s"), expected["id"] + ".end_time")
    step_count = stage.get("step_count")
    wall_clock = _finite_number(
        stage.get("wall_clock_s"), expected["id"] + ".wall_clock")
    zero_step_failure = (
        nonfinite_failure_bound and nonfinite_failure.get("phase") in {
            "pre_activation", "post_activation_pre_step"})
    timing = (
        start_time >= 0.0 and end_time >= start_time and
        abs((end_time - start_time) - duration) <= 1e-9 and
        isinstance(step_count, int) and not isinstance(step_count, bool) and
        ((step_count == 0 and zero_step_failure) or step_count > 0) and
        wall_clock >= 0.0)
    histories = stage.get("activation_history_hashes")
    history_hashes = (
        isinstance(histories, dict) and set(histories) == set(HISTORY_COMPONENTS))
    if history_hashes:
        for component in HISTORY_COMPONENTS:
            value = histories[component]
            if (not isinstance(value, dict) or set(value) != {"before", "after"}
                    or not _sha256_string(value["before"])
                    or value["before"] != value["after"]):
                history_hashes = False
                break
    stage_invariants = all(stage.get(name) is True for name in (
        "activation_history_preserved", "mass_identity_closed",
        "particle_indices_preserved", "rock_ids_preserved"))
    nullable_metrics = (linear, angular, p99, maximum, inactive_drift)
    metrics_within_gate = (
        all(value is not None for value in nullable_metrics) and
        duration <= limits["stage_max_duration_s"] and
        window >= limits["stage_min_consecutive_window_s"] and
        linear <= limits["stage_max_rock_linear_speed_m_s"] and
        angular <= limits["stage_max_rock_angular_speed_rad_s"] and
        p99 <= limits["stage_max_mpm_speed_p99_m_s"] and
        maximum <= limits["stage_max_mpm_speed_max_m_s"] and
        inactive_drift <= limits["inactive_max_pose_drift_m"])
    evidence_complete = (
        inactive_mpm and timing and history_hashes and stage_invariants)
    derived = (inventory and finite and inactive_contacts and
               metrics_within_gate and evidence_complete)
    declared_consistent = stage.get("converged") is derived
    return {
        "id": expected["id"],
        "inventory_exact": inventory,
        "finite": finite,
        "nonfinite_failure_bound": nonfinite_failure_bound,
        "inactive_contacts_zero": inactive_contacts,
        "inactive_mpm_zero": inactive_mpm,
        "timing_consistent": timing,
        "activation_history_preserved": history_hashes and stage_invariants,
        "evidence_complete": evidence_complete,
        "metrics_within_gate": metrics_within_gate,
        "derived_qualified": derived,
        "declared_consistent": declared_consistent,
    }


def _audit_hold(hold: dict, limits: dict) -> dict:
    if not isinstance(hold, dict):
        raise ValueError("staged_settling_hold_shape")
    metrics = (
        _finite_number(hold.get("duration_s"), "hold.duration") >=
        limits["hold_min_duration_s"] and
        _finite_number(hold.get("tail_window_s"), "hold.tail_window") >=
        limits["hold_tail_window_s"] and
        _finite_number(hold.get("max_rock_displacement_m"), "hold.displacement") <=
        limits["hold_max_rock_displacement_m"] and
        _finite_number(hold.get("final_max_rock_linear_speed_m_s"), "hold.linear") <=
        limits["hold_final_max_rock_linear_speed_m_s"] and
        _finite_number(hold.get("final_max_rock_angular_speed_rad_s"), "hold.angular") <=
        limits["hold_final_max_rock_angular_speed_rad_s"] and
        _finite_number(hold.get("tail_mpm_speed_p99_m_s"), "hold.p99") <=
        limits["hold_tail_max_mpm_speed_p99_m_s"] and
        _finite_number(hold.get("tail_mpm_speed_max_m_s"), "hold.max") <=
        limits["hold_tail_max_mpm_speed_max_m_s"] and
        _finite_number(hold.get("max_closure_error_m"), "hold.closure") <=
        limits["hold_max_closure_error_m"] and
        _finite_number(hold.get("min_particle_z_m"), "hold.min_z") >=
        limits["hold_min_particle_z_m"])
    invariants = all(hold.get(name) is True for name in (
        "finite", "nominal_parameters_restored", "construction_velocities_zeroed",
        "positions_unchanged_by_velocity_zeroing", "mass_identity_closed",
        "particle_indices_preserved", "rock_ids_preserved"))
    contacts = hold.get("forbidden_contact_count") == 0
    no_walls = hold.get("construction_walls_present") is False
    derived = metrics and invariants and contacts and no_walls
    return {
        "metrics_within_gate": metrics,
        "invariants_hold": invariants,
        "forbidden_contacts_zero": contacts,
        "construction_walls_absent": no_walls,
        "derived_qualified": derived,
        "declared_consistent": hold.get("qualified") is derived,
    }


def audit_candidate(config: dict, candidate: dict) -> dict:
    if candidate.get("schema") != config["candidate_report_schema"]:
        raise ValueError("staged_settling_candidate_schema")
    stages = candidate.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("staged_settling_stage_count")
    nonfinite_shape_valid, nonfinite_stage_id = _check_nonfinite_failure(
        candidate)
    identity = candidate.get("identity", {})
    identity_exact = (
        identity.get("same_solver_object") is True and
        identity.get("same_state_object") is True and
        identity.get("reset_count") == config["expected"]["reset_count"] and
        identity.get("particle_count") == config["expected"]["particle_count"] and
        identity.get("rock_count") == config["expected"]["rock_count"] and
        identity.get("particle_indices_preserved") is True and
        identity.get("rock_ids_preserved") is True)
    expected_stages = config["expected"]["stages"]
    stage_audits = [
        _audit_stage(
            stage, expected,
            expected_stages[index - 1] if index > 0 else None,
            config["thresholds"],
            candidate.get("nonfinite_failure") if nonfinite_shape_valid else None)
        for index, (stage, expected) in enumerate(zip(stages, expected_stages))]
    nonfinite_stage_ids = [
        expected["id"] for stage, expected in
        zip(stages, config["expected"]["stages"])
        if stage.get("finite") is False]
    nonfinite_evidence_consistent = (
        nonfinite_shape_valid and
        ((nonfinite_stage_id is None and not nonfinite_stage_ids) or
         (nonfinite_stage_id is not None and
          nonfinite_stage_ids == [nonfinite_stage_id])))
    staged_qualified = identity_exact and all(
        item["derived_qualified"] for item in stage_audits)
    hold_audit = _audit_hold(candidate.get("nominal_hold"), config["thresholds"])
    hold_qualified = hold_audit["derived_qualified"]
    declared = candidate.get("qualifications", {})
    declarations_consistent = (
        declared.get("staged_construction_qualified") is staged_qualified and
        declared.get("nominal_hold_qualified") is hold_qualified and
        declared.get("fullscale_scene_qualified") is False and
        declared.get("three_cycle_qualified") is False and
        all(item["declared_consistent"] for item in stage_audits) and
        hold_audit["declared_consistent"])
    expected_status = (
        "staged_settling_passed" if staged_qualified and hold_qualified
        else "staged_settling_failed")
    status_consistent = candidate.get("status") == expected_status
    checks = {
        "input_bindings_exact": _check_bindings(config, candidate),
        "implementation_bindings_complete": _check_implementation(candidate),
        "construction_controls_complete": _check_construction_controls(candidate),
        "outlier_evidence_complete": _check_outliers(candidate),
        "nonfinite_failure_evidence_consistent": nonfinite_evidence_consistent,
        "identity_and_inventory_exact": identity_exact,
        "all_stage_declarations_consistent": all(
            item["declared_consistent"] for item in stage_audits),
        "hold_declaration_consistent": hold_audit["declared_consistent"],
        "qualification_declarations_consistent": declarations_consistent,
        "status_consistent": status_consistent,
    }
    report_consistent = all(checks.values())
    admitted_for_scaleup = (
        report_consistent and staged_qualified and hold_qualified)
    return {
        "schema": REPORT_SCHEMA,
        "status": "admitted_for_fullscale_scaleup" if admitted_for_scaleup else
                  ("honest_failure_preserved" if report_consistent else
                   "candidate_report_rejected"),
        "checks": checks,
        "stage_audits": stage_audits,
        "hold_audit": hold_audit,
        "derived_qualifications": {
            "staged_construction_qualified": staged_qualified,
            "nominal_hold_qualified": hold_qualified,
            "fullscale_scaleup_authorized": admitted_for_scaleup,
            "p1a_single_scoop_authorized": False,
            "fullscale_scene_qualified": False,
            "three_cycle_qualified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.repository_root)
    candidate = json.loads(
        Path(args.candidate_report).read_text(encoding="utf-8"))
    report = audit_candidate(config, candidate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] != "candidate_report_rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
