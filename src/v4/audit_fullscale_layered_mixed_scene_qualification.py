#!/usr/bin/env python3
"""Independently derive the full-scale scene gate before P1a."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONFIG_SCHEMA = (
    "excavation_fullscale_layered_mixed_scene_qualification_audit_config_v1")
REPORT_SCHEMA = (
    "excavation_fullscale_layered_mixed_scene_qualification_audit_report_v1")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fullscale_audit_json_object_required")
    return value


def _binding(path: str | Path) -> dict:
    target = Path(path).resolve()
    if not target.is_file():
        raise ValueError("fullscale_audit_missing_artifact:" + str(target))
    return {"path": str(target), "sha256": sha256(target)}


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    config = _load_json(path)
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "pre_registered_independent_p1a_admission_contract"):
        raise ValueError("fullscale_audit_config_schema_or_status")
    expected_bindings = {
        "formal_scene_model_adapter_config",
        "formal_scene_geometry_preparation_config",
        "scaled_settling_audit_config",
    }
    bindings = config.get("frozen_inputs")
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise ValueError("fullscale_audit_frozen_input_set")
    repository = Path(repository_root).resolve()
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("fullscale_audit_binding_shape:" + name)
        target = repository / binding["path"]
        if not target.is_file() or sha256(target) != binding["sha256"]:
            raise ValueError("fullscale_audit_binding_mismatch:" + name)
    if config.get("candidate_report_schema") != (
            "excavation_fullscale_layered_mixed_scene_qualification_report_v1"):
        raise ValueError("fullscale_audit_candidate_schema_drift")
    contracts = config.get("artifact_contracts")
    if not isinstance(contracts, dict) or set(contracts) != {
            "scene_manifest", "geometry_bundle_report", "time_zero_report",
            "settling_report"}:
        raise ValueError("fullscale_audit_artifact_contract_set")
    expected_checks = {
        "stable_scene_manifest_valid", "geometry_bundle_cross_bound",
        "actual_n_time_zero_passed", "fullscale_staged_settling_passed",
        "nominal_hold_passed", "visual_review_passed",
        "mpm_connectivity_passed", "identity_and_mass_closed",
    }
    if set(config.get("required_candidate_checks", [])) != expected_checks:
        raise ValueError("fullscale_audit_candidate_check_drift")
    for name, contract in contracts.items():
        checks = contract.get("required_true_checks")
        if not isinstance(checks, list) or not checks or len(checks) != len(set(checks)):
            raise ValueError("fullscale_audit_artifact_check_contract:" + name)
    if config.get("scope") != {
            "independent_report_audit_only": True,
            "physics_execution": False,
            "gpu_run": False,
            "parameter_tuning": False,
            "may_authorize_p1a_on_pass": True,
            "three_cycle_qualification": False,
            "training": False}:
        raise ValueError("fullscale_audit_scope_expanded")
    return config


def _typed_checks(value: object, required: list[str]) -> bool:
    return (isinstance(value, dict) and set(value) == set(required) and
            all(isinstance(item, bool) for item in value.values()))


def _binding_set(candidate: dict) -> tuple[dict, dict, bool]:
    names = {
        "scaled_audit", "scene_manifest", "geometry_bundle_report",
        "time_zero_report", "settling_report",
    }
    inputs = candidate.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != names:
        raise ValueError("fullscale_audit_candidate_input_set")
    artifacts = {}
    exact = True
    for name, declared in inputs.items():
        if not isinstance(declared, dict) or set(declared) != {"path", "sha256"}:
            raise ValueError("fullscale_audit_candidate_binding_shape:" + name)
        actual = _binding(declared["path"])
        exact = exact and declared == actual
        artifacts[name] = _load_json(declared["path"])
    return inputs, artifacts, exact


def _scaled_admitted(value: dict) -> bool:
    derived = value.get("derived_qualifications", {})
    return (
        value.get("schema") ==
        "excavation_layered_mixed_scene_staged_settling_audit_report_v1" and
        value.get("status") == "admitted_for_fullscale_scaleup" and
        derived.get("staged_construction_qualified") is True and
        derived.get("nominal_hold_qualified") is True and
        derived.get("fullscale_scaleup_authorized") is True and
        derived.get("p1a_single_scoop_authorized") is False and
        derived.get("fullscale_scene_qualified") is False and
        derived.get("three_cycle_qualified") is False)


def _scene_audit(contract: dict, value: dict) -> tuple[bool, bool]:
    checks = value.get("checks")
    well_formed = (
        value.get("schema") == contract.get("schema") and
        value.get("status") == contract.get("status") and
        _typed_checks(checks, contract["required_true_checks"]) and
        isinstance(value.get("rock_count"), int) and value["rock_count"] > 0 and
        isinstance(value.get("rocks"), list) and
        len(value["rocks"]) == value["rock_count"] and
        all(isinstance(rock, dict) and isinstance(rock.get("rock_id"), str)
            for rock in value["rocks"]) and
        len({rock["rock_id"] for rock in value["rocks"]}) == value["rock_count"])
    return well_formed, bool(well_formed and all(checks.values()))


def _bundle_audit(contract: dict, value: dict, scene: dict,
                  scene_binding: dict) -> tuple[bool, bool]:
    checks = value.get("checks")
    well_formed = (
        value.get("schema") == contract.get("schema") and
        value.get("status") == contract.get("status") and
        _typed_checks(checks, contract["required_true_checks"]) and
        isinstance(value.get("rock_count"), int) and
        isinstance(value.get("expected_nq"), int) and
        isinstance(value.get("expected_nv"), int) and
        value.get("time_zero_executed") is False and
        value.get("physics_step_executed") is False and
        value.get("formal_scene_qualified") is False and
        value.get("physics_qualified") is False)
    derived = bool(
        well_formed and all(checks.values()) and
        value["scene_manifest_sha256"] == scene_binding["sha256"] and
        value["rock_count"] == scene["rock_count"])
    return well_formed, derived


def _execution_audit(contract: dict, value: dict, expected_inputs: dict,
                     expected_input_names: set[str], bundle: dict
                     ) -> tuple[bool, bool]:
    checks = value.get("checks")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        return False, False
    bindings_exact = (
        set(inputs) == expected_input_names and
        all(inputs.get(name) == expected_inputs[name]
            for name in expected_input_names))
    all_checks = bool(
        _typed_checks(checks, contract["required_true_checks"]) and
        all(checks.values()))
    expected_status = (contract["passing_status"] if all_checks else
                       contract["failure_status"])
    well_formed = (
        value.get("schema") == contract.get("schema") and
        _typed_checks(checks, contract["required_true_checks"]) and
        value.get("status") == expected_status and bindings_exact and
        value.get("rock_count") == bundle.get("rock_count") and
        value.get("expected_nq") == bundle.get("expected_nq") and
        value.get("expected_nv") == bundle.get("expected_nv"))
    return bool(well_formed), bool(well_formed and all_checks)


def audit_candidate(config: dict, candidate: dict,
                    candidate_report_binding: dict) -> dict:
    if (not isinstance(candidate_report_binding, dict) or
            set(candidate_report_binding) != {"path", "sha256"} or
            _binding(candidate_report_binding["path"]) !=
            candidate_report_binding or
            _load_json(candidate_report_binding["path"]) != candidate):
        raise ValueError("fullscale_audit_candidate_binding")
    if candidate.get("schema") != config["candidate_report_schema"]:
        raise ValueError("fullscale_audit_candidate_schema")
    inputs, artifacts, bindings_exact = _binding_set(candidate)
    contracts = config["artifact_contracts"]
    scaled_ok = _scaled_admitted(artifacts["scaled_audit"])
    scene_well_formed, scene_valid = _scene_audit(
        contracts["scene_manifest"], artifacts["scene_manifest"])
    bundle_well_formed, bundle_valid = _bundle_audit(
        contracts["geometry_bundle_report"],
        artifacts["geometry_bundle_report"], artifacts["scene_manifest"],
        inputs["scene_manifest"])
    time_zero_well_formed, time_zero_passed = _execution_audit(
        contracts["time_zero_report"], artifacts["time_zero_report"], inputs,
        {"scene_manifest", "geometry_bundle_report"},
        artifacts["geometry_bundle_report"])
    settling_well_formed, _settling_passed = _execution_audit(
        contracts["settling_report"], artifacts["settling_report"], inputs,
        {"scene_manifest", "geometry_bundle_report", "time_zero_report"},
        artifacts["geometry_bundle_report"])
    settling_checks = artifacts["settling_report"].get("checks", {})
    derived_checks = {
        "stable_scene_manifest_valid": scene_valid,
        "geometry_bundle_cross_bound": bundle_valid,
        "actual_n_time_zero_passed": time_zero_passed,
        "fullscale_staged_settling_passed": bool(
            settling_well_formed and
            settling_checks.get("same_solver_and_state") and
            settling_checks.get("actual_n_bound") and
            settling_checks.get("fullscale_staged_settling_passed")),
        "nominal_hold_passed": bool(
            settling_well_formed and
            settling_checks.get("nominal_hold_passed")),
        "visual_review_passed": bool(
            settling_well_formed and
            settling_checks.get("visual_review_passed")),
        "mpm_connectivity_passed": bool(
            settling_well_formed and
            settling_checks.get("mpm_connectivity_passed")),
        "identity_and_mass_closed": bool(
            settling_well_formed and
            settling_checks.get("identity_and_mass_closed")),
    }
    required_checks = set(config["required_candidate_checks"])
    declared_checks = candidate.get("checks")
    declaration_shape = (
        isinstance(declared_checks, dict) and
        set(declared_checks) == required_checks and
        all(isinstance(value, bool) for value in declared_checks.values()))
    derived_pass = all(derived_checks.values())
    expected_status = (
        "fullscale_scene_qualified_for_audit" if derived_pass else
        "fullscale_scene_failed")
    qualifications = candidate.get("qualifications", {})
    expected_qualifications = {
        "fullscale_scene_qualified": derived_pass,
        "p1a_single_scoop_authorized": derived_pass,
        "three_cycle_qualified": False,
    }
    declaration_consistent = (
        declaration_shape and declared_checks == derived_checks and
        candidate.get("status") == expected_status and
        qualifications == expected_qualifications)
    audit_checks = {
        "input_bindings_exact": bindings_exact,
        "scaled_scaleup_gate_admitted": scaled_ok,
        "scene_manifest_well_formed": scene_well_formed,
        "geometry_bundle_well_formed": bundle_well_formed,
        "time_zero_report_well_formed": time_zero_well_formed,
        "settling_report_well_formed": settling_well_formed,
        "candidate_declaration_consistent": declaration_consistent,
    }
    report_consistent = all(audit_checks.values())
    admitted = report_consistent and derived_pass
    return {
        "schema": REPORT_SCHEMA,
        "status": ("fullscale_scene_admitted_for_p1a" if admitted else
                   ("fullscale_scene_failure_preserved" if report_consistent else
                    "candidate_report_rejected")),
        "candidate_report": candidate_report_binding,
        "inputs": inputs,
        "checks": audit_checks,
        "derived_checks": derived_checks,
        "derived_qualifications": {
            "scaled_method_qualified": scaled_ok,
            "fullscale_scene_qualified": admitted,
            "p1a_single_scoop_authorized": admitted,
            "three_cycle_qualified": False,
        },
        "execution": {
            "physics_executed_by_auditor": False,
            "gpu_run_by_auditor": False,
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
    candidate = _load_json(args.candidate_report)
    report = audit_candidate(config, candidate, _binding(args.candidate_report))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 1 if report["status"] == "candidate_report_rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
