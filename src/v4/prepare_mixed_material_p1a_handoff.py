#!/usr/bin/env python3
"""Build an offline P1a handoff receipt without advancing physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONFIG_SCHEMA = "excavation_mixed_material_p1a_handoff_config_v1"
REPORT_SCHEMA = "excavation_mixed_material_p1a_handoff_report_v1"
GEOMETRY_BUNDLE_SCHEMA = "excavation_formal_scene_geometry_bundle_report_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: str | Path) -> dict:
    target = Path(path).resolve()
    if not target.is_file():
        raise ValueError("p1a_handoff_missing_artifact:" + str(target))
    return {"path": str(target), "sha256": sha256(target)}


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("p1a_handoff_json_object_required")
    return value


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    config = _load_json(path)
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "pre_registered_offline_handoff_only"):
        raise ValueError("p1a_handoff_config_schema_or_status")
    expected_names = {
        "scaled_settling_audit_config", "formal_geometry_preparation_config",
        "fullscale_qualification_audit_config",
        "three_cycle_config", "state_adapter", "terrain_observation_adapter",
        "three_cycle_contract", "existing_cycle_selector",
        "existing_cycle_geometry",
    }
    bindings = config.get("frozen_inputs")
    if not isinstance(bindings, dict) or set(bindings) != expected_names:
        raise ValueError("p1a_handoff_frozen_input_set")
    repository = Path(repository_root).resolve()
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("p1a_handoff_binding_shape:" + name)
        target = repository / binding["path"]
        if not target.is_file() or sha256(target) != binding["sha256"]:
            raise ValueError("p1a_handoff_binding_mismatch:" + name)
    if config.get("required_scaled_audit") != {
            "schema": "excavation_layered_mixed_scene_staged_settling_audit_report_v1",
            "status": "admitted_for_fullscale_scaleup"}:
        raise ValueError("p1a_handoff_scaled_gate_drift")
    required = config.get("required_fullscale_qualification", {})
    if (required.get("schema") !=
            "excavation_fullscale_layered_mixed_scene_qualification_audit_report_v1"
            or required.get("passing_status") !=
            "fullscale_scene_admitted_for_p1a"
            or required.get("failure_status") !=
            "fullscale_scene_failure_preserved"
            or required.get("required_true_audit_checks") != [
                "input_bindings_exact",
                "scaled_scaleup_gate_admitted",
                "scene_manifest_well_formed",
                "geometry_bundle_well_formed",
                "time_zero_report_well_formed",
                "settling_report_well_formed",
                "candidate_declaration_consistent"]
            or required.get("required_true_checks") != [
                "stable_scene_manifest_valid",
                "geometry_bundle_cross_bound",
                "actual_n_time_zero_passed",
                "fullscale_staged_settling_passed",
                "nominal_hold_passed",
                "visual_review_passed",
                "mpm_connectivity_passed",
                "identity_and_mass_closed"]):
        raise ValueError("p1a_handoff_fullscale_gate_drift")
    if config.get("adapter_chain") != [
            "fullscale_scene_manifest", "formal_geometry_bundle",
            "actual_n_time_zero_and_settling", "mixed_three_cycle_state",
            "mixed_terrain_observation", "existing_cycle_selector",
            "dispatch_geometry_revalidation",
            "existing_same_instance_three_cycle_loop"]:
        raise ValueError("p1a_handoff_adapter_chain_drift")
    if config.get("scope") != {
            "offline_contract_only": True,
            "physics_execution": False,
            "gpu_run": False,
            "scaled_pass_only_authorizes_fullscale_scaleup": True,
            "fullscale_qualification_required_for_p1a": True,
            "may_authorize_p1a_after_fullscale_qualification": True,
            "three_cycle_qualification": False,
            "training": False}:
        raise ValueError("p1a_handoff_scope_expanded")
    return config


def _validate_scaled_audit(config: dict, audit: dict) -> None:
    required = config["required_scaled_audit"]
    derived = audit.get("derived_qualifications", {})
    if (audit.get("schema") != required["schema"] or
            audit.get("status") != required["status"] or
            derived.get("staged_construction_qualified") is not True or
            derived.get("nominal_hold_qualified") is not True or
            derived.get("fullscale_scaleup_authorized") is not True or
            derived.get("p1a_single_scoop_authorized") is not False or
            derived.get("fullscale_scene_qualified") is not False or
            derived.get("three_cycle_qualified") is not False):
        raise ValueError("p1a_handoff_scaled_audit_not_admitted")


def _validate_geometry_bundle(bundle: dict, scene_sha256: str) -> None:
    if (bundle.get("schema") != GEOMETRY_BUNDLE_SCHEMA or
            bundle.get("status") != "formal_scene_geometry_bundle_candidate" or
            bundle.get("scene_manifest_sha256") != scene_sha256 or
            bundle.get("formal_scene_qualified") is not False or
            bundle.get("physics_qualified") is not False or
            bundle.get("time_zero_executed") is not False or
            bundle.get("physics_step_executed") is not False or
            not isinstance(bundle.get("checks"), dict) or
            not all(bundle["checks"].values())):
        raise ValueError("p1a_handoff_geometry_bundle_invalid")


def build_receipt(config: dict, scaled_audit_path: str | Path,
                  fullscale_audit_report_path: str | Path,
                  geometry_bundle_path: str | Path) -> dict:
    scaled_binding = _binding(scaled_audit_path)
    fullscale_binding = _binding(fullscale_audit_report_path)
    bundle_binding = _binding(geometry_bundle_path)
    scaled = _load_json(scaled_audit_path)
    fullscale = _load_json(fullscale_audit_report_path)
    bundle = _load_json(geometry_bundle_path)
    _validate_scaled_audit(config, scaled)

    required = config["required_fullscale_qualification"]
    if fullscale.get("schema") != required["schema"]:
        raise ValueError("p1a_handoff_fullscale_schema")
    candidate_binding = fullscale.get("candidate_report")
    if (not isinstance(candidate_binding, dict) or
            set(candidate_binding) != {"path", "sha256"} or
            _binding(candidate_binding["path"]) != candidate_binding):
        raise ValueError("p1a_handoff_fullscale_candidate_binding")
    inputs = fullscale.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "scaled_audit", "scene_manifest", "geometry_bundle_report",
            "time_zero_report", "settling_report"}:
        raise ValueError("p1a_handoff_fullscale_input_set")
    if (inputs["scaled_audit"] != scaled_binding or
            inputs["geometry_bundle_report"] != bundle_binding):
        raise ValueError("p1a_handoff_cross_binding_mismatch")
    for name, binding in inputs.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("p1a_handoff_fullscale_binding_shape:" + name)
        if _binding(binding["path"]) != binding:
            raise ValueError("p1a_handoff_fullscale_binding_mismatch:" + name)
    _validate_geometry_bundle(bundle, inputs["scene_manifest"]["sha256"])

    audit_checks = fullscale.get("checks")
    if (not isinstance(audit_checks, dict) or
            set(audit_checks) != set(required["required_true_audit_checks"]) or
            not all(value is True for value in audit_checks.values())):
        raise ValueError("p1a_handoff_fullscale_audit_checks")
    if fullscale.get("execution") != {
            "physics_executed_by_auditor": False,
            "gpu_run_by_auditor": False}:
        raise ValueError("p1a_handoff_fullscale_audit_execution")
    checks = fullscale.get("derived_checks")
    expected_checks = set(required["required_true_checks"])
    if not isinstance(checks, dict) or set(checks) != expected_checks:
        raise ValueError("p1a_handoff_fullscale_check_set")
    all_checks = all(checks.values())
    qualifications = fullscale.get("derived_qualifications", {})
    declared_pass = (
        qualifications.get("scaled_method_qualified") is True and
        qualifications.get("fullscale_scene_qualified") is True and
        qualifications.get("p1a_single_scoop_authorized") is True and
        qualifications.get("three_cycle_qualified") is False)
    honest_failure = (
        qualifications.get("scaled_method_qualified") is True and
        qualifications.get("fullscale_scene_qualified") is False and
        qualifications.get("p1a_single_scoop_authorized") is False and
        qualifications.get("three_cycle_qualified") is False)
    if all_checks:
        consistent = (fullscale.get("status") == required["passing_status"] and
                      declared_pass)
    else:
        consistent = (fullscale.get("status") == required["failure_status"] and
                      honest_failure)
    if not consistent:
        raise ValueError("p1a_handoff_fullscale_declaration_inconsistent")

    ready = all_checks and declared_pass
    return {
        "schema": REPORT_SCHEMA,
        "status": ("p1a_offline_handoff_ready" if ready else
                   "fullscale_failure_preserved"),
        "inputs": {
            "scaled_audit": scaled_binding,
            "fullscale_qualification": fullscale_binding,
            "fullscale_candidate_report": candidate_binding,
            "scene_manifest": inputs["scene_manifest"],
            "geometry_bundle_report": bundle_binding,
            "time_zero_report": inputs["time_zero_report"],
            "settling_report": inputs["settling_report"],
        },
        "adapter_bindings": {
            name: config["frozen_inputs"][name]
            for name in (
                "state_adapter", "terrain_observation_adapter",
                "three_cycle_contract", "existing_cycle_selector",
                "existing_cycle_geometry")
        },
        "adapter_chain": config["adapter_chain"],
        "qualifications": {
            "scaled_method_qualified": True,
            "fullscale_scene_qualified": ready,
            "p1a_single_scoop_authorized": ready,
            "three_cycle_qualified": False,
        },
        "execution": {
            "physics_executed": False,
            "gpu_run": False,
            "same_instance_three_cycle_required_after_p1a": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scaled-audit", required=True)
    parser.add_argument("--fullscale-audit-report", required=True)
    parser.add_argument("--geometry-bundle-report", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, args.repository_root)
    receipt = build_receipt(
        config, args.scaled_audit, args.fullscale_audit_report,
        args.geometry_bundle_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
