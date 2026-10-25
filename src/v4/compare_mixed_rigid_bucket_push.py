"""Compare contact and no-contact bucket-push runs with identical commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-report", required=True)
    parser.add_argument("--clear-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-report", required=True)
    args = parser.parse_args()

    contact = json.loads(Path(args.contact_report).read_text(encoding="utf-8"))
    clear = json.loads(Path(args.clear_report).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    parent_path = Path(args.parent_report)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    acceptance = config["mechanism_candidate"]["unit_scenarios"][
        "bucket_push"]["acceptance"]
    contact_run, clear_run = contact["run"], clear["run"]
    qpos_difference = [
        first - second for first, second in zip(
            contact_run["final_mechanism_qpos"],
            clear_run["final_mechanism_qpos"])]
    qvel_difference = [
        first - second for first, second in zip(
            contact_run["final_mechanism_qvel"],
            clear_run["final_mechanism_qvel"])]
    maximum_qpos_difference = max(abs(value) for value in qpos_difference)
    maximum_qvel_difference = max(abs(value) for value in qvel_difference)
    checks = {
        "both_component_reports_passed": (
            contact["status"] == "passed" and clear["status"] == "passed"),
        "upstream_rigid_units_passed": all(
            parent["admission"].get(key) for key in (
                "standalone_mujoco_probe_passed",
                "rigid_bridge_freefall_passed",
                "rigid_bridge_slope_roll_passed",
                "rigid_bridge_rock_collision_passed",
                "rigid_bridge_bucket_push_clear_passed",
                "rigid_bridge_bucket_push_passed")),
        "command_and_timing_identical": (
            contact_run["command_schedule"] == clear_run["command_schedule"] and
            contact_run["steps"] == clear_run["steps"] and
            contact_run["dt_s"] == clear_run["dt_s"] and
            contact_run["substeps"] == clear_run["substeps"]),
        "initial_mechanism_state_identical": (
            contact_run["initial_mechanism_qpos"] ==
            clear_run["initial_mechanism_qpos"]),
        "contact_only_in_push_case": (
            contact_run["rock_bucket_contact_substeps"] > 0 and
            clear_run["rock_bucket_contact_substeps"] == 0),
        "reaction_impulse_only_in_push_case": (
            contact_run["total_bucket_rock_impulse_world_ns"][1] < 0 and
            max(abs(value) for value in
                clear_run["total_bucket_rock_impulse_world_ns"]) < 1e-12),
        "contact_changes_mechanism_trajectory": (
            maximum_qpos_difference >= acceptance[
                "minimum_contact_vs_clear_mechanism_qpos_difference"]),
        "contact_perturbation_remains_bounded": (
            maximum_qpos_difference <= acceptance[
                "maximum_contact_vs_clear_mechanism_qpos_difference"]),
        "all_differences_finite": all(math.isfinite(value) for value in (
            qpos_difference + qvel_difference)),
        "same_frozen_source_model": (
            contact["source_model_sha256_after"] ==
            clear["source_model_sha256_after"] ==
            contact["source_model_sha256_before"] ==
            clear["source_model_sha256_before"]),
    }
    report = {
        "schema": "excavation_mixed_rigid_bucket_push_comparison_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "contact_report_sha256": sha256(args.contact_report),
        "clear_report_sha256": sha256(args.clear_report),
        "config_sha256": sha256(args.config),
        "maximum_final_mechanism_qpos_difference": maximum_qpos_difference,
        "maximum_final_mechanism_qvel_difference": maximum_qvel_difference,
        "final_mechanism_qpos_difference": qpos_difference,
        "bucket_reaction_impulse_world_ns": (
            contact_run["total_bucket_rock_impulse_world_ns"]),
        "scope_limit": (
            "qualifies reaction sign, torque accounting and a bounded mechanism "
            "trajectory difference for one shelf-supported side push only"
        ),
    }
    write_json(args.output, report)
    parent["admission"]["rigid_bridge_bucket_push_comparison_passed"] = (
        report["status"] == "passed")
    parent["mechanism_bucket_push_comparison_report_sha256"] = sha256(args.output)
    parent["status"] = (
        "r1_full_rock_single_chain_passed_r2_pending"
        if (report["status"] == "passed" and
            parent["admission"].get("rigid_bridge_mechanism_dump_passed")) else
        "mechanism_hold_passed_dump_pending"
        if (report["status"] == "passed" and
            parent["admission"].get("rigid_bridge_mechanism_hold_passed")) else
        "mechanism_entry_passed_hold_dump_pending"
        if (report["status"] == "passed" and
            parent["admission"].get("rigid_bridge_mechanism_entry_passed")) else
        "standalone_probe_rigid_units_through_bucket_push_passed_"
        "entry_hold_dump_pending"
        if report["status"] == "passed" else
        "bucket_push_comparison_failed")
    write_json(parent_path, parent)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
