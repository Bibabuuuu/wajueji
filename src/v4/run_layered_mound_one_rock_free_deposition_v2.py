#!/usr/bin/env python3
"""Correct v1 so the foundation truly settles without a nearby rock collider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_layered_mound_one_rock_free_deposition as base
from build_mixed_material_capacity_preflight import sha256
from run_layered_mound_single_rock_press_smoke import _resolve
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_one_rock_free_deposition_v2_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_one_rock_free_deposition_v2_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "soil_only_presettle_then_one_rock_dynamic_insertion"):
        raise ValueError("free_deposition_v2_config_schema_or_status")
    binding = value.get("base_candidate", {})
    for field in ("config", "runner", "report"):
        target = _resolve(repository, binding.get(field + "_path", ""))
        if (not target.is_file() or sha256(target) !=
                binding.get(field + "_sha256")):
            raise ValueError(f"free_deposition_v2_base_{field}_binding")
    base_config, _ = base.load_config(
        _resolve(repository, binding["config_path"]), repository)
    report = json.loads(_resolve(
        repository, binding["report_path"]).read_text(encoding="utf-8"))
    if (report.get("status") !=
            "one_rock_free_deposition_on_active_soil_stable_failed"
            or report.get("checks", {}).get(
                "no_rock_soil_contact_during_foundation_presettle") is not False
            or int(report.get("foundation_presettle", {}).get(
                "contact_samples_with_clearance_rock", 0)) <= 0):
        raise ValueError("free_deposition_v2_base_failure_contract")
    if value.get("correction") != {
            "reason":
            "v1_clearance_rock_contacted_soil_during_foundation_presettle",
            "only_change":
            "park_mpm_rock_collider_during_soil_only_presettle",
            "presettle_parking_position_world_m": [0.0, 0.0, 5.0],
            "parked_assign_calls_including_fixture_initialization": 61,
            "foundation_presettle_frames": 60,
            "required_foundation_rock_contact_samples": 0,
            "dynamic_insertion_surface_clearance_m": 0.005,
            "collider_velocity_mode_preserved": "forward",
            "material_changed": False,
            "stability_thresholds_changed": False,
            "dynamic_deposition_schedule_changed": False}:
        raise ValueError("free_deposition_v2_correction_contract")
    if value.get("scope") != {
            "active_soil": "unexcavated_foundation_stage_only",
            "active_rock_count": 1, "other_64_rocks_active": False,
            "soil_only_presettle": True, "kinematic_press": False,
            "construction_support_force": False, "gravity_ramp": False,
            "cover_soil_active": False, "full_lower_layer": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("free_deposition_v2_scope_contract")
    return value, base_config


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("free_deposition_v2_output_exists")
    config, _ = load_config(config_path, repository)
    correction = config["correction"]
    parking = np.asarray(
        correction["presettle_parking_position_world_m"], dtype=np.float64)
    parked_calls = int(
        correction["parked_assign_calls_including_fixture_initialization"])
    OriginalFixture = base.source.MultiRockFixture

    class SoilOnlyPresettleFixture(OriginalFixture):
        def __init__(self, *fixture_args, **fixture_kwargs):
            self._v2_assign_calls = 0
            self._v2_parked_assignments = 0
            super().__init__(*fixture_args, **fixture_kwargs)

        def assign(self, positions, velocities, angular=None, quaternions=None):
            actual_positions = positions
            if self._v2_assign_calls < parked_calls:
                actual_positions = [parking.copy() for _ in positions]
                self._v2_parked_assignments += 1
            self._v2_assign_calls += 1
            return super().assign(
                actual_positions, velocities, angular=angular,
                quaternions=quaternions)

    original_fixture = base.source.MultiRockFixture
    base.source.MultiRockFixture = SoilOnlyPresettleFixture
    try:
        base_args = argparse.Namespace(**vars(args))
        base_args.config = str(_resolve(
            repository, config["base_candidate"]["config_path"]))
        report = base.run(base_args)
    finally:
        base.source.MultiRockFixture = original_fixture

    checks = dict(report["checks"])
    checks.update({
        "foundation_presettle_collider_parked_for_all_soil_only_steps":
            report["foundation_presettle"][
                "contact_samples_with_clearance_rock"] == int(
                    correction["required_foundation_rock_contact_samples"]),
        "single_factor_parking_scope_preserved": bool(
            not correction["material_changed"]
            and not correction["stability_thresholds_changed"]
            and not correction["dynamic_deposition_schedule_changed"]),
    })
    report["schema"] = REPORT_SCHEMA
    report["status"] = (
        "one_rock_soil_only_presettle_then_free_deposition_stable_passed"
        if all(checks.values()) else
        "one_rock_soil_only_presettle_then_free_deposition_stable_failed")
    report["checks"] = checks
    report["soil_only_presettle_correction"] = {
        "parking_position_world_m": parking.tolist(),
        "parked_assign_calls": parked_calls,
        "foundation_contact_samples": report["foundation_presettle"][
            "contact_samples_with_clearance_rock"],
        "dynamic_insertion_clearance_m": correction[
            "dynamic_insertion_surface_clearance_m"],
        "collider_velocity_mode": correction[
            "collider_velocity_mode_preserved"],
    }
    report["inputs"].update({
        "config_sha256": sha256(config_path),
        "wrapper_runner_sha256": sha256(Path(__file__).resolve()),
        "base_v1_config_sha256": config["base_candidate"]["config_sha256"],
        "base_v1_runner_sha256": config["base_candidate"]["runner_sha256"],
        "base_v1_failed_report_sha256": config["base_candidate"]["report_sha256"],
    })
    report["scope"] = config["scope"]
    report["next_gate"] = (
        "review_true_soil_first_deposition_before_any_three_rock_expansion")
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "soil_only_presettle_correction": report[
            "soil_only_presettle_correction"],
        "deposition": {
            key: value for key, value in report["deposition"].items()
            if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
