#!/usr/bin/env python3
"""Apply one construction arrest after the three-rock slow release ramp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_layered_mound_three_rock_slow_release as slow
from build_mixed_material_capacity_preflight import sha256
from run_layered_mound_single_rock_press_smoke import _resolve
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_three_rock_arrest_release_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_three_rock_arrest_release_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "single_construction_arrest_after_slow_release_ramp"):
        raise ValueError("arrest_release_config_schema_or_status")
    binding = value.get("base_candidate", {})
    for field in ("config", "runner", "report"):
        target = _resolve(repository, binding.get(field + "_path", ""))
        if (not target.is_file() or sha256(target) !=
                binding.get(field + "_sha256")):
            raise ValueError(f"arrest_release_base_{field}_binding")
    slow_config, cover_config = slow.load_config(
        _resolve(repository, binding["config_path"]), repository)
    report = json.loads(_resolve(
        repository, binding["report_path"]).read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    if (report.get("status") !=
            "three_rocks_slow_release_post_ramp_stable_failed"
            or checks.get("gravity_release_ramp_completed") is not True
            or checks.get("fully_released_post_ramp_stable_window_reached") is not False
            or not all(result for name, result in checks.items()
                       if name in {
                           "release_state_finite", "release_contact_observed",
                           "release_speed_safety_bounds",
                           "release_forbidden_contacts_zero",
                           "release_centers_remain_in_bounds"})):
        raise ValueError("arrest_release_base_failure_contract")
    if value.get("correction") != {
            "reason":
            "remove_the_measured_one_external_step_construction_velocity_before_final_detachment",
            "change_scope": [
                "release.one_time_external_linear_arrest_force_after_gravity_ramp"],
            "gravity_release_frames_preserved": 60,
            "arrest_frame_zero_based": 60,
            "arrest_steps": 1,
            "arrest_force_formula": "F=-mass*measured_linear_velocity/dt",
            "angular_arrest_applied": False,
            "required_fully_uncontrolled_frames_after_arrest": 120,
            "target_runtime_rock_ids": [
                "r1-rock-000", "r1-rock-001", "r1-rock-002"],
            "mpm_material_changed": False,
            "rigid_damping_changed": False,
            "stability_thresholds_changed": False,
            "press_schedule_changed": False,
            "cover_schedule_changed": False,
            "gravity_release_schedule_changed": False,
            "recurring_velocity_feedback": False}:
        raise ValueError("arrest_release_correction_contract")
    if value.get("acceptance") != {
            "rock_translation_during_release_allowed": True,
            "rock_rotation_during_release_allowed": True,
            "final_manifest_pose_required": False,
            "one_time_construction_arrest_allowed": True,
            "fully_uncontrolled_post_arrest_window_required": True}:
        raise ValueError("arrest_release_acceptance_contract")
    if value.get("scope") != {
            "active_rock_count": 3, "other_62_rocks_active": False,
            "full_lower_layer": False, "full_scene": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("arrest_release_scope_contract")
    return value, {"slow": slow_config, "cover": cover_config}


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("arrest_release_output_exists")
    config, bases = load_config(config_path, repository)
    correction = config["correction"]
    cover_config = bases["cover"]
    manifest = json.loads(_resolve(
        repository, cover_config["physical_geometry"]["manifest_path"]
    ).read_text(encoding="utf-8"))
    by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
    masses = [float(by_id[target["manifest_rock_id"]]["mass_kg"])
              for target in cover_config["targets"]]
    runtime_ids = list(correction["target_runtime_rock_ids"])
    arrest_frame = int(correction["arrest_frame_zero_based"])
    arrest_state = {"calls": 0, "applications": []}
    original_step = slow.base.RuntimeBridge.step

    def step_with_one_time_arrest(self, dt: float, substeps: int,
                                  wrenches: dict[str, dict]) -> dict:
        frame = arrest_state["calls"]
        adjusted = wrenches
        if frame == arrest_frame:
            adjusted = {}
            for body_name, wrench in wrenches.items():
                adjusted[body_name] = {
                    "force": np.asarray(wrench["force"], dtype=np.float64).copy(),
                    "torque": np.asarray(wrench["torque"], dtype=np.float64).copy(),
                    "point": np.asarray(wrench["point"], dtype=np.float64).copy(),
                }
            rocks = {rock["rock_id"]: rock for rock in self.state["rocks"]}
            application = {"frame": frame, "dt_s": float(dt), "rocks": []}
            for runtime_id, mass in zip(runtime_ids, masses, strict=True):
                velocity = np.asarray(
                    rocks[runtime_id]["linear_velocity_world_m_s"],
                    dtype=np.float64)
                force = -mass * velocity / float(dt)
                adjusted[runtime_id]["force"] += force
                application["rocks"].append({
                    "runtime_rock_id": runtime_id,
                    "mass_kg": mass,
                    "measured_linear_velocity_world_m_s": velocity.tolist(),
                    "applied_arrest_force_world_n": force.tolist(),
                })
            arrest_state["applications"].append(application)
        arrest_state["calls"] += 1
        return original_step(self, dt, substeps, adjusted)

    slow.base.RuntimeBridge.step = step_with_one_time_arrest
    try:
        slow_args = argparse.Namespace(**vars(args))
        slow_args.config = str(_resolve(
            repository, config["base_candidate"]["config_path"]))
        report = slow.run(slow_args)
    finally:
        slow.base.RuntimeBridge.step = original_step

    release_steps = int(report["release"]["steps"])
    required_uncontrolled = int(
        correction["required_fully_uncontrolled_frames_after_arrest"])
    uncontrolled_steps = max(0, release_steps - arrest_frame - 1)
    base_stable = bool(
        report["checks"]["release_continuous_stable_window_reached"]
        and int(report["release"]["stable_window_frames"]) >= required_uncontrolled)
    checks = dict(report["checks"])
    checks.pop("fully_released_post_ramp_stable_window_reached", None)
    checks.update({
        "one_time_construction_arrest_applied":
            len(arrest_state["applications"]) == 1,
        "no_post_arrest_support_or_control": bool(
            len(arrest_state["applications"]) == 1
            and arrest_state["applications"][0]["frame"] == arrest_frame
            and not correction["recurring_velocity_feedback"]),
        "fully_uncontrolled_one_second_stable_window_reached": bool(
            base_stable and uncontrolled_steps >= required_uncontrolled),
        "single_factor_arrest_scope_preserved": all(not correction[name] for name in (
            "mpm_material_changed", "rigid_damping_changed",
            "stability_thresholds_changed", "press_schedule_changed",
            "cover_schedule_changed", "gravity_release_schedule_changed",
            "recurring_velocity_feedback")),
    })
    report["schema"] = REPORT_SCHEMA
    report["status"] = (
        "three_rocks_one_time_arrest_then_uncontrolled_stable_passed"
        if all(checks.values()) else
        "three_rocks_one_time_arrest_then_uncontrolled_stable_failed")
    report["checks"] = checks
    report["construction_arrest"] = {
        "method": "one_external_step_linear_impulse_from_measured_velocity",
        "frame_zero_based": arrest_frame,
        "applications": arrest_state["applications"],
        "bridge_step_calls": arrest_state["calls"],
        "fully_uncontrolled_frames_after_arrest": uncontrolled_steps,
        "required_fully_uncontrolled_frames": required_uncontrolled,
        "post_arrest_recurring_support_or_control": False,
    }
    report["inputs"].update({
        "config_sha256": sha256(config_path),
        "wrapper_runner_sha256": sha256(Path(__file__).resolve()),
        "base_slow_config_sha256": config["base_candidate"]["config_sha256"],
        "base_slow_runner_sha256": config["base_candidate"]["runner_sha256"],
        "base_slow_failed_report_sha256": config["base_candidate"]["report_sha256"],
    })
    report["acceptance"] = config["acceptance"]
    report["scope"] = config["scope"]
    report["next_gate"] = (
        "review_final_three_rock_construction_gate_and_stop_route_if_failed")
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
        "construction_arrest": report["construction_arrest"],
        "release": {key: value for key, value in report["release"].items()
                    if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
