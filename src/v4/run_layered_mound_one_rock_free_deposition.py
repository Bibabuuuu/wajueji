#!/usr/bin/env python3
"""Settle active foundation soil, then dynamically deposit one free rock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

import run_layered_mound_three_rock_adaptive_press as source
from build_mixed_material_capacity_preflight import render_capacity_bridge, sha256
from run_layered_mixed_scene_local_dynamics_smoke import _build_provisional_model
from run_layered_mound_single_rock_press_smoke import (
    _finite_tree,
    _foundation_particles,
    _resolve,
)
from run_mixed_material_r3_a_probe import _compile_bridge, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_one_rock_free_deposition_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_one_rock_free_deposition_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "settled_active_foundation_then_one_rock_dynamic_from_first_contact"):
        raise ValueError("free_deposition_config_schema_or_status")
    for group, fields in (("source_candidate", ("config",)),
                          ("failed_hold_release_gate",
                           ("config", "runner", "report"))):
        binding = value.get(group, {})
        for field in fields:
            target = _resolve(repository, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"free_deposition_{group}_{field}_binding")
    base = source.load_config(_resolve(
        repository, value["source_candidate"]["config_path"]), repository)
    failed = json.loads(_resolve(
        repository, value["failed_hold_release_gate"]["report_path"]
    ).read_text(encoding="utf-8"))
    if (failed.get("status") !=
            "three_rocks_one_time_arrest_then_uncontrolled_stable_failed"
            or failed.get("checks", {}).get(
                "fully_uncontrolled_one_second_stable_window_reached") is not False
            or failed.get("checks", {}).get("release_state_finite") is not True):
        raise ValueError("free_deposition_failed_hold_release_contract")
    if value.get("deposition") != {
            "manifest_rock_id": "lower-000",
            "runtime_rock_id": "r1-rock-000",
            "foundation_presettle_frames": 60,
            "foundation_required_stable_frames": 30,
            "surface_clearance_m": 0.005,
            "initial_linear_velocity_m_s": [0.0, 0.0, 0.0],
            "initial_angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "rock_dynamic_from_first_mujoco_step": True,
            "kinematic_press": False,
            "held_contact_phase": False,
            "construction_support_force": False,
            "gravity_ramp": False,
            "maximum_settle_frames": 360,
            "required_continuous_stable_frames": 120}:
        raise ValueError("free_deposition_deposition_contract")
    if value.get("thresholds") != {
            "maximum_stable_rock_linear_speed_m_s": 0.01,
            "maximum_stable_rock_angular_speed_rad_s": 0.15,
            "maximum_stable_particle_speed_m_s": 0.05,
            "maximum_stable_particle_p99_speed_m_s": 0.01,
            "maximum_particle_speed_m_s": 5.0,
            "maximum_rock_linear_speed_m_s": 5.0,
            "maximum_rock_angular_speed_rad_s": 25.0,
            "maximum_forbidden_contact_count": 0,
            "minimum_world_z_m": -0.05}:
        raise ValueError("free_deposition_threshold_contract")
    if value.get("acceptance") != {
            "natural_fall_and_contact_allowed": True,
            "translation_and_rotation_allowed": True,
            "final_manifest_pose_required": False,
            "contact_required_before_qualification": True,
            "one_second_uncontrolled_stable_window_required": True}:
        raise ValueError("free_deposition_acceptance_contract")
    if value.get("scope") != {
            "active_soil": "unexcavated_foundation_stage_only",
            "active_rock_count": 1, "other_64_rocks_active": False,
            "cover_soil_active": False, "full_lower_layer": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("free_deposition_scope_contract")
    return value, base


def _particle_stable(metrics: dict, thresholds: dict) -> bool:
    return bool(
        metrics["finite"]
        and metrics["particle_max_speed_m_s"] <=
        float(thresholds["maximum_stable_particle_speed_m_s"])
        and metrics["particle_p99_speed_m_s"] <=
        float(thresholds["maximum_stable_particle_p99_speed_m_s"]))


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("free_deposition_output_exists")
    config, base = load_config(config_path, repository)
    deposition = config["deposition"]
    thresholds = config["thresholds"]
    manifest_path = _resolve(
        repository, base["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(rock for rock in manifest["rocks"]
                  if rock["rock_id"] == deposition["manifest_rock_id"])
    particle_q = _foundation_particles(base, repository)
    runtime_config = json.loads(json.dumps(base))
    runtime_config["construction"]["start_surface_clearance_m"] = float(
        deposition["surface_clearance_m"])

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    fixture = source.MultiRockFixture(
        runtime_config, repository, [record], particle_q, args.device)
    dt = float(base["probe"]["output_dt_s"])
    position = fixture.start_positions[0].copy()
    presettle_steps = int(deposition["foundation_presettle_frames"])
    required_foundation = int(deposition["foundation_required_stable_frames"])
    foundation_stable = 0
    foundation_contact_samples = 0
    foundation_peak_particle_speed = 0.0
    foundation_trace = []
    for frame in range(presettle_steps):
        fixture.assign([position], [np.zeros(3)])
        _, metrics = fixture.step(dt, [position])
        foundation_contact_samples += int(metrics["contact_samples"])
        foundation_peak_particle_speed = max(
            foundation_peak_particle_speed,
            float(metrics["particle_max_speed_m_s"]))
        foundation_stable = foundation_stable + 1 if _particle_stable(
            metrics, thresholds) else 0
        if frame == 0 or (frame + 1) % 30 == 0:
            foundation_trace.append({
                "frame": frame, "stable_window_frames": foundation_stable,
                **metrics})
        if not metrics["finite"]:
            raise RuntimeError("free_deposition_nonfinite_foundation")

    release_record = dict(record)
    release_record["position_world_m"] = position.tolist()
    capacity_path = _resolve(repository, base["capacity_bridge"]["config_path"])
    trace = []
    steps = 0
    contact_samples = 0
    first_contact_frame = None
    stable_count = 0
    max_linear = 0.0
    max_angular = 0.0
    max_particle = 0.0
    max_forbidden = 0
    all_finite = True
    centers_inside = True
    model_manifest = None
    bridge_manifest = None
    compile_command = None
    final_state = None
    visual = json.loads(_resolve(
        repository, base["unexcavated_soil"]["visual_config_path"]
    ).read_text(encoding="utf-8"))
    outer = np.asarray(
        visual["mound_envelope"]["outer_xy_bounds_m"], dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="one-rock-free-deposition-") as raw:
        work = Path(raw)
        model = work / "deposition.xml"
        model_manifest = _build_provisional_model(
            capacity_path, repository, {"rocks": [release_record]}, model)
        source_path = work / "bridge.cc"
        source_code, bridge_manifest = render_capacity_bridge(
            capacity_path, repository)
        source_path.write_text(source_code, encoding="utf-8")
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source_path, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        bridge = source.RuntimeBridge(binary, model, 1)
        try:
            initial = bridge.state["rocks"][0]
            if (not np.allclose(initial["position"], position, rtol=0.0, atol=1e-6)
                    or not np.allclose(initial["linear_velocity_world_m_s"],
                                       deposition["initial_linear_velocity_m_s"],
                                       rtol=0.0, atol=1e-12)
                    or not np.allclose(initial["angular_velocity_world_rad_s"],
                                       deposition["initial_angular_velocity_rad_s"],
                                       rtol=0.0, atol=1e-12)):
                raise RuntimeError("free_deposition_initial_dynamic_state")
            for frame in range(int(deposition["maximum_settle_frames"])):
                state = bridge.state
                fixture.assign_bridge(state)
                origin = np.asarray(state["rocks"][0]["position"], dtype=np.float64)
                measured, particle = fixture.step(dt, [origin])
                state = bridge.step(
                    dt, int(base["probe"]["mujoco_internal_substeps"]),
                    {"bucket": source._zero(state["bucket_origin"]), **measured})
                steps += 1
                samples = int(particle["contact_samples"])
                contact_samples += samples
                if samples and first_contact_frame is None:
                    first_contact_frame = frame
                rock = state["rocks"][0]
                linear = float(np.linalg.norm(
                    rock["linear_velocity_world_m_s"]))
                angular = float(np.linalg.norm(
                    rock["angular_velocity_world_rad_s"]))
                max_linear = max(max_linear, linear)
                max_angular = max(max_angular, angular)
                max_particle = max(
                    max_particle, float(particle["particle_max_speed_m_s"]))
                max_forbidden = max(
                    max_forbidden, int(state.get("forbidden_contact_count", 0)))
                all_finite &= bool(particle["finite"] and _finite_tree(state))
                point = np.asarray(rock["position"], dtype=np.float64)
                centers_inside &= bool(
                    outer[0, 0] <= point[0] <= outer[1, 0]
                    and outer[0, 1] <= point[1] <= outer[1, 1]
                    and point[2] >= float(thresholds["minimum_world_z_m"]))
                stable = bool(
                    first_contact_frame is not None
                    and linear <= float(thresholds[
                        "maximum_stable_rock_linear_speed_m_s"])
                    and angular <= float(thresholds[
                        "maximum_stable_rock_angular_speed_rad_s"])
                    and _particle_stable(particle, thresholds))
                stable_count = stable_count + 1 if stable else 0
                if (frame == 0 or samples and first_contact_frame == frame
                        or (frame + 1) % 30 == 0
                        or stable_count == int(deposition[
                            "required_continuous_stable_frames"])):
                    trace.append({
                        "frame": frame, "stable_window_frames": stable_count,
                        "rock_position_world_m": point.tolist(),
                        "rock_linear_speed_m_s": linear,
                        "rock_angular_speed_rad_s": angular,
                        **particle})
                if stable_count >= int(
                        deposition["required_continuous_stable_frames"]):
                    break
            final_state = bridge.state
        finally:
            bridge.close(kill=not all_finite)

    final = final_state["rocks"][0]
    final_position = np.asarray(final["position"], dtype=np.float64)
    final_linear = float(np.linalg.norm(final["linear_velocity_world_m_s"]))
    final_angular = float(np.linalg.norm(final["angular_velocity_world_rad_s"]))
    checks = {
        "failed_hold_release_gate_bound": True,
        "foundation_active_and_finite_before_deposition":
            foundation_stable >= required_foundation,
        "no_rock_soil_contact_during_foundation_presettle":
            foundation_contact_samples == 0,
        "rock_zero_velocity_at_dynamic_insertion": True,
        "no_kinematic_press_hold_support_or_gravity_ramp": bool(
            not deposition["kinematic_press"]
            and not deposition["held_contact_phase"]
            and not deposition["construction_support_force"]
            and not deposition["gravity_ramp"]),
        "natural_rock_soil_contact_observed": contact_samples > 0,
        "coupled_state_finite": all_finite,
        "rock_speed_safety_bound": max_linear <= float(
            thresholds["maximum_rock_linear_speed_m_s"]),
        "rock_angular_safety_bound": max_angular <= float(
            thresholds["maximum_rock_angular_speed_rad_s"]),
        "particle_speed_safety_bound": max_particle <= float(
            thresholds["maximum_particle_speed_m_s"]),
        "forbidden_contacts_zero": max_forbidden <= int(
            thresholds["maximum_forbidden_contact_count"]),
        "rock_center_remains_in_foundation_bounds": centers_inside,
        "one_second_uncontrolled_stable_window_reached": stable_count >= int(
            deposition["required_continuous_stable_frames"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": (
            "one_rock_free_deposition_on_active_soil_stable_passed"
            if all(checks.values()) else
            "one_rock_free_deposition_on_active_soil_stable_failed"),
        "checks": checks,
        "foundation_presettle": {
            "steps": presettle_steps,
            "stable_window_frames": foundation_stable,
            "contact_samples_with_clearance_rock": foundation_contact_samples,
            "peak_particle_speed_m_s": foundation_peak_particle_speed,
            "trace": foundation_trace,
        },
        "deposition": {
            "manifest_rock_id": record["rock_id"],
            "runtime_rock_id": deposition["runtime_rock_id"],
            "mass_kg": record["mass_kg"],
            "surface_clearance_m": deposition["surface_clearance_m"],
            "start_position_world_m": position.tolist(),
            "steps": steps,
            "simulated_seconds": steps * dt,
            "first_contact_frame": first_contact_frame,
            "contact_samples": contact_samples,
            "stable_window_frames": stable_count,
            "maximum_rock_linear_speed_m_s": max_linear,
            "maximum_rock_angular_speed_rad_s": max_angular,
            "maximum_particle_speed_m_s": max_particle,
            "maximum_forbidden_contact_count": max_forbidden,
            "final_position_world_m": final_position.tolist(),
            "translation_from_start_m": float(np.linalg.norm(
                final_position - position)),
            "final_linear_speed_m_s": final_linear,
            "final_angular_speed_rad_s": final_angular,
            "trace": trace,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_candidate_config_sha256": config[
                "source_candidate"]["config_sha256"],
            "failed_hold_release_report_sha256": config[
                "failed_hold_release_gate"]["report_sha256"],
            "physical_manifest_sha256": sha256(manifest_path),
            "model_manifest": model_manifest,
            "bridge_manifest": bridge_manifest,
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "compile_command": compile_command,
            "newton_solver_step_calls": presettle_steps + steps,
            "mujoco_step_calls": steps,
            "wall_seconds": time.perf_counter() - started,
        },
        "acceptance": config["acceptance"],
        "scope": config["scope"],
        "stable_scene_claimed": False,
        "next_gate": "review_one_rock_free_deposition_before_any_three_rock_test",
    }
    if not _finite_tree(report):
        raise RuntimeError("free_deposition_nonfinite_report")
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
        "foundation_presettle": {
            key: value for key, value in report["foundation_presettle"].items()
            if key != "trace"},
        "deposition": {
            key: value for key, value in report["deposition"].items()
            if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
