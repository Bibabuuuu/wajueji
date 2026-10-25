#!/usr/bin/env python3
"""Qualify all nine lower-layer rocks on excavatable MPM soil.

This deliberately reuses the frozen three-rock construction runner, replacing
only its data contract and the already-qualified free-rigid proxy velocity.
Held and kinematic construction phases remain unchanged.  After the inherited
release gate passes, the same live MPM and MuJoCo states continue for another
two seconds without resets, support, damping, or velocity assignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

import run_layered_mound_three_rock_adaptive_press as shared
import run_layered_mound_three_rock_cover_release as base
from build_mixed_material_capacity_preflight import sha256
from mpm_rigid_free_velocity import predict_free_linear_velocity
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_nine_rock_lower_layer_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_nine_rock_lower_layer_report_v1"
TARGET_IDS = [f"lower-{index:03d}" for index in range(9)]
PRESS_SCHEDULE = [
    (0.012, 18), (0.008, 12), (0.016, 24),
    (0.010, 15), (0.014, 21), (0.007, 11),
    (0.011, 17), (0.009, 14), (0.013, 20),
]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "nine_variable_pressed_rocks_cover_settle_then_free_release"):
        raise ValueError("nine_lower_config_schema_or_status")
    for group, fields in (
            ("predecessor", ("config", "runner", "coupling_helper")),
            ("failed_bare_release_gate", ("config", "runner", "report")),
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("source", ("visual_config", "source_bank"))):
        binding = config.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if not target.is_file() or sha256(target) != binding.get(field + "_sha256"):
                raise ValueError(f"nine_lower_{group}_{field}_binding")
    targets = config.get("targets", [])
    if [item.get("manifest_rock_id") for item in targets] != TARGET_IDS:
        raise ValueError("nine_lower_target_ids")
    if [(float(item["press_distance_m"]), int(item["press_frames"]))
            for item in targets] != PRESS_SCHEDULE:
        raise ValueError("nine_lower_press_schedule")
    if config.get("particles") != {
            "foundation_maximum_center_height_m": 0.32,
            "cover_maximum_center_height_m": 0.56,
            "foundation_particle_count": 85620,
            "raw_cover_particle_count": 49704,
            "cover_particle_count_after_pressed_rock_carve": 43261,
            "combined_particle_count": 128881,
            "cover_carve_policy":
            "remove_cover_particle_centers_inside_nine_pressed_rocks_only",
            "activation_policy":
            "foundation_active_from_start_cover_dormant_then_history_restored"}:
        raise ValueError("nine_lower_particle_contract")
    construction = config.get("construction", {})
    if (construction.get("foundation_presettle_frames") != 24
            or construction.get("cover_maximum_settle_frames") != 240
            or construction.get("cover_stable_window_frames") != 60
            or construction.get("temporary_mpm_damping") != 0.0
            or construction.get("temporary_rigid_damping") != 0.0
            or construction.get("free_fall_from_height") is not False):
        raise ValueError("nine_lower_construction_contract")
    probe = config.get("probe", {})
    if (probe.get("release_stable_window_frames") != 120
            or probe.get("additional_uncontrolled_hold_frames") != 240):
        raise ValueError("nine_lower_verification_contract")
    if config.get("scope") != {
            "active_rock_count": 9, "other_56_rocks_active": False,
            "full_lower_layer": True, "full_scene": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("nine_lower_scope_contract")
    return config


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("nine_lower_output_exists")
    config = load_config(config_path, root)
    gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=float)
    dt = float(config["probe"]["output_dt_s"])
    hold_frames = int(config["probe"]["additional_uncontrolled_hold_frames"])
    traces: list[dict] = []
    hold_trace: list[dict] = []
    instances = []

    OriginalFixture = base.MultiRockFixture
    original_load_config = base.load_config
    original_bridge_step = base.RuntimeBridge.step
    original_bridge_close = base.RuntimeBridge.close

    class PredictorFixture(OriginalFixture):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            instances.append(self)
            self.prediction_calls = 0

        def assign_bridge(self, state):
            self.prediction_calls += 1
            self.assign(
                [np.asarray(rock["position"], dtype=float)
                 for rock in state["rocks"]],
                [predict_free_linear_velocity(
                    np.asarray(rock["linear_velocity_world_m_s"], dtype=float),
                    gravity, dt) for rock in state["rocks"]],
                [np.asarray(rock["angular_velocity_world_rad_s"], dtype=float)
                 for rock in state["rocks"]],
                [rock["qpos"][3:7] for rock in state["rocks"]])

    def measured_step(bridge, step_dt, substeps, wrenches):
        before = bridge.state
        after = original_bridge_step(bridge, step_dt, substeps, wrenches)
        rows = []
        for prior, current, record in zip(
                before["rocks"], after["rocks"], instances[0].records):
            mass = float(record["mass_kg"])
            force = np.asarray(wrenches[prior["rock_id"]]["force"], dtype=float)
            delta_p = mass * (np.asarray(current["qvel"][:3])
                              - np.asarray(prior["qvel"][:3]))
            expected = step_dt * (force + mass * gravity)
            rows.append({
                "runtime_rock_id": prior["rock_id"],
                "position_world_m": current["qpos"][:3],
                "raw_linear_velocity_world_m_s": current["qvel"][:3],
                "reported_linear_velocity_world_m_s":
                    current["linear_velocity_world_m_s"],
                "mpm_force_n": force.tolist(),
                "vertical_reaction_over_weight":
                    float(force[2] / (-mass * gravity[2])),
                "linear_momentum_residual_ns":
                    float(np.linalg.norm(delta_p - expected)),
            })
        traces.append({
            "frame": len(traces), "rocks": rows,
            "rigid_contact_substeps": after.get("contact_substeps", {})})
        return after

    def close_after_hold(bridge, kill=False):
        try:
            if not kill:
                fixture = instances[0]
                for frame in range(hold_frames):
                    state = bridge.state
                    fixture.assign_bridge(state)
                    origins = [np.asarray(rock["position"], dtype=float)
                               for rock in state["rocks"]]
                    measured, particles = fixture.step(dt, origins)
                    state = bridge.step(
                        dt, int(config["probe"]["mujoco_internal_substeps"]),
                        {"bucket": shared._zero(state["bucket_origin"]), **measured})
                    linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                                 for rock in state["rocks"])
                    angular = max(float(np.linalg.norm(
                        rock["angular_velocity_world_rad_s"]))
                        for rock in state["rocks"])
                    stable = (
                        linear <= float(config["probe"][
                            "maximum_stable_rock_linear_speed_m_s"])
                        and angular <= float(config["probe"][
                            "maximum_stable_rock_angular_speed_rad_s"])
                        and particles["particle_max_speed_m_s"] <= float(
                            config["probe"]["maximum_stable_particle_speed_m_s"])
                        and particles["particle_p99_speed_m_s"] <= float(
                            config["probe"]["maximum_stable_particle_p99_speed_m_s"]))
                    hold_trace.append({
                        "frame": frame, "stable": bool(stable),
                        "max_raw_linear_speed_m_s": linear,
                        "max_angular_speed_rad_s": angular, **particles})
                    if not particles["finite"] or not np.isfinite(linear + angular):
                        kill = True
                        break
        finally:
            original_bridge_close(bridge, kill=kill)

    base.load_config = lambda path, repository: config
    base.MultiRockFixture = PredictorFixture
    base.RuntimeBridge.step = measured_step
    base.RuntimeBridge.close = close_after_hold
    try:
        report = base.run(args)
    finally:
        base.load_config = original_load_config
        base.MultiRockFixture = OriginalFixture
        base.RuntimeBridge.step = original_bridge_step
        base.RuntimeBridge.close = original_bridge_close

    checks = dict(report["checks"])
    checks.pop("bare_release_failure_bound", None)
    checks.pop("unequal_press_schedule_preserved", None)
    checks.update({
        "three_rock_predecessor_bound": True,
        "all_nine_lower_rocks_active": len(instances[0].records) == 9,
        "nine_rock_press_schedule_preserved": [
            (item["press_distance_m"], item["press_frames"])
            for item in report["construction"]["presses"]] == PRESS_SCHEDULE,
        "gravity_and_mpm_impulse_accounted_once": max(
            row["linear_momentum_residual_ns"]
            for frame in traces for row in frame["rocks"]) < 1e-4,
        "no_rigid_ground_or_bucket_support": max(
            (count for frame in traces
             for count in frame["rigid_contact_substeps"].values()),
            default=0) == 0,
        "predictor_called_once_per_dynamic_step":
            instances[0].prediction_calls == len(traces),
        "additional_uncontrolled_hold_all_frames_stable":
            len(hold_trace) == hold_frames
            and all(row["stable"] and row["finite"] for row in hold_trace),
    })
    tail = traces[-120:]
    final_rows = traces[-1]["rocks"]
    report["schema"] = REPORT_SCHEMA
    report["original_runner_status"] = report["status"]
    report["status"] = ("nine_rock_lower_layer_passed"
                        if all(checks.values()) else
                        "nine_rock_lower_layer_failed")
    report["checks"] = checks
    report["coupling"] = {
        "formula": "mpm_proxy_v=rigid_v+g*dt; actual_mujoco_state_not_mutated",
        "dynamic_frames": len(traces),
        "maximum_linear_momentum_residual_ns": max(
            row["linear_momentum_residual_ns"]
            for frame in traces for row in frame["rocks"]),
        "maximum_rigid_contact_substeps": max(
            (count for frame in traces
             for count in frame["rigid_contact_substeps"].values()), default=0),
        "last_120_frames_max_raw_linear_speed_m_s": max(
            float(np.linalg.norm(row["raw_linear_velocity_world_m_s"]))
            for frame in tail for row in frame["rocks"]),
        "last_120_frames_position_change_m": {
            TARGET_IDS[index]: float(np.linalg.norm(
                np.asarray(final_rows[index]["position_world_m"])
                - np.asarray(tail[0]["rocks"][index]["position_world_m"])))
            for index in range(9)},
        "final_rocks": final_rows,
    }
    report["additional_uncontrolled_hold"] = {
        "requested_frames": hold_frames, "executed_frames": len(hold_trace),
        "all_frames_stable": (len(hold_trace) == hold_frames
                              and all(row["stable"] for row in hold_trace)),
        "trace": hold_trace,
    }
    report["inputs"]["nine_rock_config_sha256"] = sha256(config_path)
    report["inputs"]["nine_rock_runner_sha256"] = sha256(Path(__file__).resolve())
    report["execution"]["newton_solver_step_calls"] += len(hold_trace)
    report["execution"]["mujoco_step_calls"] = len(traces)
    report["execution"]["wall_seconds"] = time.perf_counter() - started
    report["scope"] = config["scope"]
    report["stable_scene_claimed"] = False
    report["next_gate"] = "review_full_lower_layer_before_adding_second_layer"
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
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "particles": report["particles"], "coupling": report["coupling"],
        "execution": report["execution"]}, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "nine_rock_lower_layer_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
