#!/usr/bin/env python3
"""Controlled A/B of gravity-aware MPM proxies, using frozen scene runners.

No velocity assignment reaches MuJoCo. Per-step diagnostics compare its raw
free-joint momentum change against gravity plus the measured MPM impulse.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import run_layered_mound_one_rock_free_deposition_v2 as one
import run_layered_mound_three_rock_adaptive_press as shared
import run_layered_mound_three_rock_cover_release as three
from build_mixed_material_capacity_preflight import sha256
from mpm_rigid_free_velocity import predict_free_linear_velocity
from run_mixed_material_r3_a_probe import _write_json


def run(args):
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((root / "configs/v4/mpm-rigid-free-velocity-probe.json").read_text())
    for relative, digest in cfg["frozen_files"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"frozen_file_changed:{relative}")
    mode = args.case
    entry = one if mode == "one" else three
    config_path = root / cfg["cases"][mode]
    if mode == "one":
        _, original = one.load_config(config_path, root)
        _, physical = one.base.load_config(
            root / json.loads(config_path.read_text())["base_candidate"]["config_path"], root)
        dt = float(physical["probe"]["output_dt_s"])
    else:
        physical = three.load_config(config_path, root)
        dt = float(physical["probe"]["output_dt_s"])
    gravity = np.asarray(physical["mpm"]["gravity_m_s2"], dtype=float)
    trace = []
    hold_trace = []
    instances = []
    OriginalFixture = shared.MultiRockFixture
    original_bridge_step = shared.RuntimeBridge.step
    original_bridge_close = shared.RuntimeBridge.close
    old_three_fixture = three.MultiRockFixture

    class PredictorFixture(OriginalFixture):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            instances.append(self)
            self.prediction_calls = 0

        def assign_bridge(self, state):
            self.prediction_calls += 1
            velocities = [np.asarray(rock["linear_velocity_world_m_s"], dtype=float)
                          for rock in state["rocks"]]
            if args.predictor == "on":
                velocities = [predict_free_linear_velocity(v, gravity, dt)
                              for v in velocities]
            self.assign(
                [np.asarray(rock["position"], dtype=float) for rock in state["rocks"]],
                velocities,
                [np.asarray(rock["angular_velocity_world_rad_s"], dtype=float)
                 for rock in state["rocks"]],
                [rock["qpos"][3:7] for rock in state["rocks"]])

    def measured_step(bridge, step_dt, substeps, wrenches):
        before = bridge.state
        # The exact force is passed through without adding support or damping.
        after = original_bridge_step(bridge, step_dt, substeps, wrenches)
        records = instances[0].records
        rows = []
        for i, (a, b, record) in enumerate(zip(before["rocks"], after["rocks"], records)):
            mass = float(record["mass_kg"])
            force = np.asarray(wrenches[a["rock_id"]]["force"], dtype=float)
            # Free-joint qvel[0:3] is world translation; avoids cached cvel lag.
            delta_p = mass * (np.asarray(b["qvel"][:3]) - np.asarray(a["qvel"][:3]))
            expected = step_dt * (force + mass * gravity)
            rows.append({
                "id": a["rock_id"], "mass_kg": mass,
                "position": b["qpos"][:3],
                "velocity": b["qvel"][:3],
                "reported_velocity": b["linear_velocity_world_m_s"],
                "mpm_force_n": force.tolist(),
                "vertical_reaction_over_weight": float(force[2] / (-mass * gravity[2])),
                "linear_momentum_residual_ns": float(np.linalg.norm(delta_p - expected)),
                "cached_velocity_difference_m_s": float(np.linalg.norm(
                    np.asarray(b["qvel"][:3]) - np.asarray(b["linear_velocity_world_m_s"]))),
            })
        trace.append({"frame": len(trace), "rocks": rows,
                      "rigid_contact_substeps": after.get("contact_substeps", {})})
        return after

    def close_after_verification(bridge, kill=False):
        # Continue the SAME live states after the legacy qualification loop.
        # No reset, support, or velocity clamping. The legacy report describes
        # the earlier gate; verification below separately audits the extra time.
        try:
            if not kill:
                for frame in range(args.verification_hold_frames):
                    state = bridge.state
                    fixture = instances[0]
                    fixture.assign_bridge(state)
                    origins = [np.asarray(r["position"], dtype=float) for r in state["rocks"]]
                    measured, particles = fixture.step(dt, origins)
                    state = bridge.step(dt, int(physical["probe"]["mujoco_internal_substeps"]),
                                        {"bucket": shared._zero(state["bucket_origin"]), **measured})
                    speed = max(float(np.linalg.norm(r["qvel"][:3])) for r in state["rocks"])
                    angular = max(float(np.linalg.norm(r["angular_velocity_world_rad_s"]))
                                  for r in state["rocks"])
                    stable = (speed <= 0.01 and angular <= 0.15
                              and particles["particle_max_speed_m_s"] <= 0.05
                              and particles["particle_p99_speed_m_s"] <= 0.01)
                    hold_trace.append({"frame": frame, "stable": bool(stable),
                                       "max_raw_linear_speed_m_s": speed,
                                       "max_angular_speed_rad_s": angular, **particles})
                    if not particles["finite"] or not np.isfinite(speed + angular):
                        kill = True
                        break
        finally:
            original_bridge_close(bridge, kill=kill)

    shared.MultiRockFixture = PredictorFixture
    three.MultiRockFixture = PredictorFixture
    shared.RuntimeBridge.step = measured_step
    shared.RuntimeBridge.close = close_after_verification
    try:
        legacy_args = argparse.Namespace(**vars(args))
        legacy_args.config = str(config_path)
        report = entry.run(legacy_args)
    finally:
        shared.MultiRockFixture = OriginalFixture
        three.MultiRockFixture = old_three_fixture
        shared.RuntimeBridge.step = original_bridge_step
        shared.RuntimeBridge.close = original_bridge_close
    final_rows = trace[-1]["rocks"]
    tail = trace[-120:]
    momentum_max = max(row["linear_momentum_residual_ns"] for frame in trace
                       for row in frame["rocks"])
    contact_max = max((n for frame in trace for n in frame["rigid_contact_substeps"].values()),
                      default=0)
    checks = dict(report["checks"])
    checks.update({
        "gravity_and_mpm_impulse_accounted_once": momentum_max < 1e-4,
        "no_rigid_ground_or_bucket_support": contact_max == 0,
        "predictor_called_once_per_dynamic_step": instances[0].prediction_calls == len(trace),
    })
    if args.verification_hold_frames:
        checks["additional_uncontrolled_hold_all_frames_stable"] = (
            len(hold_trace) == args.verification_hold_frames
            and all(row["stable"] and row["finite"] for row in hold_trace))
    report["original_runner_status"] = report["status"]
    report["schema"] = "mpm_rigid_free_velocity_probe_v1"
    report["status"] = "passed" if all(checks.values()) else "failed"
    report["checks"] = checks
    report["coupling"] = {
        "case": mode, "predictor": args.predictor,
        "formula": ("mpm_proxy_v=rigid_v+g*dt" if args.predictor == "on" else "mpm_proxy_v=rigid_v")
                   + "; mujoco advances original state with g and measured_J/dt",
        "dt_s": dt, "gravity": gravity.tolist(),
        "dynamic_frames": len(trace),
        "max_linear_momentum_residual_ns": momentum_max,
        "max_rigid_contact_substeps": contact_max,
        "last_120_frames_max_raw_linear_speed_m_s": max(
            float(np.linalg.norm(row["velocity"])) for frame in tail for row in frame["rocks"]),
        "last_120_frames_position_change_m": [float(np.linalg.norm(
            np.asarray(final_rows[i]["position"]) - np.asarray(tail[0]["rocks"][i]["position"])))
            for i in range(len(final_rows))],
        "final_rocks": final_rows,
        "trace": trace,
    }
    report["additional_verification_hold"] = {
        "requested_frames": args.verification_hold_frames,
        "executed_frames": len(hold_trace), "trace": hold_trace,
        "all_frames_stable": bool(hold_trace) and all(row["stable"] for row in hold_trace),
    }
    report["execution"]["legacy_mujoco_step_calls"] = report["execution"]["mujoco_step_calls"]
    report["execution"]["mujoco_step_calls"] = len(trace)
    report["execution"]["newton_solver_step_calls"] += len(hold_trace)
    report["execution"]["wall_seconds"] = time.perf_counter() - started
    report["probe_inputs"] = {
        "runner_sha256": sha256(Path(__file__)),
        "predictor_sha256": sha256(root / "src/v4/mpm_rigid_free_velocity.py"),
        "config_sha256": sha256(root / "configs/v4/mpm-rigid-free-velocity-probe.json"),
        "frozen_files": cfg["frozen_files"],
    }
    report["stable_scene_claimed"] = False
    _write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=["one", "three"], default="one")
    parser.add_argument("--predictor", choices=["off", "on"], required=True)
    parser.add_argument("--verification-hold-frames", type=int, default=0,
                        choices=[0, 120, 240])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "coupling": {k: v for k, v in report["coupling"].items() if k != "trace"},
                      "execution": report["execution"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
