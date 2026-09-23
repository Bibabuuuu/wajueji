#!/usr/bin/env python3
"""Runtime primitives shared by checkpoint continuation and read-only audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import newton
import numpy as np

from layered_mound_checkpoint import (
    restore_fixture_prefix,
    restore_solver_continuation,
)
from mpm_rigid_free_velocity import predict_free_linear_velocity
from run_layered_mixed_scene_local_dynamics_smoke import _build_provisional_model
from run_layered_mound_three_rock_adaptive_press import (
    MultiRockFixture,
    RuntimeBridge,
    _finite_tree,
    _zero,
)
from run_layered_mound_three_rock_cover_release import _stable_particle
from run_mixed_material_r3_a_probe import _compile_bridge
from build_mixed_material_capacity_preflight import render_capacity_bridge


def records_at_qpos(records: list[dict], qpos: np.ndarray) -> list[dict]:
    if qpos.shape != (len(records), 7):
        raise ValueError("checkpoint_record_qpos_shape")
    result = []
    for record, pose in zip(records, qpos, strict=True):
        item = copy.deepcopy(record)
        item["position_world_m"] = pose[:3].tolist()
        item["orientation_quaternion_wxyz"] = pose[3:7].tolist()
        result.append(item)
    return result


def build_model_and_bridge(
        args, root: Path, capacity_config: Path, records: list[dict],
        work: Path, label: str, binary: Path | None = None) -> tuple[Path, Path, dict, dict, list[str]]:
    model = work / f"{label}.xml"
    model_manifest = _build_provisional_model(
        capacity_config, root, {"rocks": records}, model)
    source = work / "capacity-bridge.cc"
    if binary is None:
        code, bridge_manifest = render_capacity_bridge(capacity_config, root)
        source.write_text(code, encoding="utf-8")
        binary = work / "capacity-bridge"
        compile_command = _compile_bridge(
            args.cxx, source, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
    else:
        _, bridge_manifest = render_capacity_bridge(capacity_config, root)
        compile_command = []
    return model, binary, model_manifest, bridge_manifest, compile_command


class RestoredRuntimeBridge(RuntimeBridge):
    """Initialize the existing bridge protocol with explicit qpos/qvel."""

    def __init__(self, binary: Path, model: Path, count: int,
                 mechanism_qpos: list[float], mechanism_qvel: list[float],
                 rock_qpos: np.ndarray, rock_qvel: np.ndarray,
                 controller_state: dict | None = None):
        self.count = count
        self.ids = [f"r1-rock-{index:03d}" for index in range(count)]
        if (len(mechanism_qpos) != 19 or len(mechanism_qvel) != 18
                or rock_qpos.shape != (count, 7)
                or rock_qvel.shape != (count, 6)):
            raise ValueError("restored_bridge_state_shape")
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@READY "):
            self._fail("ready", line)
        self.ready = json.loads(line[7:])
        if (self.ready.get("rock_count") != count
                or self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", *self.ids]):
            self.close(kill=True)
            raise RuntimeError("restored_bridge_ready_contract")
        values = (
            list(mechanism_qpos) + rock_qpos.reshape(-1).tolist()
            + list(mechanism_qvel) + rock_qvel.reshape(-1).tolist()
        )
        operation = "init"
        if controller_state is not None:
            target = np.asarray(
                controller_state.get("controller_target"), dtype=np.float64)
            velocity = np.asarray(
                controller_state.get("controller_target_velocity"),
                dtype=np.float64)
            integral = np.asarray(
                controller_state.get("bucket_integral_effort"),
                dtype=np.float64)
            if (target.shape != (4,) or velocity.shape != (4,)
                    or integral.shape != (4,)
                    or not np.isfinite(target).all()
                    or not np.isfinite(velocity).all()
                    or not np.isfinite(integral).all()
                    or np.any(integral[:3] != 0.0)):
                self.close(kill=True)
                raise ValueError("restored_bridge_controller_state")
            operation = "init_loaded"
            values += target.tolist() + velocity.tolist() + [float(integral[3])]
        self._write(operation + " " + " ".join(
            f"{float(value):.17g}" for value in values) + "\n")
        self.state = self._read_state("restored_init")


def restore_runtime(
        args, config: dict, root: Path, records: list[dict],
        arrays: dict[str, np.ndarray], work: Path, label: str,
        binary: Path | None = None) -> tuple[MultiRockFixture, RestoredRuntimeBridge, dict]:
    restored_records = records_at_qpos(records, arrays["rigid_qpos"])
    capacity_config = root / config["capacity_bridge"]["config_path"]
    model, binary, model_manifest, bridge_manifest, compile_command = (
        build_model_and_bridge(
            args, root, capacity_config, restored_records, work, label, binary))
    default_bridge = RuntimeBridge(binary, model, len(records))
    default_state = default_bridge.state
    default_bridge.close()
    bridge = RestoredRuntimeBridge(
        binary, model, len(records),
        default_state["mechanism_qpos"], default_state["mechanism_qvel"],
        arrays["rigid_qpos"], arrays["rigid_qvel"])
    # The fixed Newton grid is a function of the particle positions supplied
    # when the fixture is constructed.  Recreate it from the saved allocation
    # seed, then overwrite the evolving material-point state below.
    fixture = MultiRockFixture(
        config, root, records, arrays["grid_reference_particle_q"], args.device)
    restore_report = restore_fixture_prefix(fixture, arrays)
    fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    fixture.assign(
        [pose[:3] for pose in arrays["rigid_qpos"]],
        [velocity[:3] for velocity in arrays["rigid_qvel"]],
        [velocity[3:] for velocity in arrays["rigid_qvel"]],
        [pose[3:7].tolist() for pose in arrays["rigid_qpos"]])
    solver_restore_report = restore_solver_continuation(fixture, arrays)
    qpos_error = float(np.max(np.abs(
        np.asarray([rock["qpos"] for rock in bridge.state["rocks"]])
        - arrays["rigid_qpos"])))
    qvel_error = float(np.max(np.abs(
        np.asarray([rock["qvel"] for rock in bridge.state["rocks"]])
        - arrays["rigid_qvel"])))
    restore_report.update({
        "solver_continuation": solver_restore_report,
        "rigid_qpos_max_abs_readback_error": qpos_error,
        "rigid_qvel_max_abs_readback_error": qvel_error,
        "rigid_qpos_readback_within_tolerance": qpos_error <= 1e-12,
        "rigid_qvel_readback_within_tolerance": qvel_error <= 1e-12,
    })
    if (not all(restore_report["field_readback_exact"].values())
            or not all(solver_restore_report["grid_readback_exact"].values())
            or not all(solver_restore_report[
                "solver_field_readback_exact"].values())
            or not restore_report["rigid_qpos_readback_within_tolerance"]
            or not restore_report["rigid_qvel_readback_within_tolerance"]):
        bridge.close(kill=True)
        raise RuntimeError("checkpoint_rigid_restore_readback")
    return fixture, bridge, {
        "restore": restore_report,
        "model_manifest": model_manifest,
        "bridge_manifest": bridge_manifest,
        "compile_command": compile_command,
        "binary": str(binary),
    }


def dynamic_step(fixture, bridge, records: list[dict], config: dict,
                 frame: int) -> tuple[dict, dict]:
    dt = float(config["probe"]["output_dt_s"])
    gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
    before = bridge.state
    fixture.assign(
        [np.asarray(rock["position"], dtype=np.float64)
         for rock in before["rocks"]],
        [predict_free_linear_velocity(
            np.asarray(rock["qvel"][:3], dtype=np.float64), gravity, dt)
         for rock in before["rocks"]],
        [np.asarray(rock["angular_velocity_world_rad_s"], dtype=np.float64)
         for rock in before["rocks"]],
        [rock["qpos"][3:7] for rock in before["rocks"]])
    origins = [np.asarray(rock["position"], dtype=np.float64)
               for rock in before["rocks"]]
    measured, particles = fixture.step(dt, origins)
    after = bridge.step(
        dt, int(config["probe"]["mujoco_internal_substeps"]),
        {"bucket": _zero(before["bucket_origin"]), **measured})
    delta = sum(float(record["mass_kg"]) * (
        np.asarray(current["qvel"][:3]) - np.asarray(previous["qvel"][:3]))
        for record, previous, current in zip(
            records, before["rocks"], after["rocks"], strict=True))
    expected = dt * sum(
        np.asarray(measured[previous["rock_id"]]["force"])
        + float(record["mass_kg"]) * gravity
        for record, previous in zip(records, before["rocks"], strict=True))
    residual = float(np.linalg.norm(delta - expected))
    linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                 for rock in after["rocks"])
    angular = max(float(np.linalg.norm(rock["qvel"][3:]))
                  for rock in after["rocks"])
    position_delta = max(float(np.linalg.norm(
        np.asarray(current["qpos"][:3]) - np.asarray(previous["qpos"][:3])))
        for previous, current in zip(
            before["rocks"], after["rocks"], strict=True))
    contacts = after.get("contact_substeps", {})
    external = sum(int(contacts.get(name, 0)) for name in (
        "rock_ground", "rock_bucket", "rock_ramp", "rock_truck"))
    finite = bool(particles["finite"] and _finite_tree(after))
    stable = bool(
        linear <= float(config["probe"]["maximum_stable_rock_linear_speed_m_s"])
        and angular <= float(
            config["probe"]["maximum_stable_rock_angular_speed_rad_s"])
        and _stable_particle(particles, config["probe"]))
    return after, {
        "frame": frame,
        "finite": finite,
        "stable": stable,
        "maximum_rock_linear_speed_m_s": linear,
        "maximum_rock_angular_speed_rad_s": angular,
        "maximum_rock_position_step_m": position_delta,
        "global_momentum_residual_ns": residual,
        "external_rigid_support_substeps": external,
        "rock_rock_contact_substeps": int(contacts.get("rock_rock", 0)),
        "forbidden_contact_count": int(after.get("forbidden_contact_count", 0)),
        "contact_substeps": contacts,
        **particles,
    }


def run_continuity_hold(fixture, bridge, records: list[dict], config: dict,
                        frames: int) -> dict:
    trace = []
    for frame in range(frames):
        _, row = dynamic_step(fixture, bridge, records, config, frame)
        trace.append(row)
        if not row["finite"]:
            break
    executed = len(trace)
    dt = float(config["probe"]["output_dt_s"])
    maximum_one_step_position = (
        float(config["probe"]["maximum_stable_rock_linear_speed_m_s"])
        * dt * 1.01)
    checks = {
        "all_requested_frames_executed": executed == frames,
        "all_states_finite": bool(trace) and all(row["finite"] for row in trace),
        "first_step_stable": bool(trace) and trace[0]["stable"],
        "all_frames_stable": bool(trace) and all(row["stable"] for row in trace),
        "first_step_position_continuous": bool(trace) and
            trace[0]["maximum_rock_position_step_m"] <= maximum_one_step_position,
        "no_external_rigid_support": bool(trace) and all(
            row["external_rigid_support_substeps"] == 0 for row in trace),
        "forbidden_contacts_zero": bool(trace) and all(
            row["forbidden_contact_count"] == 0 for row in trace),
        "momentum_accounted_once": bool(trace) and max(
            row["global_momentum_residual_ns"] for row in trace) < 1e-4,
    }
    return {
        "requested_frames": frames,
        "executed_frames": executed,
        "checks": checks,
        "maximum_one_step_position_m": maximum_one_step_position,
        "maximum_rock_linear_speed_m_s": max((
            row["maximum_rock_linear_speed_m_s"] for row in trace), default=None),
        "maximum_rock_angular_speed_rad_s": max((
            row["maximum_rock_angular_speed_rad_s"] for row in trace), default=None),
        "maximum_particle_speed_m_s": max((
            row["particle_max_speed_m_s"] for row in trace), default=None),
        "maximum_global_momentum_residual_ns": max((
            row["global_momentum_residual_ns"] for row in trace), default=None),
        "trace": trace,
    }
