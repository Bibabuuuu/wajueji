#!/usr/bin/env python3
"""Execute the audited 47-rock non-contact bucket approach.

The source checkpoint remains read-only.  The program selects one entry strip
from the restored particle/rock surface, plans a fully sampled free-space
path, re-observes the strip after one zero-command physics frame, and then
drives the real four-axis MuJoCo velocity interface while Newton MPM and all
rigid rocks remain live.  The endpoint is still above the material; no digging
or checkpoint write occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_mixed_material_capacity_preflight import sha256
from layered_excavation_approach_planner import (
    BUCKET_MOUTH_WIDTH_M,
    BUCKET_PATH_CLEARANCE_M,
    COMMAND_LIMITS_M_S,
    GRID_BOUNDS_XY_M,
    GRID_RESOLUTION_M,
    MECHANISM_PATH_CLEARANCE_M,
    TARGET_BUCKET_PITCH_RAD,
    TARGET_TIP_HEIGHT_ABOVE_SURFACE_M,
    _terrain_clearance,
    build_fused_terrain,
    plan_noncontact_approach,
)
from layered_mound_checkpoint import (
    checkpoint_identity,
    load_checkpoint_readonly,
    restore_fixture_prefix,
    transfer_solver_warmstart,
)
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    restore_runtime,
)
from mpm_rigid_free_velocity import predict_free_linear_velocity
from planning.cycle import validate_cut
from run_layered_47_excavation_preflight import (
    DEFAULT_PROFILE as PREFLIGHT_PROFILE,
    ExcavationFixture,
    _bucket_transform,
    _bucket_world_vertices,
    _finite_tree,
    _load_bound_json,
)
import run_layered_47_excavation_preflight as preflight
from run_mixed_material_r3_a_probe import _make_transform, _write_json


APPROACH_CONFIG_SCHEMA = "excavation_layered_noncontact_approach_config_v1"
REPORT_SCHEMA = "excavation_layered_47_excavation_approach_report_v1"
AXES = np.asarray([7, 9, 12, 15], dtype=np.int64)
PROFILE = {
    **PREFLIGHT_PROFILE,
    "temporary_directory_prefix": "layered-47-excavation-approach-",
    "restored_runtime_label": "approach-source-restored-47",
    "progress_label": "PROGRESS_47_APPROACH",
    "report_schema": REPORT_SCHEMA,
    "passed_status": "layered_47_excavation_approach_passed",
    "failed_status": "layered_47_excavation_approach_failed",
    "passed_next_gate": "contact_handoff_from_verified_noncontact_prepoint",
    "failed_next_gate": "fix_noncontact_approach",
}


def load_approach_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema": APPROACH_CONFIG_SCHEMA,
        "status": "program_select_and_execute_noncontact_approach",
        "observation": {
            "grid_bounds_xy_m": list(GRID_BOUNDS_XY_M),
            "grid_resolution_m": GRID_RESOLUTION_M,
            "bucket_mouth_width_m": BUCKET_MOUTH_WIDTH_M,
            "maximum_dispatch_surface_change_m": 0.01,
        },
        "path": {
            "target_tip_height_above_surface_m":
                TARGET_TIP_HEIGHT_ABOVE_SURFACE_M,
            "target_bucket_pitch_rad": TARGET_BUCKET_PITCH_RAD,
            "minimum_bucket_proxy_clearance_m": BUCKET_PATH_CLEARANCE_M,
            "minimum_other_mechanism_clearance_m":
                MECHANISM_PATH_CLEARANCE_M,
            "command_limits": COMMAND_LIMITS_M_S.tolist(),
            "output_dt_s": 1.0 / 120.0,
            "maximum_path_frames": 2400,
            "endpoint_hold_frames": 60,
            "maximum_endpoint_axis_error": 0.015,
            "maximum_path_axis_tracking_error": 0.025,
            "maximum_closure_m": 0.001,
            "maximum_endpoint_bucket_linear_speed_m_s": 0.02,
            "maximum_endpoint_bucket_angular_speed_rad_s": 0.02,
        },
        "acceptance": {
            "program_selected_current_observation": True,
            "dispatch_reobservation_required": True,
            "zero_bucket_soil_contact_all_frames": True,
            "zero_bucket_rock_contact_all_frames": True,
            "source_checkpoint_mutated": False,
            "writes_checkpoint": False,
            "digging_executed": False,
        },
    }
    if config != expected:
        raise ValueError("excavation_approach_config_contract")
    return config


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _bridge_step(bridge: RestoredRuntimeBridge, dt: float, substeps: int,
                 command: np.ndarray, wrenches: dict[str, dict]) -> dict:
    velocity = np.asarray(command, dtype=np.float64)
    if velocity.shape != (4,) or not np.isfinite(velocity).all():
        raise ValueError("excavation_approach_command_shape")
    if np.any(np.abs(velocity) > COMMAND_LIMITS_M_S + 1.0e-10):
        raise ValueError("excavation_approach_command_limit")
    fields = [
        "step", f"{float(bridge.state['time_s']):.17g}", f"{dt:.17g}",
        str(substeps), *[f"{float(value):.17g}" for value in velocity],
        str(bridge.count + 1),
    ]
    for body_name in ["bucket", *bridge.ids]:
        wrench = wrenches[body_name]
        fields.append(body_name)
        for key in ("force", "torque", "point"):
            fields.extend(f"{float(value):.17g}" for value in wrench[key])
    bridge._write(" ".join(fields) + "\n")
    bridge.state = bridge._read_state("command_step")
    return bridge.state


def _physical_step(
        fixture: ExcavationFixture,
        bridge: RestoredRuntimeBridge,
        config: dict,
        command: np.ndarray,
        frame: int,
        reference_qpos: np.ndarray | None,
        proxy: dict,
        terrain: dict) -> tuple[dict, dict]:
    dt = float(config["probe"]["output_dt_s"])
    prior = bridge.state
    gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
    rock_origins = {
        rock["rock_id"]: np.asarray(rock["position"], dtype=np.float64)
        for rock in prior["rocks"]}
    origins = {
        **rock_origins,
        "bucket": np.asarray(prior["bucket_origin"], dtype=np.float64),
    }
    transforms = [
        _make_transform(rock["qpos"][:3], rock["qpos"][3:7])
        for rock in prior["rocks"]] + [_bucket_transform(prior)]
    velocities = [wp.spatial_vector(
        *[float(value) for value in predict_free_linear_velocity(
            np.asarray(rock["qvel"][:3], dtype=np.float64), gravity, dt)],
        *[float(value) for value in
          rock["angular_velocity_world_rad_s"]])
        for rock in prior["rocks"]]
    velocities.append(wp.spatial_vector(
        *[float(value) for value in
          prior["bucket_linear_velocity_world_m_s"]],
        *[float(value) for value in
          prior["bucket_angular_velocity_world_rad_s"]]))
    fixture.state.body_q.assign(wp.array(
        transforms, dtype=wp.transform, device=fixture.device))
    fixture.state.body_qd.assign(wp.array(
        velocities, dtype=wp.spatial_vector, device=fixture.device))
    measured, particles = fixture.step(dt, origins)
    after = _bridge_step(
        bridge, dt, int(config["probe"]["mujoco_internal_substeps"]),
        command, measured)
    rock_linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                      for rock in after["rocks"])
    rock_angular = max(float(np.linalg.norm(rock["qvel"][3:]))
                       for rock in after["rocks"])
    contacts = after.get("contact_substeps", {})
    mechanism_qpos = np.asarray(after["mechanism_qpos"], dtype=np.float64)
    tracking_error = (None if reference_qpos is None else float(np.max(
        np.abs(mechanism_qpos[AXES] - reference_qpos[AXES]))))
    bucket_vertices = _bucket_world_vertices(proxy, after)
    clearance = _terrain_clearance(
        bucket_vertices, terrain, require_known=True)
    stable = bool(
        rock_linear <= float(config["probe"][
            "maximum_stable_rock_linear_speed_m_s"])
        and rock_angular <= float(config["probe"][
            "maximum_stable_rock_angular_speed_rad_s"])
        and particles["particle_max_speed_m_s"] <= float(config["probe"][
            "maximum_stable_particle_speed_m_s"])
        and particles["particle_p99_speed_m_s"] <= float(config["probe"][
            "maximum_stable_particle_p99_speed_m_s"]))
    row = {
        "frame": frame,
        "finite": bool(particles["finite"] and _finite_tree(after)),
        "stable": stable,
        "command": np.asarray(command, dtype=np.float64).tolist(),
        "axis_tracking_error": tracking_error,
        "bucket_terrain_clearance_m": clearance,
        "maximum_rock_linear_speed_m_s": rock_linear,
        "maximum_rock_angular_speed_rad_s": rock_angular,
        "bucket_mpm_contact_samples": int(
            particles["contact_samples_by_body"]["bucket"]),
        "rock_bucket_contact_substeps": int(
            contacts.get("rock_bucket", 0)),
        "external_rigid_support_substeps": sum(int(
            contacts.get(name, 0)) for name in (
                "rock_ground", "rock_ramp", "rock_truck")),
        "forbidden_contact_count": int(
            after.get("forbidden_contact_count", 0)),
        "closure_m": float(after.get("closure_m", math.inf)),
        "bucket_linear_speed_m_s": float(np.linalg.norm(
            after["bucket_linear_velocity_world_m_s"])),
        "bucket_angular_speed_rad_s": float(np.linalg.norm(
            after["bucket_angular_velocity_world_rad_s"])),
        **particles,
    }
    return after, row


def _restore_exact(restored: dict) -> bool:
    restore = restored["restore"]
    solver = restored.get(
        "solver_restore", restore.get("solver_continuation"))
    return bool(
        all(restore["field_readback_exact"].values())
        and restore["rigid_qpos_readback_within_tolerance"]
        and restore["rigid_qvel_readback_within_tolerance"]
        and all(solver["grid_readback_exact"].values())
        and all(solver["solver_field_readback_exact"].values()))


def run(args, *, config_loader=preflight.load_config,
        profile: dict | None = None) -> dict:
    profile = PROFILE if profile is None else profile
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    preflight_config_path = Path(args.preflight_config).resolve()
    approach_config_path = Path(args.approach_config).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = Path(args.checkpoint_metadata).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = config_loader(preflight_config_path, root)
    approach_config = load_approach_config(approach_config_path)
    arrays, source_metadata, before = load_checkpoint_readonly(
        checkpoint, metadata_path, config["input_checkpoint"])
    manifest_path = root / config["physical_geometry"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in profile["layers"]]
    if ([rock["rock_id"] for rock in records] != profile["expected_ids"]):
        raise ValueError("excavation_approach_manifest_order")
    _, operator = _load_bound_json(
        root, config["operator_start"], "approach_operator_start")
    proxy_path, proxy = _load_bound_json(
        root, config["open_bucket_proxy"], "approach_open_bucket_proxy")

    terrain, observation = build_fused_terrain(
        root, records, arrays["particle_q"], arrays["rigid_qpos"])
    path = plan_noncontact_approach(
        terrain=terrain,
        selection=observation["selection"],
        operator_qpos=operator["qpos"],
        open_bucket_proxy=proxy,
        kinematics_python_directory=args.kinematics_python_dir,
        kinematics_model_path=args.kinematics_model,
        kinematics_mesh_directory=args.kinematics_mesh_dir,
    )
    planned_qpos = np.asarray(path["qpos"], dtype=np.float64)
    planned_velocity = np.asarray(path["velocity"], dtype=np.float64)
    if (len(planned_velocity) > int(approach_config["path"][
            "maximum_path_frames"])
            or planned_qpos.shape != (len(planned_velocity) + 1, 19)):
        raise ValueError("excavation_approach_path_shape")

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    source_bridge = None
    target_bridge = None
    rows = []
    dispatch_validation = None
    restored = None
    particle_restore = None
    warmstart_transfer = None
    fixture = None
    dispatch_terrain = terrain
    try:
        with tempfile.TemporaryDirectory(
                prefix=profile["temporary_directory_prefix"]) as raw:
            work = Path(raw)
            source_fixture, source_bridge, restored = restore_runtime(
                args, config, root, records, arrays, work,
                profile["restored_runtime_label"])
            source_model = Path(restored["model_manifest"]["model_path"]) if (
                "model_path" in restored["model_manifest"]) else (
                    work / f"{profile['restored_runtime_label']}.xml")
            binary = Path(restored["binary"])
            source_bridge.close()
            source_bridge = None
            target_bridge = RestoredRuntimeBridge(
                binary, source_model, len(records),
                [float(value) for value in operator["qpos"]], [0.0] * 18,
                arrays["rigid_qpos"], arrays["rigid_qvel"])
            fixture = ExcavationFixture(
                config, root, records, arrays["grid_reference_particle_q"],
                proxy, target_bridge.state, args.device)
            particle_restore = restore_fixture_prefix(fixture, arrays)
            fixture.solver.notify_model_changed(
                newton.ModelFlags.MODEL_PROPERTIES)
            warmstart_transfer = transfer_solver_warmstart(
                source_fixture, fixture)
            fixture.assign(target_bridge.state)

            # A real zero-command frame is the dispatch boundary.  Re-observe
            # the current particle and rigid-rock state before any motion.
            after, row = _physical_step(
                fixture, target_bridge, config, np.zeros(4), 0,
                planned_qpos[0], proxy, terrain)
            row["phase"] = "dispatch_reobservation"
            rows.append(row)
            current_rock_qpos = np.asarray(
                [rock["qpos"] for rock in after["rocks"]], dtype=np.float64)
            dispatch_terrain, dispatch_observation = build_fused_terrain(
                root, records, fixture.state.particle_q.numpy(),
                current_rock_qpos)
            dispatch_validation = validate_cut(
                dispatch_terrain, observation["selection"],
                bucket_width=BUCKET_MOUTH_WIDTH_M,
                maximum_surface_change_m=float(approach_config[
                    "observation"]["maximum_dispatch_surface_change_m"]))
            observation["dispatch_terrain"] = dispatch_observation["terrain"]

            for index, command in enumerate(planned_velocity, start=1):
                after, row = _physical_step(
                    fixture, target_bridge, config, command, index,
                    planned_qpos[index], proxy, dispatch_terrain)
                row["phase"] = "approach"
                rows.append(row)
                if index == 1 or index % 60 == 0 or index == len(
                        planned_velocity):
                    print(
                        f"{profile['progress_label']} "
                        f"frame={index}/{len(planned_velocity)} "
                        f"tracking={row['axis_tracking_error']:.9g} "
                        f"clearance={row['bucket_terrain_clearance_m']:.9g} "
                        f"bucket_mpm={row['bucket_mpm_contact_samples']} "
                        f"rock_bucket={row['rock_bucket_contact_substeps']}",
                        flush=True)
                if (not row["finite"] or row["axis_tracking_error"] > float(
                        approach_config["path"][
                            "maximum_path_axis_tracking_error"])):
                    break

            approach_executed = sum(row["phase"] == "approach" for row in rows)
            if approach_executed == len(planned_velocity):
                hold_frames = int(approach_config["path"][
                    "endpoint_hold_frames"])
                for hold in range(hold_frames):
                    frame = len(planned_velocity) + 1 + hold
                    after, row = _physical_step(
                        fixture, target_bridge, config, np.zeros(4), frame,
                        planned_qpos[-1], proxy, dispatch_terrain)
                    row["phase"] = "endpoint_hold"
                    rows.append(row)
                    if not row["finite"]:
                        break
            target_bridge.close(
                kill=not rows or not rows[-1]["finite"])
            target_bridge = None
    finally:
        if source_bridge is not None:
            source_bridge.close(kill=True)
        if target_bridge is not None:
            target_bridge.close(kill=True)

    after_identity = checkpoint_identity(checkpoint)
    approach_rows = [row for row in rows if row["phase"] == "approach"]
    hold_rows = [row for row in rows if row["phase"] == "endpoint_hold"]
    all_finite = bool(rows) and all(row["finite"] for row in rows)
    all_stable = bool(rows) and all(row["stable"] for row in rows)
    zero_soil = bool(rows) and all(
        row["bucket_mpm_contact_samples"] == 0 for row in rows)
    zero_rock = bool(rows) and all(
        row["rock_bucket_contact_substeps"] == 0 for row in rows)
    maximum_tracking = max((
        row["axis_tracking_error"] for row in rows
        if row["axis_tracking_error"] is not None), default=math.inf)
    minimum_clearance = min((
        row["bucket_terrain_clearance_m"] for row in rows
        if row["bucket_terrain_clearance_m"] is not None), default=-math.inf)
    endpoint = rows[-1] if rows else {}
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] ==
            config["input_checkpoint"]["sha256"],
        "source_checkpoint_metadata_verified": source_metadata[
            "checkpoint_sha256"] == before["sha256"],
        "source_checkpoint_unchanged": after_identity == before,
        "source_restore_exact": restored is not None and
            _restore_exact(restored),
        "expanded_particle_fields_restored_exactly": (
            particle_restore is not None and all(
                particle_restore["field_readback_exact"].values())),
        "grid_warmstart_transferred_for_bucket_expansion": (
            warmstart_transfer is not None
            and warmstart_transfer["body_q_prev_transferred"] is False
            and warmstart_transfer["collider_velocity_mode"] == "forward"),
        "program_selected_current_observation": observation[
            "selection"]["source"] == "fresh_observed_strip",
        "dispatch_reobservation_passed": bool(
            dispatch_validation and dispatch_validation["valid"]),
        "all_planned_approach_frames_executed": len(approach_rows) == len(
            planned_velocity),
        "all_endpoint_hold_frames_executed": len(hold_rows) == int(
            approach_config["path"]["endpoint_hold_frames"]),
        "all_states_finite": all_finite,
        "all_frames_stable": all_stable,
        "path_axis_tracking_within_tolerance": maximum_tracking <= float(
            approach_config["path"]["maximum_path_axis_tracking_error"]),
        "endpoint_axis_error_within_tolerance": bool(rows) and float(
            endpoint.get("axis_tracking_error", math.inf)) <= float(
                approach_config["path"]["maximum_endpoint_axis_error"]),
        "bucket_clearance_preserved": minimum_clearance >= float(
            approach_config["path"]["minimum_bucket_proxy_clearance_m"]),
        "zero_bucket_soil_contact_all_frames": zero_soil,
        "zero_bucket_rock_contact_all_frames": zero_rock,
        "no_external_rigid_support": bool(rows) and all(
            row["external_rigid_support_substeps"] == 0 for row in rows),
        "no_unmapped_or_nonfinite_collider_samples": bool(rows) and all(
            row["unmapped_active_collider_sample_count"] == 0
            and row["nonfinite_collider_sample_count"] == 0 for row in rows),
        "forbidden_contacts_zero": bool(rows) and all(
            row["forbidden_contact_count"] == 0 for row in rows),
        "mechanism_closure_within_one_mm": bool(rows) and all(
            row["closure_m"] <= float(approach_config["path"][
                "maximum_closure_m"]) for row in rows),
        "endpoint_bucket_stationary": bool(rows) and float(endpoint.get(
            "bucket_linear_speed_m_s", math.inf)) <= float(approach_config[
                "path"]["maximum_endpoint_bucket_linear_speed_m_s"])
            and float(endpoint.get(
                "bucket_angular_speed_rad_s", math.inf)) <= float(
                    approach_config["path"][
                        "maximum_endpoint_bucket_angular_speed_rad_s"]),
        "no_checkpoint_written": True,
        "digging_not_executed": True,
    }
    passed = all(checks.values())
    report = {
        "schema": profile["report_schema"],
        "status": (profile["passed_status"] if passed
                   else profile["failed_status"]),
        "checks": checks,
        "source_checkpoint": before,
        "source_checkpoint_after": after_identity,
        "source_metadata": source_metadata,
        "observation": observation,
        "dispatch_validation": dispatch_validation,
        "plan": {
            "start_qpos": planned_qpos[0].tolist(),
            "goal_qpos": planned_qpos[-1].tolist(),
            "target_tip_world_m": np.asarray(
                path["target_tip_world_m"]).tolist(),
            "target_pitch_rad": float(path["target_pitch_rad"]),
            "path_frames": int(path["steps"]),
            "duration_sim_s": float(path["duration_sim_s"]),
            "minimum_planned_bucket_terrain_clearance_m": float(
                path["minimum_bucket_terrain_clearance_m"]),
            "endpoint_planned_bucket_terrain_clearance_m": float(
                path["endpoint_bucket_terrain_clearance_m"]),
            "maximum_planned_closure_m": float(path["maximum_closure_m"]),
            "qpos_float64_sha256": _array_sha256(planned_qpos),
            "velocity_float64_sha256": _array_sha256(planned_velocity),
            "continuous_collision_proven": False,
            "physical_execution_completed": passed,
        },
        "execution_trace": rows,
        "metrics": {
            "maximum_axis_tracking_error": maximum_tracking,
            "minimum_actual_bucket_terrain_clearance_m": minimum_clearance,
            "approach_frames_executed": len(approach_rows),
            "endpoint_hold_frames_executed": len(hold_rows),
            "endpoint_axis_error": endpoint.get("axis_tracking_error"),
            "endpoint_bucket_linear_speed_m_s": endpoint.get(
                "bucket_linear_speed_m_s"),
            "endpoint_bucket_angular_speed_rad_s": endpoint.get(
                "bucket_angular_speed_rad_s"),
        },
        "restore": {
            "source": restored,
            "expanded_particle_restore": particle_restore,
            "expanded_warmstart": warmstart_transfer,
        },
        "inputs": {
            "preflight_config_sha256": sha256(preflight_config_path),
            "approach_config_sha256": sha256(approach_config_path),
            "manifest_sha256": sha256(manifest_path),
            "open_bucket_proxy_sha256": sha256(proxy_path),
            "kinematics_model_sha256": sha256(args.kinematics_model),
            "kinematics_python_files_sha256": {
                name: sha256(Path(args.kinematics_python_dir) / name)
                for name in ("scene_assembly.py", "fast_kinematics.py",
                             "closed_chain.py")},
        },
        "environment": {
            "python": platform.python_version(),
            "newton": newton.__version__, "warp": wp.__version__,
            "numpy": np.__version__, "device": str(wp.get_device()),
        },
        "execution": {
            "newton_solver_step_calls": len(rows),
            "mujoco_step_calls": len(rows),
            "wall_seconds": time.perf_counter() - started,
        },
        "scope": {
            **approach_config["acceptance"],
            "training": False,
        },
        "next_gate": (profile["passed_next_gate"] if passed
                      else profile["failed_next_gate"]),
    }
    if not _finite_tree(report):
        raise RuntimeError("excavation_approach_nonfinite_report")
    _write_json(output, report)
    return report


def _parser(default_device: str = "cuda:0") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-config", required=True)
    parser.add_argument("--approach-config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    parser.add_argument("--kinematics-python-dir", required=True)
    parser.add_argument("--kinematics-model", required=True)
    parser.add_argument("--kinematics-mesh-dir", required=True)
    return parser


def _summary(report: dict) -> dict:
    return {
        "status": report["status"], "checks": report["checks"],
        "source_checkpoint": report["source_checkpoint"],
        "selection": report["observation"]["selection"],
        "plan": report["plan"], "metrics": report["metrics"],
        "execution": report["execution"], "next_gate": report["next_gate"],
    }


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(_summary(report), sort_keys=True, allow_nan=False))
    return 0 if report["status"] == PROFILE["passed_status"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
