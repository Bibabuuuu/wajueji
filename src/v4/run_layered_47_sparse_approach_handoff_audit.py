#!/usr/bin/env python3
"""Independent new-process restore audit of the sparse approach handoff."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

from layered_47_optimized_bridge import optimized_bridge
import layered_47_rigid_frequency_candidate as rigid_frequency
import layered_47_sparse_approach_handoff_checkpoint as handoff
from layered_excavation_approach_planner import (
    _load_kinematics,
    _proxy_local_vertices,
    _terrain_clearance,
    build_fused_terrain,
)
from planning.rear_lower_prepoint import (
    MECHANISM_CLEARANCE_M,
    TRUCK_CLEARANCE_M,
    ground_completed_terrain,
    mechanism_and_truck_clearance,
    proxy_surface_samples,
    sampled_surface_clearance,
)
import run_layered_47_sparse_approach_handoff as approach
import run_layered_47_sparse_motion as motion


HZ = 60
RIGID_INTERNAL_HZ = 4800
AUDIT_FRAMES = 60
MINIMUM_RESTORE_CLEARANCE_M = 0.35
LOWERED_PREPOINT_MINIMUM_RESTORE_CLEARANCE_M = 0.32


def minimum_restore_clearance(target_tip_height_above_surface_m: float) -> float:
    if math.isclose(target_tip_height_above_surface_m, 0.60,
                    rel_tol=0.0, abs_tol=1.0e-12):
        return MINIMUM_RESTORE_CLEARANCE_M
    if math.isclose(target_tip_height_above_surface_m, 0.55,
                    rel_tol=0.0, abs_tol=1.0e-12):
        return LOWERED_PREPOINT_MINIMUM_RESTORE_CLEARANCE_M
    raise ValueError("sparse_handoff_audit_unsupported_target_height")


def audit_frame_passes(
        row: dict, minimum_clearance_m: float =
        MINIMUM_RESTORE_CLEARANCE_M,
        route: str = approach.LEGACY_TOP_ROUTE) -> bool:
    strict = bool(row["strict"])
    if (route == approach.REAR_LOWER_ROUTE and not strict
            and row.get("stable", False)
            and row.get("guard_failures")
                == ["bucket_mpm_contact_samples"]
            and not row.get("bucket_mpm_effective_contact", True)):
        strict = True
    base = bool(
        strict
        and row["joint_bounds_ok"]
        and row["bucket_terrain_clearance_m"]
            >= minimum_clearance_m
        and row["axis_tracking_error"]
            <= approach.MAXIMUM_ENDPOINT_AXIS_ERROR
        and row["closure_m"]
            <= approach.MAXIMUM_PHYSICAL_CLOSURE_M)
    if route == approach.LEGACY_TOP_ROUTE:
        return base
    if route == approach.REAR_LOWER_ROUTE:
        return bool(
            base
            and not row.get("bucket_mpm_effective_contact", False)
            and row["mechanism_terrain_clearance_m"]
                >= MECHANISM_CLEARANCE_M
            and row["static_truck_aabb_separation_m"]
                >= TRUCK_CLEARANCE_M)
    raise ValueError("sparse_handoff_audit_route")


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".progress.ndjson")
    if journal.exists():
        raise FileExistsError(journal)
    started = time.perf_counter()
    rows = []
    bridge = None
    report = {
        "schema": "layered_47_sparse_approach_handoff_restore_audit_v1",
        "status": "initializing",
        "hz": HZ,
        "rigid_internal_hz": RIGID_INTERNAL_HZ,
        "scope": {
            "contact_executed": False,
            "manual_collection_eligible": False,
            "training_eligible": False,
            "formal_a30_equivalent": False,
        },
        "protocol": {
            "audit_frames": AUDIT_FRAMES,
            "audit_seconds": AUDIT_FRAMES / HZ,
            "zero_command": True,
            "minimum_restore_clearance_m": None,
        },
        "runner_sha256": motion.probe.bench.sha256(Path(__file__)),
    }
    motion.probe.bench._write_json(output, report)

    def append(event, **values):
        motion.probe.bench._append_journal(
            journal, {"event": event, **values})

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    try:
        if args.rigid_internal_hz != RIGID_INTERNAL_HZ:
            raise ValueError("sparse_handoff_audit_requires_rigid_4800hz")
        root = Path(args.repository_root).resolve()
        config_path = Path(args.config).resolve()
        local_source_config = Path(args.local_source_config).resolve()
        candidate_report = Path(args.candidate_report).resolve()
        resume_report = Path(args.resume_report).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        metadata_path = Path(args.metadata).resolve()
        execution_path = Path(args.approach_execution).resolve()
        approach_report_path = Path(args.approach_report).resolve()
        approach_report = json.loads(
            approach_report_path.read_text(encoding="utf-8"))
        if (approach_report.get("schema") !=
                "layered_47_sparse_approach_handoff_run_v1"
                or approach_report.get("status") !=
                "sparse_approach_handoff_saved_unaudited"
                or not all(approach_report.get("checks", {}).values())):
            raise ValueError("sparse_handoff_audit_approach_report")
        route = approach_report["protocol"].get(
            "route", approach.LEGACY_TOP_ROUTE)
        if route not in approach.SUPPORTED_ROUTES:
            raise ValueError("sparse_handoff_audit_route")
        target_tip_height = approach_report["protocol"].get(
            "target_tip_height_above_surface_m")
        if route == approach.REAR_LOWER_ROUTE:
            minimum_clearance = float(approach_report["protocol"][
                "minimum_bucket_clearance_m"])
            clearance_basis = (
                "rear_lower_dense_open_bucket_proxy_with_known_ground_"
                "completion_and_the_execution_clearance_threshold")
        else:
            target_tip_height = float(target_tip_height)
            minimum_clearance = minimum_restore_clearance(target_tip_height)
            clearance_basis = (
                "lowered_55cm_prepoint_retains_at_least_32mm_geometric_"
                "clearance_gap_above_the_measured_first_contact_clearance"
                if math.isclose(target_tip_height, 0.55,
                                rel_tol=0.0, abs_tol=1.0e-12) else
                "original_60cm_prepoint_threshold")
        report["protocol"].update(
            route=route,
            target_tip_height_above_surface_m=target_tip_height,
            minimum_restore_clearance_m=minimum_clearance,
            minimum_mechanism_clearance_m=(
                MECHANISM_CLEARANCE_M
                if route == approach.REAR_LOWER_ROUTE else None),
            minimum_static_truck_aabb_separation_m=(
                TRUCK_CLEARANCE_M
                if route == approach.REAR_LOWER_ROUTE else None),
            clearance_basis=clearance_basis)
        provenance = approach_report.get("handoff_provenance")
        arrays, metadata, before = handoff.load_checkpoint_readonly(
            checkpoint_path, metadata_path, provenance)
        if (approach_report["handoff_checkpoint"] != before
                or approach_report["handoff_metadata"] !=
                motion.probe.bench.checkpoint_identity(metadata_path)):
            raise ValueError("sparse_handoff_audit_saved_identity")
        if motion.probe.bench.sha256(execution_path) != provenance[
                "approach_execution_report_sha256"]:
            raise ValueError("sparse_handoff_audit_execution_report")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if (execution.get("schema") !=
                "layered_47_sparse_approach_execution_v1"
                or execution.get("status") !=
                "sparse_noncontact_approach_passed"
                or not all(execution.get("checks", {}).values())):
            raise ValueError("sparse_handoff_audit_execution_contract")

        loader, identity = motion.probe.select(47)
        base = loader(config_path, root)
        base["input_checkpoint"] = json.loads(
            local_source_config.read_text(encoding="utf-8"))[
                "input_checkpoint"]
        source_arrays, source, source_before = motion.audit.load_candidate(
            candidate_report, base, 47, HZ)
        motion.audit.validate_seed_audit(
            json.loads(resume_report.read_text(encoding="utf-8")),
            candidate_report, source, source_before, 47, HZ)
        if (source_before["sha256"] != provenance[
                    "source_checkpoint_sha256"]
                or motion.probe.bench.sha256(
                    Path(source_before["path"] + ".json")) != provenance[
                    "source_checkpoint_metadata_sha256"]
                or motion.probe.bench.sha256(candidate_report) != provenance[
                    "candidate_report_sha256"]
                or motion.probe.bench.sha256(resume_report) != provenance[
                    "resume_report_sha256"]
                or motion.probe.bench.sha256(config_path) != provenance[
                    "preflight_config_sha256"]
                or motion.probe.bench.sha256(local_source_config)
                != provenance["local_source_config_sha256"]):
            raise ValueError("sparse_handoff_audit_source_lineage")
        del source_arrays

        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity[
                "expected_ids"]:
            raise ValueError("sparse_handoff_audit_manifest_order")
        proxy_path, proxy = motion.probe.p47._load_bound_json(
            root, base["open_bucket_proxy"], "proxy")
        base["probe"] = {
            **base["probe"],
            "output_dt_s": 1.0 / HZ,
            "mujoco_internal_substeps": RIGID_INTERNAL_HZ // HZ,
        }
        terrain, observation = build_fused_terrain(
            root, records, arrays["particle_q"], arrays["rigid_qpos"])
        assembly, chain = _load_kinematics(
            args.kinematics_python_dir, args.kinematics_model,
            args.kinematics_mesh_dir)
        if list(chain.axes) != list(motion.AXES):
            raise ValueError("sparse_handoff_audit_kinematic_axes")
        truck_path = truck_world_vertices = None
        if route == approach.REAR_LOWER_ROUTE:
            if not args.truck_mesh:
                raise ValueError("sparse_handoff_audit_truck_mesh")
            truck_path = Path(args.truck_mesh).resolve()
            if (not truck_path.is_file()
                    or motion.probe.bench.sha256(truck_path) !=
                    "bd379fc56bd492b4e0a342f8432869f2926a6e1f10a4face407d70c5784fa640"):
                raise ValueError("sparse_handoff_audit_truck_mesh_binding")
            with np.load(truck_path, allow_pickle=False) as truck_archive:
                truck_world_vertices = truck_archive[
                    "world_vertices"].copy()

        binding_paths = [
            checkpoint_path, metadata_path, execution_path,
            approach_report_path, candidate_report, resume_report,
            config_path, local_source_config, manifest_path, proxy_path,
            Path(args.kinematics_model),
            *[Path(args.kinematics_python_dir) / name for name in (
                "scene_assembly.py", "fast_kinematics.py",
                "closed_chain.py")],
        ]
        if truck_path is not None:
            binding_paths.append(truck_path)
        bindings = {str(path): motion.probe.bench.sha256(path)
                    for path in binding_paths}
        report.update(
            handoff_checkpoint=before,
            handoff_metadata=motion.probe.bench.checkpoint_identity(
                metadata_path),
            approach_report=str(approach_report_path),
            approach_execution=str(execution_path),
            source_checkpoint=source_before,
            solver_profile=approach_report["solver_profile"],
            observation=observation,
            input_bindings=bindings,
        )
        append("validated_inputs", handoff_checkpoint=before)

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with optimized_bridge(args.mechanism_header_dir,
                              output.parent / "audit-build") as optimization, \
                tempfile.TemporaryDirectory(
                    prefix="sparse-approach-audit-") as raw:
            fixture, held, restored = motion.audit.restore_candidate_runtime(
                args, base, root, records, arrays, proxy,
                approach_report["solver_profile"], Path(raw))
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            bridge.command = np.zeros(4)
            report["restore"] = restored
            report["optimization"] = optimization.copy()
            local_proxy = _proxy_local_vertices(proxy)
            local_surface_samples = (
                proxy_surface_samples(proxy)
                if route == approach.REAR_LOWER_ROUTE else None)
            collision_terrain = (
                ground_completed_terrain(terrain)
                if route == approach.REAR_LOWER_ROUTE else None)
            sweep_geometries, _ = assembly.assemble(
                np.asarray(bridge.state["mechanism_qpos"],
                           dtype=np.float64))
            predicted_pose = assembly.assemble(
                np.asarray(bridge.state["mechanism_qpos"]),
                include_meshes=False)[1]["bucket"]
            predicted_vertices = (
                local_proxy @ predicted_pose[:3, :3].T
                + predicted_pose[:3, 3])
            actual_vertices = motion.probe.p47._bucket_world_vertices(
                proxy, bridge.state)
            geometry_exact = bool(
                predicted_vertices.shape == actual_vertices.shape
                and np.allclose(predicted_vertices, actual_vertices,
                                rtol=0.0, atol=1.0e-7))
            if not geometry_exact:
                raise ValueError("sparse_handoff_audit_prediction_geometry")
            saved_qpos = arrays["mechanism_qpos"].copy()
            for frame in range(AUDIT_FRAMES):
                row, _ = motion.probe.bench._advance_frame(
                    fixture, bridge, base, frame + 1)
                mechanism_qpos = np.asarray(
                    bridge.state["mechanism_qpos"], dtype=np.float64)
                rock_positions = np.asarray([
                    rock["qpos"][:3] for rock in bridge.state["rocks"]],
                    dtype=np.float64)
                bucket_vertices = motion.probe.p47._bucket_world_vertices(
                    proxy, bridge.state)
                mechanism_clearance = None
                truck_clearance = None
                if route == approach.REAR_LOWER_ROUTE:
                    rotation = np.asarray(
                        bridge.state["bucket_rotation"],
                        dtype=np.float64).reshape(3, 3)
                    dense_bucket_world = (
                        local_surface_samples @ rotation.T
                        + np.asarray(bridge.state["bucket_origin"],
                                     dtype=np.float64))
                    bucket_clearance = sampled_surface_clearance(
                        dense_bucket_world,
                        collision_terrain)["clearance_m"]
                    _, bodies = assembly.assemble(
                        mechanism_qpos, include_meshes=False)
                    sweep = mechanism_and_truck_clearance(
                        sweep_geometries, bodies, terrain,
                        truck_world_vertices)
                    mechanism_clearance = sweep[
                        "mechanism_terrain_clearance_m"]
                    truck_clearance = sweep[
                        "static_truck_aabb_separation_m"]
                else:
                    bucket_clearance = _terrain_clearance(
                        bucket_vertices, terrain, require_known=True)
                row.update(
                    phase="independent_restore_hold",
                    command_si=[0.0] * 4,
                    axis_positions=mechanism_qpos[motion.AXES].tolist(),
                    axis_velocity=np.asarray(
                        bridge.state["mechanism_qvel"],
                        dtype=np.float64)[motion.AXES - 1].tolist(),
                    axis_tracking_error=float(np.max(np.abs(
                        mechanism_qpos[motion.AXES]
                        - saved_qpos[motion.AXES]))),
                    bucket_terrain_clearance_m=float(bucket_clearance),
                    mechanism_terrain_clearance_m=mechanism_clearance,
                    static_truck_aabb_separation_m=truck_clearance,
                    bucket_linear_speed_m_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_linear_velocity_world_m_s"])),
                    bucket_angular_speed_rad_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_angular_velocity_world_rad_s"])),
                    rock_displacement_from_source_m=float(np.linalg.norm(
                        rock_positions - arrays["rigid_qpos"][:, :3],
                        axis=1).max()),
                    joint_bounds_ok=approach._joint_bounds_ok(
                        mechanism_qpos, chain),
                )
                row["finite"] = bool(
                    row["finite"]
                    and motion.probe.bench._finite_tree(bridge.state))
                if row["bucket_mpm_contact_samples"] > 0:
                    row["bucket_contact_diagnostics"] = (
                        approach._bucket_contact_diagnostics(fixture))
                row["bucket_mpm_effective_contact"] = False
                motion.probe.classify_frame(
                    row, "measured", 0, transition="frequency")
                row["audit_guard"] = audit_frame_passes(
                    row, minimum_clearance, route)
                row["stopped"] = motion.stopped(row)
                rows.append(row)
                append("audit_frame", row=row)
                if frame == 0 or (frame + 1) % 15 == 0 \
                        or not row["audit_guard"]:
                    print(
                        f"SPARSE_HANDOFF_AUDIT frame={frame + 1}/"
                        f"{AUDIT_FRAMES} safe={row['audit_guard']} "
                        f"clearance={row['bucket_terrain_clearance_m']:.9g}",
                        flush=True)
                if not row["audit_guard"]:
                    report["rejection"] = {
                        "reason": "audit_frame_guard", "row": row}
                    break
            checks = {
                "restore_exact": restored["exact"],
                "prediction_geometry_exact": geometry_exact,
                "all_frames_executed": len(rows) == AUDIT_FRAMES,
                "all_frames_noncontact_safe": bool(rows) and all(
                    row["audit_guard"] for row in rows),
                "all_mechanism_clearances_safe": (
                    bool(rows) and all(
                        row["mechanism_terrain_clearance_m"]
                            >= MECHANISM_CLEARANCE_M for row in rows)
                    if route == approach.REAR_LOWER_ROUTE else True),
                "all_static_truck_separations_safe": (
                    bool(rows) and all(
                        row["static_truck_aabb_separation_m"]
                            >= TRUCK_CLEARANCE_M for row in rows)
                    if route == approach.REAR_LOWER_ROUTE else True),
                "zero_effective_bucket_soil_contact_all_frames": (
                    bool(rows) and all(
                        not row["bucket_mpm_effective_contact"]
                        for row in rows)),
                "zero_bucket_rock_contact_all_frames": bool(rows) and all(
                    row["rock_bucket_contact_substeps"] == 0
                    for row in rows),
                "first_step_continuous": bool(rows) and rows[0][
                    "maximum_rock_position_step_m"]
                    <= 0.01 / HZ * 1.01,
                "endpoint_stopped_tail": len(rows) >= 15 and all(
                    row["stopped"] for row in rows[-15:]),
                "handoff_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        checkpoint_path) == before,
                "handoff_metadata_unchanged":
                    motion.probe.bench.checkpoint_identity(metadata_path)
                    == report["handoff_metadata"],
                "source_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        source_before["path"]) == source_before,
                "input_bindings_unchanged": all(
                    motion.probe.bench.sha256(Path(path)) == sha256
                    for path, sha256 in bindings.items()),
            }
            report["checks"] = checks
            passed = all(checks.values())
            report["status"] = (
                "sparse_approach_handoff_restore_audit_passed"
                if passed else
                "sparse_approach_handoff_restore_audit_rejected")
            report["contact_handoff_eligible"] = passed
            report["next_gate"] = (
                "review_short_contact_handoff_protocol_no_collection"
                if passed else
                "fix_restore_before_any_contact")
            bridge.close(kill=not passed)
            bridge = None
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="sparse_approach_handoff_restore_audit_failed",
            contact_handoff_eligible=False,
            next_gate="fix_restore_before_any_contact",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if bridge is not None:
            try:
                bridge.close(kill=True)
            except Exception as error:
                report["cleanup_error"] = str(error)
        report.update(
            trace=rows,
            timing={
                "frames": len(rows),
                "simulation_seconds": len(rows) / HZ,
                "physics_wall_seconds": sum(
                    row["wall_seconds"] for row in rows),
                "total_wall_seconds": time.perf_counter() - started,
            },
        )
        motion.probe.bench._write_json(output, clean(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in (
            "config", "repository-root", "output", "candidate-report",
            "resume-report", "local-source-config", "checkpoint",
            "metadata", "approach-execution", "approach-report", "device",
            "mujoco-include", "mujoco-library-dir", "mechanism-header-dir",
            "controller-header-dir", "kinematics-python-dir",
            "kinematics-model", "kinematics-mesh-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--truck-mesh")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--rigid-internal-hz", type=int, choices=(RIGID_INTERNAL_HZ,),
        default=RIGID_INTERNAL_HZ)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result.get(key) for key in (
        "status", "checks", "contact_handoff_eligible", "next_gate",
        "timing", "failure")}, indent=2))
    return 0 if result["status"] == (
        "sparse_approach_handoff_restore_audit_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
