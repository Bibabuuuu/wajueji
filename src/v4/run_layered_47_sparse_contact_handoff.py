#!/usr/bin/env python3
"""One-shot sparse CONTACT_HANDOFF diagnostic from the audited prepoint.

The route descends at one quarter of the normal command envelope, stops the
command after the first bucket/MPM contact frame, and then holds zero command
for a bounded tail.  It never curls, digs, saves a continuation checkpoint, or
grants collection/training eligibility.
"""

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
    BUCKET_MOUTH_WIDTH_M,
    BUCKET_PATH_CLEARANCE_M,
    COMMAND_LIMITS_M_S,
    _load_kinematics,
    _proxy_local_vertices,
    _terrain_clearance,
    build_fused_terrain,
    plan_noncontact_approach,
)
from planning.cycle import validate_cut
import run_layered_47_sparse_approach_handoff as approach
import run_layered_47_sparse_motion as motion


HZ = 60
RIGID_INTERNAL_HZ = 4800
COMMAND_SCALE = 0.25
CONTACT_COMMAND_LIMITS_M_S = COMMAND_LIMITS_M_S * COMMAND_SCALE
TARGET_TIP_HEIGHT_ABOVE_SURFACE_M = 0.40
MAXIMUM_PATH_FRAMES = 900
POST_CONTACT_HOLD_FRAMES = 30
STOPPED_TAIL_FRAMES = 15
MAXIMUM_DISPATCH_SURFACE_CHANGE_M = 0.01
MAXIMUM_PLANNED_CLOSURE_M = 1.0e-4
MAXIMUM_PHYSICAL_CLOSURE_M = 1.0e-3
MAXIMUM_AXIS_TRACKING_ERROR = 0.025
MAXIMUM_BUCKET_REFERENCE_VERTEX_ERROR_M = 0.001
MINIMUM_NONZERO_FORCE_N = 1.0e-9


class ContactAbort(RuntimeError):
    """A protective guard stopped the one-shot experiment."""


def audit_report_passes(document: dict, checkpoint: dict,
                        metadata: dict) -> bool:
    return bool(
        document.get("schema") ==
            "layered_47_sparse_approach_handoff_restore_audit_v1"
        and document.get("status") ==
            "sparse_approach_handoff_restore_audit_passed"
        and document.get("contact_handoff_eligible") is True
        and document.get("handoff_checkpoint") == checkpoint
        and document.get("handoff_metadata") == metadata
        and bool(document.get("checks"))
        and all(document["checks"].values()))


def wrench_summary(wrench: dict) -> dict:
    force = np.asarray(wrench.get("force"), dtype=np.float64)
    torque = np.asarray(wrench.get("torque"), dtype=np.float64)
    point = np.asarray(wrench.get("point"), dtype=np.float64)
    finite = bool(
        force.shape == (3,) and torque.shape == (3,) and point.shape == (3,)
        and np.isfinite(force).all() and np.isfinite(torque).all()
        and np.isfinite(point).all())
    return {
        "bucket_mpm_force_world_n": force.tolist(),
        "bucket_mpm_torque_world_nm": torque.tolist(),
        "bucket_mpm_wrench_point_world_m": point.tolist(),
        "bucket_mpm_force_norm_n": (
            float(np.linalg.norm(force)) if force.shape == (3,)
            else math.inf),
        "bucket_mpm_torque_norm_nm": (
            float(np.linalg.norm(torque)) if torque.shape == (3,)
            else math.inf),
        "bucket_mpm_wrench_finite": finite,
    }


def contact_guard_checks(row: dict) -> dict:
    """Retain every diagnostic abort except the now-expected soil contact."""
    checks = motion.probe.guard_checks(row, "measured")
    checks.pop("bucket_mpm_contact_samples")
    checks.update({
        "joint_bounds_ok": bool(row["joint_bounds_ok"]),
        "bucket_clearance": bool(
            np.isfinite(row["bucket_terrain_clearance_m"])
            and row["bucket_terrain_clearance_m"]
                >= BUCKET_PATH_CLEARANCE_M),
        "axis_tracking": bool(
            np.isfinite(row["axis_tracking_error"])
            and row["axis_tracking_error"]
                <= MAXIMUM_AXIS_TRACKING_ERROR),
        "bucket_reference_vertex_tracking": bool(
            np.isfinite(row["bucket_reference_vertex_error_m"])
            and row["bucket_reference_vertex_error_m"]
                <= MAXIMUM_BUCKET_REFERENCE_VERTEX_ERROR_M),
        "physical_closure": bool(
            np.isfinite(row["closure_m"])
            and row["closure_m"] <= MAXIMUM_PHYSICAL_CLOSURE_M),
        "bucket_wrench_finite": bool(row["bucket_mpm_wrench_finite"]),
        "command_within_contact_envelope": bool(
            np.asarray(row["command_si"], dtype=np.float64).shape == (4,)
            and np.isfinite(row["command_si"]).all()
            and np.all(np.abs(row["command_si"])
                       <= CONTACT_COMMAND_LIMITS_M_S + 1.0e-15)),
    })
    return checks


def precontact_frame_passes(row: dict) -> bool:
    return bool(
        row["strict"]
        and row["bucket_mpm_contact_samples"] == 0
        and all(contact_guard_checks(row).values()))


def first_contact_frame_passes(row: dict) -> bool:
    return bool(
        row["bucket_mpm_contact_samples"] > 0
        and row["bucket_mpm_force_norm_n"] > MINIMUM_NONZERO_FORCE_N
        and all(contact_guard_checks(row).values()))


def _plan(terrain, selection, qpos, proxy, args):
    return plan_noncontact_approach(
        terrain=terrain,
        selection=selection,
        operator_qpos=np.asarray(qpos, dtype=np.float64).tolist(),
        open_bucket_proxy=proxy,
        kinematics_python_directory=args.kinematics_python_dir,
        kinematics_model_path=args.kinematics_model,
        kinematics_mesh_directory=args.kinematics_mesh_dir,
        output_dt_s=1.0 / HZ,
        maximum_path_frames=MAXIMUM_PATH_FRAMES,
        maximum_planned_closure_m=MAXIMUM_PLANNED_CLOSURE_M,
        validation_substeps=2,
        command_speed_limits=CONTACT_COMMAND_LIMITS_M_S,
        target_tip_height_above_surface_m=
            TARGET_TIP_HEIGHT_ABOVE_SURFACE_M,
    )


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    journal = output / "progress.ndjson"
    started = time.perf_counter()
    rows: list[dict] = []
    bridge = fixture = None
    report = {
        "schema": "layered_47_sparse_contact_handoff_v1",
        "status": "initializing",
        "hz": HZ,
        "rigid_internal_hz": RIGID_INTERNAL_HZ,
        "device": args.device,
        "scope": {
            "single_first_contact_smoke": True,
            "digging_executed": False,
            "bucket_curl_executed": False,
            "manual_collection_eligible": False,
            "training_eligible": False,
            "formal_a30_equivalent": False,
            "checkpoint_written": False,
        },
        "protocol": {
            "solver_grid_type": "sparse",
            "solver_voxel_size_m": 0.1,
            "output_dt_s": 1.0 / HZ,
            "rigid_internal_hz": RIGID_INTERNAL_HZ,
            "command_scale_of_normal": COMMAND_SCALE,
            "command_speed_limits_si": CONTACT_COMMAND_LIMITS_M_S.tolist(),
            "target_tip_height_above_surface_m":
                TARGET_TIP_HEIGHT_ABOVE_SURFACE_M,
            "maximum_path_frames": MAXIMUM_PATH_FRAMES,
            "first_contact_response":
                "next_output_frame_zero_command_then_bounded_hold",
            "post_contact_hold_frames": POST_CONTACT_HOLD_FRAMES,
            "post_contact_hold_seconds": POST_CONTACT_HOLD_FRAMES / HZ,
            "stopped_tail_frames": STOPPED_TAIL_FRAMES,
            "minimum_geometric_clearance_m": BUCKET_PATH_CLEARANCE_M,
            "maximum_physical_closure_m": MAXIMUM_PHYSICAL_CLOSURE_M,
            "maximum_bucket_reference_vertex_error_m":
                MAXIMUM_BUCKET_REFERENCE_VERTEX_ERROR_M,
            "rock_bucket_contact_allowed": False,
            "force_policy":
                "finite_and_nonzero_at_first_soil_contact;_motion_guards_bound_response",
            "continuous_collision_proven": False,
        },
        "runner_sha256": motion.probe.bench.sha256(Path(__file__)),
    }
    motion.probe.bench._write_json(report_path, report)

    def append(event, **values):
        motion.probe.bench._append_journal(
            journal, {"event": event, **values})

    try:
        if args.rigid_internal_hz != RIGID_INTERNAL_HZ:
            raise ValueError("sparse_contact_requires_rigid_4800hz")
        root = Path(args.repository_root).resolve()
        config_path = Path(args.config).resolve()
        local_source_config = Path(args.local_source_config).resolve()
        candidate_report = Path(args.candidate_report).resolve()
        resume_report = Path(args.resume_report).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        metadata_path = Path(args.metadata).resolve()
        execution_path = Path(args.approach_execution).resolve()
        approach_report_path = Path(args.approach_report).resolve()
        audit_path = Path(args.restore_audit).resolve()

        approach_report = json.loads(
            approach_report_path.read_text(encoding="utf-8"))
        if (approach_report.get("schema") !=
                "layered_47_sparse_approach_handoff_run_v1"
                or approach_report.get("status") !=
                "sparse_approach_handoff_saved_unaudited"
                or not all(approach_report.get("checks", {}).values())):
            raise ValueError("sparse_contact_approach_report")
        provenance = approach_report.get("handoff_provenance")
        arrays, _, checkpoint_before = handoff.load_checkpoint_readonly(
            checkpoint_path, metadata_path, provenance)
        metadata_before = motion.probe.bench.checkpoint_identity(metadata_path)
        if (approach_report["handoff_checkpoint"] != checkpoint_before
                or approach_report["handoff_metadata"] != metadata_before):
            raise ValueError("sparse_contact_handoff_identity")

        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if (motion.probe.bench.sha256(execution_path) != provenance[
                    "approach_execution_report_sha256"]
                or execution.get("schema") !=
                    "layered_47_sparse_approach_execution_v1"
                or execution.get("status") !=
                    "sparse_noncontact_approach_passed"
                or not all(execution.get("checks", {}).values())):
            raise ValueError("sparse_contact_execution_report")
        selection = execution["observation"]["selection"]
        audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit_report_passes(
                audit_document, checkpoint_before, metadata_before):
            raise ValueError("sparse_contact_restore_audit")

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
        del source_arrays
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
                or motion.probe.bench.sha256(local_source_config) != provenance[
                    "local_source_config_sha256"]):
            raise ValueError("sparse_contact_source_lineage")

        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity[
                "expected_ids"]:
            raise ValueError("sparse_contact_manifest_order")
        proxy_path, proxy = motion.probe.p47._load_bound_json(
            root, base["open_bucket_proxy"], "proxy")
        base["probe"] = {
            **base["probe"],
            "output_dt_s": 1.0 / HZ,
            "mujoco_internal_substeps": RIGID_INTERNAL_HZ // HZ,
        }
        terrain, handoff_observation = build_fused_terrain(
            root, records, arrays["particle_q"], arrays["rigid_qpos"])
        initial_validation = validate_cut(
            terrain, selection, bucket_width=BUCKET_MOUTH_WIDTH_M,
            maximum_surface_change_m=MAXIMUM_DISPATCH_SURFACE_CHANGE_M)
        initial_plan = _plan(
            terrain, selection, arrays["mechanism_qpos"], proxy, args)
        initial_plan_summary = approach._plan_summary(initial_plan)
        assembly, chain = _load_kinematics(
            args.kinematics_python_dir, args.kinematics_model,
            args.kinematics_mesh_dir)
        if list(chain.axes) != list(motion.AXES):
            raise ValueError("sparse_contact_kinematic_axes")

        binding_paths = [
            checkpoint_path, metadata_path, execution_path,
            approach_report_path, audit_path, candidate_report,
            resume_report, config_path, local_source_config, manifest_path,
            proxy_path, Path(args.kinematics_model), Path(__file__),
            Path(approach.__file__),
            Path(plan_noncontact_approach.__code__.co_filename),
            *[Path(args.kinematics_python_dir) / name for name in (
                "scene_assembly.py", "fast_kinematics.py",
                "closed_chain.py")],
        ]
        bindings = {str(path): motion.probe.bench.sha256(path)
                    for path in binding_paths}
        report.update(
            handoff_checkpoint=checkpoint_before,
            handoff_metadata=metadata_before,
            restore_audit={
                "path": str(audit_path),
                "sha256": motion.probe.bench.sha256(audit_path),
                "status": audit_document["status"],
            },
            source_checkpoint=source_before,
            solver_profile=approach_report["solver_profile"],
            acquisition_domain=handoff.SOURCE_DOMAIN,
            selection=selection,
            handoff_observation=handoff_observation,
            initial_validation=initial_validation,
            initial_plan=initial_plan_summary,
            input_bindings=bindings,
        )
        append("validated_inputs", handoff_checkpoint=checkpoint_before,
               restore_audit=report["restore_audit"],
               initial_validation=initial_validation,
               initial_plan=initial_plan_summary)

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with optimized_bridge(args.mechanism_header_dir,
                              output / "build") as optimization, \
                tempfile.TemporaryDirectory(
                    prefix="sparse-contact-handoff-") as raw:
            fixture, held, restored = motion.audit.restore_candidate_runtime(
                args, base, root, records, arrays, proxy,
                approach_report["solver_profile"], Path(raw))
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            report["restore"] = restored
            report["optimization"] = optimization.copy()

            local_proxy = _proxy_local_vertices(proxy)
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
                raise ValueError("sparse_contact_prediction_geometry")

            captured: dict[str, dict] = {}
            original_fixture_step = fixture.step

            def recording_fixture_step(dt, origins):
                measured, particles = original_fixture_step(dt, origins)
                captured["wrenches"] = measured
                return measured, particles

            fixture.step = recording_fixture_step

            def advance(command, phase, reference_qpos, active_terrain):
                bridge.command = np.asarray(command, dtype=np.float64)
                row, _ = motion.probe.bench._advance_frame(
                    fixture, bridge, base, len(rows) + 1)
                mechanism_qpos = np.asarray(
                    bridge.state["mechanism_qpos"], dtype=np.float64)
                rock_positions = np.asarray([
                    rock["qpos"][:3] for rock in bridge.state["rocks"]],
                    dtype=np.float64)
                bucket_vertices = motion.probe.p47._bucket_world_vertices(
                    proxy, bridge.state)
                reference_pose = assembly.assemble(
                    np.asarray(reference_qpos, dtype=np.float64),
                    include_meshes=False)[1]["bucket"]
                reference_vertices = (
                    local_proxy @ reference_pose[:3, :3].T
                    + reference_pose[:3, 3])
                row.update(
                    phase=phase,
                    command_si=bridge.command.tolist(),
                    axis_positions=mechanism_qpos[motion.AXES].tolist(),
                    axis_velocity=np.asarray(
                        bridge.state["mechanism_qvel"],
                        dtype=np.float64)[motion.AXES - 1].tolist(),
                    axis_tracking_error=float(np.max(np.abs(
                        mechanism_qpos[motion.AXES]
                        - np.asarray(reference_qpos)[motion.AXES]))),
                    bucket_terrain_clearance_m=float(_terrain_clearance(
                        bucket_vertices, active_terrain,
                        require_known=False)),
                    bucket_reference_vertex_error_m=float(np.max(
                        np.linalg.norm(
                            bucket_vertices - reference_vertices, axis=1))),
                    bucket_linear_speed_m_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_linear_velocity_world_m_s"])),
                    bucket_angular_speed_rad_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_angular_velocity_world_rad_s"])),
                    rock_displacement_from_source_m=float(np.linalg.norm(
                        rock_positions - arrays["rigid_qpos"][:, :3],
                        axis=1).max()),
                    rock_displacement_from_handoff_m=float(np.linalg.norm(
                        rock_positions - arrays["rigid_qpos"][:, :3],
                        axis=1).max()),
                    joint_bounds_ok=approach._joint_bounds_ok(
                        mechanism_qpos, chain),
                    elapsed_simulation_seconds=(len(rows) + 1) / HZ,
                    **wrench_summary(captured["wrenches"]["bucket"]),
                )
                row["finite"] = bool(
                    row["finite"]
                    and motion.probe.bench._finite_tree(bridge.state)
                    and row["bucket_mpm_wrench_finite"])
                motion.probe.classify_frame(
                    row, "measured", 0, transition="frequency")
                row["contact_guard_checks"] = contact_guard_checks(row)
                row["contact_guard"] = all(
                    row["contact_guard_checks"].values())
                row["precontact_guard"] = precontact_frame_passes(row)
                row["first_contact_guard"] = first_contact_frame_passes(row)
                row["stopped"] = motion.stopped(row)
                rows.append(row)
                append("frame", row=row)
                if (len(rows) == 1 or len(rows) % 30 == 0
                        or row["bucket_mpm_contact_samples"] > 0
                        or not row["contact_guard"]):
                    print(
                        f"SPARSE_CONTACT frame={len(rows)} phase={phase} "
                        f"samples={row['bucket_mpm_contact_samples']} "
                        f"force={row['bucket_mpm_force_norm_n']:.9g} "
                        f"hard_safe={row['contact_guard']} "
                        f"clearance={row['bucket_terrain_clearance_m']:.9g}",
                        flush=True)
                return row

            dispatch = advance(
                np.zeros(4), "dispatch_reobservation",
                arrays["mechanism_qpos"], terrain)
            dispatch["first_step_continuous"] = bool(
                dispatch["maximum_rock_position_step_m"]
                <= 0.01 / HZ * 1.01)
            if (not dispatch["first_step_continuous"]
                    or not dispatch["precontact_guard"]):
                bridge.command = np.zeros(4)
                raise ContactAbort("sparse_contact_dispatch_guard")

            current_rock_qpos = np.asarray([
                rock["qpos"] for rock in bridge.state["rocks"]],
                dtype=np.float64)
            dispatch_terrain, dispatch_observation = build_fused_terrain(
                root, records, fixture.state.particle_q.numpy(),
                current_rock_qpos)
            dispatch_validation = validate_cut(
                dispatch_terrain, selection,
                bucket_width=BUCKET_MOUTH_WIDTH_M,
                maximum_surface_change_m=
                    MAXIMUM_DISPATCH_SURFACE_CHANGE_M)
            dispatch_plan = _plan(
                dispatch_terrain, selection,
                bridge.state["mechanism_qpos"], proxy, args)
            dispatch_plan_summary = approach._plan_summary(dispatch_plan)
            report.update(
                dispatch_observation=dispatch_observation,
                dispatch_validation=dispatch_validation,
                dispatch_plan=dispatch_plan_summary,
                prediction_geometry_exact=geometry_exact,
            )
            append("dispatch_validated",
                   validation=dispatch_validation,
                   observation=dispatch_observation,
                   plan=dispatch_plan_summary)

            planned_qpos = np.asarray(
                dispatch_plan["qpos"], dtype=np.float64)
            planned_velocity = np.asarray(
                dispatch_plan["velocity"], dtype=np.float64)
            contact_row = None
            for index, command in enumerate(planned_velocity, start=1):
                row = advance(command, "contact_descent",
                              planned_qpos[index], dispatch_terrain)
                if row["bucket_mpm_contact_samples"] > 0:
                    bridge.command = np.zeros(4)
                    if not row["first_contact_guard"]:
                        report["rejection"] = {
                            "reason": "first_contact_guard", "row": row}
                        raise ContactAbort("sparse_contact_first_contact_guard")
                    contact_row = row
                    break
                if not row["precontact_guard"]:
                    bridge.command = np.zeros(4)
                    report["rejection"] = {
                        "reason": "precontact_guard", "row": row}
                    raise ContactAbort("sparse_contact_precontact_guard")
            if contact_row is None:
                bridge.command = np.zeros(4)
                raise ContactAbort("sparse_contact_not_observed_before_endpoint")

            contact_qpos = np.asarray(
                bridge.state["mechanism_qpos"], dtype=np.float64).copy()
            for _ in range(POST_CONTACT_HOLD_FRAMES):
                row = advance(np.zeros(4), "post_contact_hold",
                              contact_qpos, dispatch_terrain)
                if not row["contact_guard"]:
                    bridge.command = np.zeros(4)
                    report["rejection"] = {
                        "reason": "post_contact_guard", "row": row}
                    raise ContactAbort("sparse_contact_post_contact_guard")

            descent_rows = [row for row in rows
                            if row["phase"] == "contact_descent"]
            hold_rows = [row for row in rows
                         if row["phase"] == "post_contact_hold"]
            commanded_contact_rows = [row for row in descent_rows
                                      if row["bucket_mpm_contact_samples"] > 0]
            checks = {
                "handoff_restore_exact": restored["exact"],
                "independent_restore_audit_bound_and_passed":
                    audit_report_passes(
                        audit_document, checkpoint_before, metadata_before),
                "prediction_geometry_exact": geometry_exact,
                "dispatch_reobservation_passed": bool(
                    dispatch_validation["valid"]),
                "first_step_continuous": dispatch[
                    "first_step_continuous"],
                "all_frames_hard_safe": all(
                    row["contact_guard"] for row in rows),
                "all_precontact_frames_strict": all(
                    row["precontact_guard"]
                    for row in rows
                    if row["phase"] in (
                        "dispatch_reobservation", "contact_descent")
                    and row["bucket_mpm_contact_samples"] == 0),
                "first_soil_contact_observed": contact_row[
                    "bucket_mpm_contact_samples"] > 0,
                "first_contact_force_finite_nonzero": bool(
                    contact_row["bucket_mpm_wrench_finite"]
                    and contact_row["bucket_mpm_force_norm_n"]
                        > MINIMUM_NONZERO_FORCE_N),
                "only_one_commanded_contact_frame":
                    len(commanded_contact_rows) == 1,
                "descent_stopped_at_first_contact": bool(
                    descent_rows and descent_rows[-1] is contact_row),
                "zero_rock_bucket_contact_all_frames": all(
                    row["rock_bucket_contact_substeps"] == 0
                    for row in rows),
                "all_post_contact_hold_frames_executed":
                    len(hold_rows) == POST_CONTACT_HOLD_FRAMES,
                "zero_command_on_first_post_contact_frame": bool(
                    hold_rows and hold_rows[0]["command_si"] == [0.0] * 4),
                "zero_command_throughout_post_contact_hold": bool(
                    hold_rows and all(
                        row["command_si"] == [0.0] * 4
                        for row in hold_rows)),
                "post_contact_stopped_tail": bool(
                    len(hold_rows) >= STOPPED_TAIL_FRAMES
                    and all(row["stopped"]
                            for row in hold_rows[-STOPPED_TAIL_FRAMES:])),
                "handoff_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        checkpoint_path) == checkpoint_before,
                "handoff_metadata_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        metadata_path) == metadata_before,
                "source_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        source_before["path"]) == source_before,
                "input_bindings_unchanged": all(
                    motion.probe.bench.sha256(Path(path)) == sha256
                    for path, sha256 in bindings.items()),
                "no_checkpoint_written": True,
                "collection_and_training_not_granted": bool(
                    report["scope"]["manual_collection_eligible"] is False
                    and report["scope"]["training_eligible"] is False),
            }
            passed = all(checks.values())
            report.update(
                checks=checks,
                contact={
                    "trace_frame": contact_row["frame"],
                    "descent_frame": len(descent_rows),
                    "samples": contact_row["bucket_mpm_contact_samples"],
                    "force_world_n":
                        contact_row["bucket_mpm_force_world_n"],
                    "force_norm_n":
                        contact_row["bucket_mpm_force_norm_n"],
                    "torque_world_nm":
                        contact_row["bucket_mpm_torque_world_nm"],
                    "torque_norm_nm":
                        contact_row["bucket_mpm_torque_norm_nm"],
                    "geometric_clearance_m":
                        contact_row["bucket_terrain_clearance_m"],
                },
                status=("sparse_contact_handoff_passed" if passed
                        else "sparse_contact_handoff_rejected"),
                next_gate=(
                    "review_contact_trace_before_any_scoop_or_collection"
                    if passed else
                    "inspect_failed_contact_acceptance_before_retry"),
            )
            bridge.close(kill=not passed)
            bridge = None
    except ContactAbort as error:
        report.update(
            status="sparse_contact_handoff_aborted",
            next_gate="inspect_protective_abort_before_retry",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="sparse_contact_handoff_failed",
            next_gate="fix_contact_handoff_implementation_before_retry",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if bridge is not None:
            try:
                bridge.command = np.zeros(4)
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
        motion.probe.bench._write_json(report_path, _clean(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in (
            "config", "repository-root", "output", "candidate-report",
            "resume-report", "local-source-config", "checkpoint",
            "metadata", "approach-execution", "approach-report",
            "restore-audit", "device", "mujoco-include",
            "mujoco-library-dir", "mechanism-header-dir",
            "controller-header-dir", "kinematics-python-dir",
            "kinematics-model", "kinematics-mesh-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--rigid-internal-hz", type=int, choices=(RIGID_INTERNAL_HZ,),
        default=RIGID_INTERNAL_HZ)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result.get(key) for key in (
        "status", "checks", "contact", "next_gate", "timing",
        "failure")}, indent=2))
    return 0 if result["status"] == "sparse_contact_handoff_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
