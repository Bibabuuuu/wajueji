#!/usr/bin/env python3
"""Protected manual single-scoop pilot from the audited sparse handoff.

The operator owns every excavation command.  This runner only restores the
audited prepoint, gates manual input, projects commands away from joint/end
stops, stops on hard physics faults, and records diagnostic evidence.  The
result is deliberately not a training episode until a separate post-hoc
review grants eligibility.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import signal
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

from contracts import ACTION_SPEED_LIMIT_SI
from layered_47_optimized_bridge import optimized_bridge
from layered_47_bucket_integral_candidate import bucket_integral_bridge
from layered_47_active_curl_candidate import active_curl_bridge
from legacy_push_shelf_cleanup import cleaned_manual_model
import layered_47_rigid_frequency_candidate as rigid_frequency
import layered_47_sparse_approach_handoff_checkpoint as handoff
import layered_47_loaded_handoff_checkpoint as loaded_handoff
import layered_47_truck_physics_candidate as truck_physics
from layered_excavation_approach_planner import (
    _load_kinematics,
    _proxy_local_vertices,
    build_fused_terrain,
    quaternion_matrix_wxyz,
    terrain_from_points,
)
from manual_action_input import (
    GamepadFourAxisState,
    KeyboardFourAxisState,
    load_manual_input_config,
)
import run_layered_47_interactive_proxy_practice as display_helpers
from run_layered_47_optimized_practice import InputGate
import run_layered_47_sparse_approach_handoff as approach
import run_layered_47_sparse_contact_handoff as contact
import run_layered_47_sparse_motion as motion
from drive_load_observation import AXIS_LABELS, AXIS_UNITS, observe_drive_load
from soil_surface import SoilSurface


HZ = 60
RIGID_INTERNAL_HZ = 4800
DECISION_OBSERVATION_STRIDE = 3
MAXIMUM_WALL_SECONDS = 900.0
MAXIMUM_SIMULATION_SECONDS = 30.0
MAXIMUM_FRAMES = int(HZ * MAXIMUM_SIMULATION_SECONDS)
FINAL_ZERO_HOLD_FRAMES = 30
STOPPED_TAIL_FRAMES = 15
MAXIMUM_PHYSICAL_CLOSURE_M = 1.0e-3
MAXIMUM_PARTICLE_SPEED_M_S = 5.0
MAXIMUM_ROCK_LINEAR_SPEED_M_S = 5.0
MAXIMUM_ROCK_ANGULAR_SPEED_RAD_S = 25.0
BUCKET_INTERNAL_UPPER_TARGET_M = 0.852
COMMAND_PREVIEW_SECONDS = 0.15
KINEMATIC_JOINT_MARGIN_M = 0.005
COMMAND_LIMITS_SI = np.asarray(ACTION_SPEED_LIMIT_SI, dtype=np.float64)
KINEMATIC_PREVIEW_COMMAND_LIMITS_SI = np.asarray(
    [0.12, 0.0225, 0.0225, 0.0225], dtype=np.float64)
ACQUISITION_DOMAIN = "local_47_rejected_ancestry_manual_contact_pilot_only"
HENCKY_CANDIDATE_SHA256 = (
    "7b58ede49369e28486b678dc77bd71d0a7b3003ccb9dbb0c3c19277b9944647a")

PHASE_MARKERS = {
    "1": "penetrate_deepen",
    "2": "curl_load",
    "3": "lift_clear",
    "4": "off_soil_confirm",
    "R": "uncertain_recovery",
}
TRANSPORT_PHASE_MARKERS = {
    "1": "lift_transport",
    "2": "align_receiver",
    "3": "unload_open",
    "4": "receiver_settle_return",
    "R": "uncertain_recovery",
}
EVENT_MARKERS = {
    "5": "jam_observed",
    "6": "spill_observed",
    "7": "load_detach_observed",
    "8": "clear_observed",
}


class ManualSessionBudget:
    """Bound each operator run segment without closing a paused session."""

    def __init__(self, now):
        self.last_time = now
        self.was_running = False
        self.active_seconds = 0.0
        self.first_frame = 0

    def poll(self, now, frames, paused):
        if self.was_running:
            self.active_seconds += max(0.0, now - self.last_time)
        self.last_time = now
        self.was_running = not paused
        reason = None
        if self.active_seconds >= MAXIMUM_WALL_SECONDS:
            reason = "wall_time_budget"
        elif frames - self.first_frame >= MAXIMUM_FRAMES:
            reason = "simulation_time_budget"
        if reason:
            self.active_seconds = 0.0
            self.first_frame = frames
            self.was_running = False
        return reason


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            _clean(value), sort_keys=True, separators=(",", ":"),
            allow_nan=False))
        stream.write("\n")
        stream.flush()


def selected_contact_target(execution: dict) -> dict:
    """Use the explicit rear-face marker when the staged route provides it."""
    target = execution.get("manual_contact_target")
    if target is None:
        target = execution["observation"]["selection"]
    entry = np.asarray(target.get("entry_xy_m"), dtype=np.float64)
    surface = float(target.get("surface_z_m", math.nan))
    if (entry.shape != (2,) or not np.isfinite(entry).all()
            or not math.isfinite(surface)):
        raise ValueError("manual_contact_target_contract")
    return target


class ContactCommandProjector:
    """Limit outward curl without blocking other axes on bucket soft overshoot."""

    def __init__(self, chain):
        self.chain = chain
        self.reason = None
        self.bucket_end_stop_projected = False

    def physical_bounds_ok(self, qpos) -> bool:
        q = np.asarray(qpos, dtype=np.float64)
        return bool(
            q.ndim == 1
            and q.size > max(*motion.AXES, *self.chain.limits)
            and np.isfinite(q).all()
            and abs(q[motion.AXES[0]]) <= 2.992
            and all(low <= q[index] <= high
                    for index, (low, high) in self.chain.limits.items()))

    def _advance_preview(self, command, dt):
        # The legacy geometry helper applies a 5 mm margin to *every*
        # joint, even held joints. Only for an already overshot bucket, let
        # its preview hold or retreat from the actual position. Never clamp
        # the pose or alter the physical bounds used by this projector.
        limits = self.chain.limits
        bucket = motion.AXES[3]
        low, high = limits[bucket]
        current = float(self.chain.q[bucket])
        try:
            if high - KINEMATIC_JOINT_MARGIN_M < current <= high:
                self.chain.limits = {
                    **limits,
                    bucket: (low, np.nextafter(
                        current + KINEMATIC_JOINT_MARGIN_M, np.inf)),
                }
            predicted, error = self.chain.advance(command, dt)
        finally:
            self.chain.limits = limits
        if (current > high - KINEMATIC_JOINT_MARGIN_M
                and predicted[bucket] > current + 1.0e-12):
            raise ValueError("bucket_outward_soft_overshoot")
        return predicted, error

    def project(self, qpos, requested):
        q = np.asarray(qpos, dtype=np.float64)
        command = np.asarray(requested, dtype=np.float64).copy()
        if (command.shape != (4,) or not np.isfinite(command).all()
                or np.any(np.abs(command) > COMMAND_LIMITS_SI + 1.0e-15)):
            raise ValueError("manual_contact_invalid_requested_command")
        self.reason = None
        self.bucket_end_stop_projected = False
        if not self.physical_bounds_ok(q):
            self.reason = "joint_working_range"
            return np.zeros(4, dtype=np.float64)
        # Bound the entire preview displacement, not just its starting pose.
        # A full curl request near the end stop must not cancel a valid lift.
        bucket_upper = min(
            BUCKET_INTERNAL_UPPER_TARGET_M,
            self.chain.limits[motion.AXES[3]][1] - KINEMATIC_JOINT_MARGIN_M)
        outward_limit = max(
            0.0, (bucket_upper - q[motion.AXES[3]]) / COMMAND_PREVIEW_SECONDS)
        if command[3] > outward_limit:
            command[3] = outward_limit
            self.reason = "bucket_internal_upper_target"
            self.bucket_end_stop_projected = True
        if not np.any(command):
            return command
        self.chain.q = q.copy()
        try:
            # The kinematic helper predates the 3x bucket command envelope.
            # Reparameterize the same 150 ms joint displacement at a lower
            # solver velocity; this preserves the geometric path and avoids
            # misclassifying a valid full-rate bucket command as a fault.
            scale = max(1.0, float(np.max(
                np.abs(command) / KINEMATIC_PREVIEW_COMMAND_LIMITS_SI)))
            preview_command = command / scale
            remaining = COMMAND_PREVIEW_SECONDS * scale
            while remaining > 1.0e-12:
                dt = min(0.05, remaining)
                predicted, _ = self._advance_preview(preview_command, dt)
                if not self.physical_bounds_ok(predicted):
                    raise ValueError("joint_working_range")
                remaining -= dt
        except ValueError as error:
            self.reason = str(error)
            return np.zeros(4, dtype=np.float64)
        return command


def hard_guard_checks(row: dict, *, allow_truck: bool = False) -> dict:
    """Irrecoverable contact/runtime guards; soil and rock contact are valid."""
    command = np.asarray(row["applied_command_si"], dtype=np.float64)
    rigid_contacts = row.get("rigid_contact_substeps")
    rigid_contact_breakdown_present = bool(
        isinstance(rigid_contacts, dict)
        and all(name in rigid_contacts for name in (
            "rock_ground", "rock_ramp", "rock_truck")))
    material = row.get("hencky_material_guard", {"enabled": False})
    sweep = row.get("moving_bucket_sweep", {"enabled": False})
    return {
        "finite": bool(row["finite"]),
        "forbidden_contact_zero": row["forbidden_contact_count"] == 0,
        # A displaced rock settling onto the real ground is valid during an
        # excavation episode. Ramp/truck contact remains a hard stop.
        "rigid_contact_breakdown_present":
            rigid_contact_breakdown_present,
        "rock_ramp_truck_contact_zero": bool(
            rigid_contact_breakdown_present
            and int(rigid_contacts["rock_ramp"]) == 0
            and (allow_truck or int(rigid_contacts["rock_truck"]) == 0)),
        "unmapped_active_collider_zero":
            row["unmapped_active_collider_sample_count"] == 0,
        "nonfinite_collider_sample_zero":
            row["nonfinite_collider_sample_count"] == 0,
        "physical_closure": bool(
            np.isfinite(row["closure_m"])
            and row["closure_m"] <= MAXIMUM_PHYSICAL_CLOSURE_M),
        "joint_bounds": bool(row["joint_bounds_ok"]),
        "bucket_wrench_finite": bool(row["bucket_mpm_wrench_finite"]),
        "moving_bucket_sweep_safe": bool(
            not sweep.get("enabled", False) or sweep.get("safe", False)),
        "hencky_material_state_safe": bool(
            not material.get("enabled", False) or material.get("safe", False)),
        "applied_command_bounded": bool(
            command.shape == (4,) and np.isfinite(command).all()
            and np.all(np.abs(command) <= COMMAND_LIMITS_SI + 1.0e-15)),
    }


def advisory_checks(row: dict) -> dict:
    """Uncalibrated velocity references recorded without stopping physics."""
    return {
        "particle_speed_reference_exceeded": bool(
            np.isfinite(row["particle_max_speed_m_s"])
            and row["particle_max_speed_m_s"]
                > MAXIMUM_PARTICLE_SPEED_M_S),
        "rock_linear_speed_reference_exceeded": bool(
            np.isfinite(row["maximum_rock_linear_speed_m_s"])
            and row["maximum_rock_linear_speed_m_s"]
                > MAXIMUM_ROCK_LINEAR_SPEED_M_S),
        "rock_angular_speed_reference_exceeded": bool(
            np.isfinite(row["maximum_rock_angular_speed_rad_s"])
            and row["maximum_rock_angular_speed_rad_s"]
                > MAXIMUM_ROCK_ANGULAR_SPEED_RAD_S),
    }


def latch_hard_fault(state: dict, row: dict, frame: int) -> bool:
    """Latch the first hard fault so later rendering cannot erase evidence."""
    if state["active"] or row["hard_guard"]:
        return False
    state.update(
        active=True,
        frame=int(frame),
        failed_checks=[
            name for name, passed in row["hard_guard_checks"].items()
            if not passed
        ],
    )
    return True


def compact_state(state: dict) -> dict:
    return {
        **({"bucket_active_curl_effort": list(state["bucket_active_curl_effort"])}
           if "bucket_active_curl_effort" in state else {}),
        "bucket_integral_effort": list(state.get(
            "bucket_integral_effort", [0.0] * 4)),
        "time_s": float(state["time_s"]),
        "mechanism_qpos": np.asarray(
            state["mechanism_qpos"], dtype=np.float64).tolist(),
        "mechanism_qvel": np.asarray(
            state["mechanism_qvel"], dtype=np.float64).tolist(),
        "controller_target": np.asarray(
            state["controller_target"], dtype=np.float64).tolist(),
        "controller_target_velocity": np.asarray(
            state["controller_target_velocity"], dtype=np.float64).tolist(),
        "bucket_origin_world_m": np.asarray(
            state["bucket_origin"], dtype=np.float64).tolist(),
        "bucket_linear_velocity_world_m_s": np.asarray(
            state["bucket_linear_velocity_world_m_s"],
            dtype=np.float64).tolist(),
        "bucket_angular_velocity_world_rad_s": np.asarray(
            state["bucket_angular_velocity_world_rad_s"],
            dtype=np.float64).tolist(),
        "rocks": [{
            "rock_id": rock["rock_id"],
            "qpos": np.asarray(rock["qpos"], dtype=np.float64).tolist(),
            "qvel": np.asarray(rock["qvel"], dtype=np.float64).tolist(),
        } for rock in state["rocks"]],
    }


def terrain_observation(
        fixture, state: dict, cached_rock_vertices: list[np.ndarray]) -> dict:
    particles = np.asarray(fixture.state.particle_q.numpy(), dtype=np.float64)
    rock_points = []
    for local, rock in zip(cached_rock_vertices, state["rocks"], strict=True):
        pose = np.asarray(rock["qpos"], dtype=np.float64)
        rock_points.append(
            local @ quaternion_matrix_wxyz(pose[3:7]).T + pose[:3])
    fused = terrain_from_points(np.concatenate(
        [particles, *rock_points], axis=0))
    height = np.asarray(fused["height_m"], dtype=np.float64)
    count = np.asarray(fused["point_count"], dtype=np.int64)
    return {
        "schema": "layered_47_manual_contact_terrain_observation_v1",
        "origin_xy_m": list(fused["origin_xy_m"]),
        "resolution_m": float(fused["resolution_m"]),
        "shape_yx": list(height.shape),
        "height_m": [[None if not np.isfinite(item) else float(item)
                      for item in row] for row in height],
        "point_count": count.tolist(),
        "known_cell_count": int(np.count_nonzero(count)),
    }


def draw_with_selected_cut(
        viewer, display_state, fixture, bridge_state, records,
        assembly, names, to_pose, device: str, display: dict,
        selected_cut: dict, frame: int, soil_surface: SoilSurface) -> None:
    """Draw the existing scene plus a bright green selected-entry column."""
    poses = assembly.assemble(
        np.asarray(bridge_state["mechanism_qpos"], dtype=np.float64),
        include_meshes=False)[1]
    if np.linalg.norm(np.asarray(bridge_state["bucket_origin"])
                      - poses["bucket"][:3, 3]) > 1.0e-7:
        raise RuntimeError("manual_contact_display_alignment")
    display_state.body_q.assign(wp.array(
        [to_pose(poses[name]) for name in names], dtype=wp.transform,
        device=device))
    viewer.begin_frame(float(bridge_state["time_s"]))
    viewer.log_state(display_state)
    update_token = frame // int(display["surface_update_interval_frames"])
    soil_surface.update(fixture.state.particle_q, update_token)
    soil_surface.draw(viewer, fixture.state.particle_q)
    for prototype in sorted({record["prototype_id"] for record in records}):
        indices = [index for index, record in enumerate(records)
                   if record["prototype_id"] == prototype]
        xforms = [display_helpers._make_transform(
            bridge_state["rocks"][index]["qpos"][:3],
            bridge_state["rocks"][index]["qpos"][3:7])
            for index in indices]
        scales = [wp.vec3(float(records[index]["equivalent_diameter_m"]))
                  for index in indices]
        viewer.log_instances(
            f"/practice/rocks/{prototype}",
            f"/practice/prototype/{prototype}",
            wp.array(xforms, dtype=wp.transform, device=device),
            wp.array(scales, dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec3(0.40, 0.37, 0.32),
                dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec4(0.92, 0.0, 0.0, 1.0),
                dtype=wp.vec4, device=device))
    entry_x, entry_y = selected_cut["entry_xy_m"]
    surface_z = float(selected_cut["surface_z_m"])
    marker = np.asarray([
        [entry_x, entry_y, surface_z + height]
        for height in np.linspace(0.05, 0.65, 9)], dtype=np.float32)
    viewer.log_points(
        "/manual_contact/selected_entry",
        wp.array(marker, dtype=wp.vec3, device=device),
        radii=0.035,
        colors=wp.full(
            len(marker), wp.vec3(0.10, 1.0, 0.10),
            dtype=wp.vec3, device=device))
    viewer.end_frame()


def _maximum_abs_row_vector(rows: list[dict], field: str) -> list[float]:
    if not rows:
        return [0.0] * 4
    values = np.asarray([
        row.get(field, [0.0] * 4) for row in rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("manual_contact_load_summary_shape")
    return np.max(np.abs(values), axis=0).tolist()


def summarize_episode(rows: list[dict], phase_events: list[dict],
                      event_annotations: list[dict]) -> dict:
    contact_rows = [row for row in rows
                    if row["bucket_mpm_contact_samples"] > 0]
    return {
        "frames": len(rows),
        "simulation_seconds": len(rows) / HZ,
        "soil_contact_observed": bool(contact_rows),
        "rock_bucket_contact_observed": any(
            row["rock_bucket_contact_substeps"] > 0 for row in rows),
        "maximum_bucket_mpm_contact_samples": max(
            (row["bucket_mpm_contact_samples"] for row in rows), default=0),
        "maximum_bucket_mpm_force_norm_n": max(
            (row["bucket_mpm_force_norm_n"] for row in rows), default=0.0),
        "maximum_bucket_mpm_torque_norm_nm": max(
            (row["bucket_mpm_torque_norm_nm"] for row in rows), default=0.0),
        "maximum_abs_equivalent_drive_load_by_axis": (
            _maximum_abs_row_vector(rows, "equivalent_drive_load")),
        "maximum_abs_drive_load_frame_delta_by_axis": (
            _maximum_abs_row_vector(
                rows, "equivalent_drive_load_delta_from_previous")),
        "drive_load_axis_labels": list(AXIS_LABELS),
        "drive_load_axis_units": list(AXIS_UNITS),
        "load_contact_thresholds_calibrated": False,
        "maximum_rock_bucket_contact_substeps": max(
            (row["rock_bucket_contact_substeps"] for row in rows), default=0),
        "maximum_rock_ground_contact_substeps": max((
            row.get("rigid_contact_substeps", {}).get("rock_ground", 0)
            for row in rows), default=0),
        "maximum_rock_ramp_contact_substeps": max((
            row.get("rigid_contact_substeps", {}).get("rock_ramp", 0)
            for row in rows), default=0),
        "maximum_rock_truck_contact_substeps": max((
            row.get("rigid_contact_substeps", {}).get("rock_truck", 0)
            for row in rows), default=0),
        "all_frames_hard_safe": all(
            row["hard_guard"] for row in rows),
        "velocity_advisory_observed": any(
            row.get("advisory_active", False) for row in rows),
        "velocity_advisory_frame_count": sum(
            bool(row.get("advisory_active", False)) for row in rows),
        "phase_labels_seen": list(dict.fromkeys(
            event["label"] for event in phase_events)),
        "operator_event_labels": [
            event["label"] for event in event_annotations],
        "posthoc_review_required": True,
        "automatic_success_label": None,
        "training_eligible": False,
    }


def _cache_rock_vertices(root: Path, records: list[dict]) -> list[np.ndarray]:
    cached = []
    for record in records:
        vertices, _ = motion._read_obj(
            root / "assets/v4/mixed-material-r0" /
            f"{record['prototype_id']}.obj")
        cached.append(np.asarray(vertices, dtype=np.float64)
                      * float(record["equivalent_diameter_m"]))
    return cached


def run(args):
    root = Path(args.repository_root).resolve()
    transport_mode = bool(getattr(args, "loaded_transport", False))
    cycle_mode = bool(getattr(args, "cycle_checkpoint", False))
    cycle_save = bool(getattr(args, "cycle_save", False) or cycle_mode)
    physics_truck = transport_mode or cycle_mode
    cycle_metadata = None
    phase_markers = (
        TRANSPORT_PHASE_MARKERS if transport_mode else PHASE_MARKERS)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    started = time.perf_counter()
    rows: list[dict] = []
    phase_events: list[dict] = []
    event_annotations: list[dict] = []
    phase = "unmarked"
    stop_reason = None
    fault_frozen = {
        "active": False,
        "frame": None,
        "failed_checks": [],
        "snapshot": None,
    }
    fixture = bridge = viewer = controller = None
    sweep_runtime = None
    live_fill = None
    fill_requested = False
    prior_sigterm = None
    source_before = metadata_before = checkpoint_before = None
    bindings: dict[str, str] = {}
    report = {
        "schema": (
            "layered_47_loaded_manual_transport_episode_v1"
            if transport_mode else
            "layered_47_sparse_manual_contact_episode_v1"),
        "status": "initializing",
        "scope": {
            "operator_controls_complete_contact_episode": True,
            "scripted_excavation_motion": False,
            "contact_enabled": True,
            "training_eligible": False,
            "formal_a30_equivalent": False,
            "posthoc_review_required": True,
            "source_checkpoint_written": False,
            "transport_unload_development_only": transport_mode,
        },
        "device": args.device,
        "runner_sha256": motion.probe.bench.sha256(Path(__file__)),
    }
    motion.probe.bench._write_json(report_path, report)

    def progress(event, **values):
        append_json(output / "progress.ndjson", {
            "event": event, **values})

    def write_snapshot(label):
        if fixture is None or bridge is None or not rows:
            return None
        values = motion.probe.snapshots.capture_arrays(
            fixture, bridge.state,
            float(arrays["simulation_time_s"][0]) + len(rows) / HZ)
        try:
            info = motion.probe.snapshots.write_checkpoint_atomic(
                output / f"{label}-{len(rows):06d}.diagnostic.npz", values)
        except ValueError as error:
            raw = motion.probe.snapshots.write_rejected_diagnostic_atomic(
                output / f"{label}-{len(rows):06d}.rejected-raw.npz",
                values, str(error))
            report.setdefault("rejected_diagnostic_snapshots", []).append(raw)
            progress("rejected_raw_diagnostic_saved", snapshot=raw)
            raise
        metadata = {
            "schema": "layered_47_manual_contact_diagnostic_snapshot_v1",
            "status": "diagnostic_only_not_resume_or_training_qualified",
            "source_handoff": checkpoint_before,
            "frames": len(rows),
            "phase": phase,
            "stop_reason": stop_reason,
            "runtime_state": (
                "fault_frozen" if fault_frozen["active"] else
                "budget_paused" if label == "budget" else "running"),
            "scope": report["scope"],
            "controller_state_diagnostic_only": compact_state(bridge.state),
            "moving_bucket_sweep_state": (
                sweep_runtime.diagnostic_state() if sweep_runtime else None),
            **info,
        }
        metadata_path = Path(info["path"] + ".json")
        motion.probe.bench._write_json(metadata_path, metadata)
        info["metadata_sha256"] = motion.probe.bench.sha256(metadata_path)
        report.setdefault("diagnostic_snapshots", []).append(info)
        progress("diagnostic_snapshot", label=label, snapshot=info)
        if cycle_save:
            import layered_47_cycle_checkpoint as cycle_checkpoint
            from layered_47_cycle_bridge import controller_snapshot
            parent = cycle_metadata or dict(source_identity=checkpoint_before,
                lineage_root=info['sha256'], historical_native_history_missing=False)
            saved = cycle_checkpoint.save(output / f"{label}-{len(rows):06d}.cycle.npz",
                values, controller_snapshot(bridge.state), sweep_runtime.diagnostic_state(),
                parent=parent, stage='failure' if label=='failure' else 'loaded_manual_endpoint',
                cycle_index=parent.get('cycle_index', 1), truck_enabled=physics_truck,
                solver_profile=approach_report['solver_profile'], runtime_bindings=bindings,
                selected_cut=parent.get('selected_cut'))
            report.setdefault('cycle_checkpoints', []).append(saved)
        return info

    try:
        if args.rigid_internal_hz != RIGID_INTERNAL_HZ:
            raise ValueError("manual_contact_requires_rigid_4800hz")
        if args.manual_input not in ("keyboard", "gamepad"):
            raise ValueError("manual_contact_input_device")

        config_path = Path(args.config).resolve()
        local_source_config = Path(args.local_source_config).resolve()
        candidate_report = Path(args.candidate_report).resolve()
        resume_report = Path(args.resume_report).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        checkpoint_metadata_path = Path(args.metadata).resolve()
        execution_path = Path(args.approach_execution).resolve()
        approach_report_path = Path(args.approach_report).resolve()
        audit_path = Path(args.restore_audit).resolve()
        manual_config_path = Path(args.manual_config).resolve()
        display_config_path = Path(args.display_config).resolve()

        if cycle_mode:
            import layered_47_cycle_checkpoint as cycle_checkpoint
            arrays, cycle_metadata = cycle_checkpoint.load(checkpoint_path)
            checkpoint_before = cycle_metadata['source_identity']
            metadata_before = cycle_metadata['source_metadata_identity']
            audit_document = json.loads(audit_path.read_text())
            if (cycle_metadata['stage'] != 'prepoint'
                    or cycle_metadata['cycle_index'] not in (2, 3)
                    or not cycle_metadata['truck_enabled']
                    or audit_document.get('status') != 'hold_passed'
                    or audit_document.get('source') != checkpoint_before
                    or audit_document.get('frames_completed', 0) < 3
                    or not audit_document.get('zero_step', {}).get('exact')
                    or max(audit_document['zero_step']['cycle_errors'].values()) > 1e-12):
                raise ValueError('cycle_manual_requires_audited_prepoint')
            face = cycle_metadata['selected_cut']['contact']
            selected_cut = dict(entry_xy_m=face['contact_xy_m'], surface_z_m=face['contact_surface_z_m'])
            approach_report = {'solver_profile': cycle_metadata['solver_profile']}
            report['cycle_lineage'] = {key: cycle_metadata[key] for key in (
                'cycle_index', 'lineage_root', 'parent_checkpoint', 'historical_native_history_missing')}
        else:
            approach_report = json.loads(
                approach_report_path.read_text(encoding="utf-8"))
            if (approach_report.get("schema") !=
                    "layered_47_sparse_approach_handoff_run_v1"
                    or approach_report.get("status") !=
                    "sparse_approach_handoff_saved_unaudited"
                    or not all(approach_report.get("checks", {}).values())):
                raise ValueError("manual_contact_approach_report")
            provenance = approach_report["handoff_provenance"]
            if transport_mode:
                arrays, loaded_metadata, checkpoint_before = (
                    loaded_handoff.load_checkpoint_readonly(
                        checkpoint_path, checkpoint_metadata_path))
            else:
                arrays, _, checkpoint_before = handoff.load_checkpoint_readonly(
                    checkpoint_path, checkpoint_metadata_path, provenance)
            metadata_before = motion.probe.bench.checkpoint_identity(
                checkpoint_metadata_path)
            if transport_mode:
                audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
                if (audit_document.get("status") !=
                        "truck_physics_static_gate_passed"
                        or not all(audit_document.get("checks", {}).values())
                        or audit_document.get("checks", {}).get(
                            "static_shape_mapping_verified") is not True
                        or audit_document.get("checks", {}).get(
                            "static_shape_query_probe_passed") is not True
                        or audit_document.get("source_checkpoint") !=
                            checkpoint_before
                        or audit_document.get("source_metadata") !=
                            metadata_before
                        or audit_document.get("scope", {}).get(
                            "transport_unload_development_eligible") is not True):
                    raise ValueError("manual_transport_truck_gate")
            elif (approach_report["handoff_checkpoint"] != checkpoint_before
                    or approach_report["handoff_metadata"] != metadata_before):
                raise ValueError("manual_contact_handoff_identity")

            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            if (motion.probe.bench.sha256(execution_path) !=
                    provenance["approach_execution_report_sha256"]
                    or execution.get("schema") !=
                        "layered_47_sparse_approach_execution_v1"
                    or execution.get("status") !=
                        "sparse_noncontact_approach_passed"
                    or not all(execution.get("checks", {}).values())):
                raise ValueError("manual_contact_execution_report")
            selected_cut = selected_contact_target(execution)
            if not transport_mode:
                audit_document = json.loads(audit_path.read_text(encoding="utf-8"))
                if not contact.audit_report_passes(
                        audit_document, checkpoint_before, metadata_before):
                    raise ValueError("manual_contact_restore_audit")

        loader, identity = motion.probe.select(47)
        base = loader(config_path, root)
        base["input_checkpoint"] = json.loads(
            local_source_config.read_text(encoding="utf-8"))[
                "input_checkpoint"]
        if cycle_mode:
            source_before = checkpoint_before
        elif transport_mode:
            source_before = approach_report["source_checkpoint"]
        else:
            source_arrays, source, source_before = motion.audit.load_candidate(
                candidate_report, base, 47, HZ)
            motion.audit.validate_seed_audit(
                json.loads(resume_report.read_text(encoding="utf-8")),
                candidate_report, source, source_before, 47, HZ,
                auditor_source_path=getattr(args, "seed_auditor_source", None))
            del source_arrays
        if (not transport_mode and not cycle_mode and (source_before["sha256"] !=
                provenance["source_checkpoint_sha256"]
                or motion.probe.bench.sha256(
                    Path(source_before["path"] + ".json")) !=
                    provenance["source_checkpoint_metadata_sha256"]
                or motion.probe.bench.sha256(candidate_report) !=
                    provenance["candidate_report_sha256"]
                or motion.probe.bench.sha256(resume_report) !=
                    provenance["resume_report_sha256"]
                or motion.probe.bench.sha256(config_path) !=
                    provenance["preflight_config_sha256"]
                or motion.probe.bench.sha256(local_source_config) !=
                    provenance["local_source_config_sha256"])):
            raise ValueError("manual_contact_source_lineage")

        truck_document = None
        truck_fixtures = None
        active_curl_enabled = bool(getattr(args, "active_curl_candidate", False))
        if active_curl_enabled and (
                not getattr(args, "bucket_integral_candidate", False)
                or transport_mode):
            raise ValueError("active_curl_requires_integral_and_original_prepoint")
        if physics_truck:
            if (not getattr(args, "truck_config", None)
                    or not getattr(args, "truck_repository_root", None)
                    or not getattr(args, "bucket_integral_candidate", False)
                    or not getattr(args, "hencky_update_candidate", False)):
                raise ValueError("manual_transport_required_candidates")
            truck_document, truck_fixtures = truck_physics.load_candidate(
                args.truck_config, args.truck_repository_root)

        if cycle_save and not all(getattr(args, key, False) for key in (
                'bucket_integral_candidate', 'active_curl_candidate',
                'hencky_update_candidate', 'moving_bucket_sweep_candidate')):
            raise ValueError('cycle_save_requires_full_candidate_stack')
        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity[
                "expected_ids"]:
            raise ValueError("manual_contact_manifest_order")
        proxy_path, proxy = motion.probe.p47._load_bound_json(
            root, base["open_bucket_proxy"], "proxy")
        base["probe"] = {
            **base["probe"],
            "output_dt_s": 1.0 / HZ,
            "mujoco_internal_substeps": RIGID_INTERNAL_HZ // HZ,
        }
        manual_config = load_manual_input_config(manual_config_path)
        display = json.loads(display_config_path.read_text(
            encoding="utf-8"))["display"]
        kinematic_assembly, chain = _load_kinematics(
            args.kinematics_python_dir, args.kinematics_model,
            args.kinematics_mesh_dir)
        if list(chain.axes) != list(motion.AXES):
            raise ValueError("manual_contact_kinematic_axes")
        projector = ContactCommandProjector(chain)
        cached_rock_vertices = _cache_rock_vertices(root, records)
        fill_config = None
        if getattr(args, "live_fill_config", None):
            from live_bucket_fill import load_config, RESOURCE_KEYS
            if transport_mode:
                raise ValueError("live_fill_currently_contact_only")
            fill_config = load_config(args.live_fill_config)
            report["live_bucket_fill"] = {
                "enabled": True, "trigger": "operator_off_soil_key_4",
                "display_only": True, "automatically_confirms_detachment": False,
                "rocks_and_heap_included": True,
                "config": fill_config,
            }

        binding_paths = [
            checkpoint_path, checkpoint_metadata_path, execution_path,
            approach_report_path, audit_path, candidate_report,
            resume_report, config_path, local_source_config,
            manual_config_path, display_config_path, manifest_path,
            proxy_path, Path(args.kinematics_model), Path(__file__),
            Path(__file__).with_name("soil_surface.py"),
            Path(__file__).with_name("mixed_material_r1.py"),
            Path(__file__).with_name("legacy_push_shelf_cleanup.py"),
            Path(motion.probe.p47.__file__),
            Path(motion.probe.snapshots.__file__),
        ]
        if fill_config:
            binding_paths.extend([
                Path(args.live_fill_config).resolve(),
                Path(__file__).with_name("live_bucket_fill.py"),
                *[Path(fill_config[key]) for key in RESOURCE_KEYS],
            ])
        if getattr(args, "seed_auditor_source", None):
            binding_paths.append(Path(args.seed_auditor_source).resolve())
            report["historical_seed_auditor_source"] = str(
                Path(args.seed_auditor_source).resolve())
        if getattr(args, "bucket_integral_candidate", False):
            binding_paths.extend([
                Path(__file__).with_name("layered_47_bucket_integral_candidate.py"),
                Path(__file__).with_name("bucket_integral_candidate.h"),
            ])
        if active_curl_enabled:
            binding_paths.extend([
                Path(__file__).with_name("layered_47_active_curl_candidate.py"),
                Path(__file__).with_name("bucket_active_curl_candidate.h"),
            ])
        if getattr(args, "hencky_update_candidate", False):
            binding_paths.extend([
                Path(__file__).with_name("soil_hencky_update_candidate.py"),
                Path(__file__).with_name("soil_hencky_update_guard.py"),
            ])
        if getattr(args, "moving_bucket_sweep_candidate", False):
            binding_paths.extend([
                Path(__file__).with_name("moving_bucket_sweep_candidate.py"),
                Path(__file__).with_name("moving_bucket_sweep_runtime.py"),
            ])
        if physics_truck:
            binding_paths.extend([
                Path(args.truck_config), audit_path,
                Path(__file__).with_name(
                    "layered_47_truck_physics_candidate.py"),
                Path(__file__).with_name("mpm_static_contact_ownership.py"),
                Path(__file__).with_name(
                    "layered_47_loaded_handoff_checkpoint.py"),
            ])
        if cycle_save:
            binding_paths.extend(Path(__file__).with_name(name) for name in (
                'layered_47_cycle_bridge.py', 'layered_47_cycle_checkpoint.py'))
        bindings = {str(path): motion.probe.bench.sha256(path)
                    for path in binding_paths}
        initial_terrain, initial_observation = build_fused_terrain(
            root, records, arrays["particle_q"], arrays["rigid_qpos"])
        report.update(
            handoff_checkpoint=checkpoint_before,
            handoff_metadata=metadata_before,
            restore_audit={
                "path": str(audit_path),
                "sha256": motion.probe.bench.sha256(audit_path),
                "status": audit_document["status"],
            },
            source_checkpoint=source_before,
            acquisition_domain=ACQUISITION_DOMAIN,
            selected_cut=selected_cut,
            initial_observation=initial_observation,
            input_bindings=bindings,
            protocol={
                "hz": HZ,
                "rigid_internal_hz": RIGID_INTERNAL_HZ,
                "manual_input": args.manual_input,
                "start_paused": True,
                "neutral_rearm_required": True,
                "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
                "budget_behavior": "save_pause_keep_window_resume_new_segment",
                "wall_budget_excludes_initialization_and_paused_intervals": True,
                "maximum_simulation_seconds": MAXIMUM_SIMULATION_SECONDS,
                "maximum_command_si": COMMAND_LIMITS_SI.tolist(),
                "command_preview_seconds": COMMAND_PREVIEW_SECONDS,
                "bucket_soft_limit_policy": (
                    "clip_outward_preview_allow_in_bounds_hold_or_retreat"),
                "kinematic_preview_command_limits_si":
                    KINEMATIC_PREVIEW_COMMAND_LIMITS_SI.tolist(),
                "kinematic_preview_preserves_requested_joint_displacement":
                    True,
                "bucket_internal_upper_target_m":
                    BUCKET_INTERNAL_UPPER_TARGET_M,
                "soil_contact_allowed": True,
                "rock_bucket_contact_allowed": True,
                "rock_ground_contact_allowed": True,
                "rock_ramp_contact_allowed": False,
                "rock_truck_contact_allowed": physics_truck,
                "soil_truck_contact_allowed": physics_truck,
                "continuous_collision_proven": False,
                "postsolve_projection": {
                    "enabled": True,
                    "reaction_routing": (
                        "unattributed_diagnostic_not_applied_to_mujoco"),
                    "continuous_collision_proven": False,
                },
                "soil_rendering": {
                    "mode": display["soil_render_mode"],
                    "surface_voxel_m": display["surface_voxel_m"],
                    "surface_update_interval_frames": (
                        display["surface_update_interval_frames"]),
                    "display_only": True,
                },
                "decision_observation_stride_frames":
                    DECISION_OBSERVATION_STRIDE,
                "decision_observations_are_resampling_anchors": True,
                "final_zero_hold_frames": FINAL_ZERO_HOLD_FRAMES,
                "phase_markers": phase_markers,
                "event_markers": EVENT_MARKERS,
                "hard_limits": {
                    "maximum_physical_closure_m":
                        MAXIMUM_PHYSICAL_CLOSURE_M,
                },
                "diagnostic_velocity_references": {
                    "maximum_particle_speed_m_s":
                        MAXIMUM_PARTICLE_SPEED_M_S,
                    "maximum_rock_linear_speed_m_s":
                        MAXIMUM_ROCK_LINEAR_SPEED_M_S,
                    "maximum_rock_angular_speed_rad_s":
                        MAXIMUM_ROCK_ANGULAR_SPEED_RAD_S,
                    "enforced_as_hard_limits": False,
                    "calibrated_for_operational_contact": False,
                },
                "hard_fault_behavior": "freeze_viewer_until_operator_exit",
                "hencky_update_candidate_requested": bool(
                    getattr(args, "hencky_update_candidate", False)),
                "loaded_transport_mode": transport_mode,
                "truck_contact_ownership": (
                    truck_document["contact_ownership"]
                    if physics_truck else None),
            },
        )
        progress("validated_inputs", handoff=checkpoint_before,
                 audit=report["restore_audit"])

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with ExitStack() as contexts:
            contexts.enter_context(cleaned_manual_model())
            optimization = contexts.enter_context(optimized_bridge(
                args.mechanism_header_dir, output / "build"))
            integral_policy = contexts.enter_context(bucket_integral_bridge(
                getattr(args, "bucket_integral_candidate", False),
                output / "bucket-integral-build"))
            active_curl_policy = contexts.enter_context(active_curl_bridge(
                active_curl_enabled, output / "active-curl-build",
                initialize_from_d120=cycle_mode))
            if cycle_save:
                from layered_47_cycle_bridge import installed_cycle_bridge
                contexts.enter_context(installed_cycle_bridge(output / 'cycle-build'))
            if physics_truck:
                truck_fixtures = contexts.enter_context(
                    truck_physics.installed_candidate(
                        args.truck_config, args.truck_repository_root))
            hencky_enabled = bool(
                getattr(args, "hencky_update_candidate", False))
            hencky_policy = {
                "enabled": hencky_enabled,
                "candidate_source_sha256": None,
                "recorded_trajectory_qualified": False,
                "manual_single_scoop_qualified": False,
                "training_eligible": False,
            }
            if hencky_enabled:
                import soil_hencky_update_candidate as hencky_candidate
                candidate_path = Path(hencky_candidate.__file__).resolve()
                candidate_sha256 = motion.probe.bench.sha256(candidate_path)
                if candidate_sha256 != HENCKY_CANDIDATE_SHA256:
                    raise ValueError("unvalidated_hencky_candidate_source")
                contexts.enter_context(hencky_candidate.installed_candidate())
                hencky_policy.update(
                    candidate_source_sha256=candidate_sha256,
                    recorded_trajectory_qualified=True,
                    recorded_trajectory_evidence=(
                        "work/soil-hencky-full-replay-20260923-v1"),
                )
            raw = contexts.enter_context(tempfile.TemporaryDirectory(
                prefix="sparse-manual-contact-"))
            if cycle_mode:
                with loaded_handoff.restored_sparse_grid_topology(arrays):
                    fixture, held, restored = motion.audit.restore_candidate_runtime(
                        args, base, root, records, arrays, proxy,
                        cycle_metadata['solver_profile'], Path(raw),
                        cycle_metadata['controller_state'], truck_fixtures)
                from layered_47_cycle_bridge import restore_frame_interface, controller_snapshot
                restore_frame_interface(held.bridge.state, arrays, cycle_metadata['controller_state'])
                fixture.assign(held.bridge.state)
                actual = controller_snapshot(held.bridge.state)
                errors = {key: float(np.max(np.abs(np.asarray(value)-np.asarray(actual[key]))))
                          for key,value in cycle_metadata['controller_state'].items()
                          if key != 'native_state_kind'}
                if max(errors.values()) > 1e-12: raise ValueError('cycle_manual_native_restore_inexact')
                restored['cycle_errors'] = errors
            elif transport_mode:
                with loaded_handoff.restored_sparse_grid_topology(arrays):
                    fixture, held, restored = (
                        motion.audit.restore_candidate_runtime(
                            args, base, root, records, arrays, proxy,
                            approach_report["solver_profile"], Path(raw),
                            loaded_metadata["controller_state"],
                            truck_fixtures))
            else:
                fixture, held, restored = (
                    motion.audit.restore_candidate_runtime(
                        args, base, root, records, arrays, proxy,
                        approach_report["solver_profile"], Path(raw)))
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            report["restore"] = restored
            report["legacy_push_shelf_cleanup"] = restored[
                "runtime_build"]["model"]["legacy_push_shelf_cleanup"]
            report["bucket_side_winding_audit"] = fixture.bucket_side_winding_audit
            report["optimization"] = optimization.copy()
            report["bucket_integral_candidate"] = integral_policy.copy()
            report["active_curl_candidate"] = active_curl_policy.copy()
            report["hencky_update_candidate"] = hencky_policy.copy()
            material_guard = None
            if hencky_enabled:
                from soil_hencky_update_guard import installed_material_guard
                material_guard = contexts.enter_context(
                    installed_material_guard(fixture.solver))

            local_proxy = _proxy_local_vertices(proxy)
            predicted_pose = kinematic_assembly.assemble(
                np.asarray(bridge.state["mechanism_qpos"]),
                include_meshes=False)[1]["bucket"]
            predicted_vertices = (
                local_proxy @ predicted_pose[:3, :3].T
                + predicted_pose[:3, 3])
            actual_vertices = motion.probe.p47._bucket_world_vertices(
                proxy, bridge.state)
            if (predicted_vertices.shape != actual_vertices.shape
                    or not np.allclose(
                        predicted_vertices, actual_vertices,
                        rtol=0.0, atol=1.0e-7)):
                raise ValueError("manual_contact_prediction_geometry")

            sweep_policy = {
                "enabled": bool(getattr(args, "moving_bucket_sweep_candidate", False)),
                "candidate_source_sha256": None,
                "native_projection_reaction_attributed": False,
                "actual_motion_reaction_delay_frames": 1,
                "exact_rotational_ccd": False,
                "collection_purpose": "contact_bc_candidate_demonstrations",
                "training_admission_automatic": False,
            }
            if sweep_policy["enabled"]:
                from moving_bucket_sweep_runtime import MovingBucketSweepRuntime
                sweep_runtime = MovingBucketSweepRuntime(
                    fixture, bridge, proxy, base["mpm"]["material"]["friction"])
                if cycle_mode:
                    cycle_checkpoint.restore_sweep(sweep_runtime, cycle_metadata["moving_bucket_sweep_state"])
                sweep_policy["candidate_source_sha256"] = bindings[str(
                    Path(__file__).with_name("moving_bucket_sweep_candidate.py"))]
            report["moving_bucket_sweep_candidate"] = sweep_policy.copy()
            captured: dict[str, dict] = {}
            original_fixture_step = fixture.step

            def recording_fixture_step(dt, origins):
                measured, particles = (
                    sweep_runtime.step_mpm(original_fixture_step, dt, origins)
                    if sweep_runtime else original_fixture_step(dt, origins))
                captured["wrenches"] = measured
                return measured, particles

            fixture.step = recording_fixture_step

            initial_drive_load = np.asarray(
                bridge.state["equivalent_drive_load"], dtype=np.float64)
            load_marker = {
                "label": "initial_unmarked",
                "frame": 0,
                "value": initial_drive_load.copy(),
                "peak_abs_change": np.zeros(4, dtype=np.float64),
            }

            def step(requested, applied, input_record, *, final_hold=False):
                before_state = compact_state(bridge.state)
                bridge.command = np.asarray(applied, dtype=np.float64)
                if material_guard is not None:
                    material_guard.prepare_frame()
                row, _ = motion.probe.bench._advance_frame(
                    fixture, bridge, base, len(rows) + 1)
                row["moving_bucket_sweep"] = (
                    sweep_runtime.finish_frame() if sweep_runtime
                    else {"enabled": False})
                if sweep_runtime:
                    report["moving_bucket_sweep_state"] = sweep_runtime.diagnostic_state()
                    # Guards and logs see the final protected particle state.
                    protected_q = fixture.state.particle_q.numpy()
                    protected_v = fixture.state.particle_qd.numpy()
                    protected_speed = np.linalg.norm(protected_v, axis=1)
                    row.update(
                        finite=bool(row["finite"] and np.isfinite(protected_q).all()
                                    and np.isfinite(protected_v).all()),
                        particle_max_speed_m_s=float(protected_speed.max()),
                        particle_p99_speed_m_s=float(np.quantile(protected_speed, .99)),
                        particle_minimum_z_m=float(protected_q[:, 2].min()))
                row["hencky_material_guard"] = (
                    material_guard.summarize(fixture.state)
                    if material_guard is not None else {"enabled": False})
                qpos = np.asarray(
                    bridge.state["mechanism_qpos"], dtype=np.float64)
                rock_positions = np.asarray([
                    rock["qpos"][:3] for rock in bridge.state["rocks"]],
                    dtype=np.float64)
                row.update(
                    phase=("terminal_zero_hold" if final_hold else phase),
                    requested_command_si=np.asarray(
                        requested, dtype=np.float64).tolist(),
                    applied_command_si=bridge.command.tolist(),
                    command_projection_reason=projector.reason,
                    bucket_end_stop_projected=
                        projector.bucket_end_stop_projected,
                    axis_positions=qpos[motion.AXES].tolist(),
                    axis_velocity=np.asarray(
                        bridge.state["mechanism_qvel"],
                        dtype=np.float64)[motion.AXES - 1].tolist(),
                    bucket_linear_speed_m_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_linear_velocity_world_m_s"])),
                    bucket_angular_speed_rad_s=float(np.linalg.norm(
                        bridge.state[
                            "bucket_angular_velocity_world_rad_s"])),
                    rock_displacement_from_source_m=float(np.max(
                        np.linalg.norm(
                            rock_positions - arrays["rigid_qpos"][:, :3],
                            axis=1))),
                    joint_bounds_ok=projector.physical_bounds_ok(qpos),
                    elapsed_simulation_seconds=(len(rows) + 1) / HZ,
                    before_state=before_state,
                    after_state=compact_state(bridge.state),
                    bucket_integral_effort=list(bridge.state.get(
                        "bucket_integral_effort", [0.0] * 4)),
                    rigid_contact_substeps={
                        name: int(value) for name, value in
                        bridge.state.get("contact_substeps", {}).items()
                    },
                    **contact.wrench_summary(captured["wrenches"]["bucket"]),
                )
                row.update(observe_drive_load(
                    bridge.state,
                    previous=(
                        rows[-1]["equivalent_drive_load"]
                        if rows else initial_drive_load),
                    marker_baseline=load_marker["value"]))
                load_marker["peak_abs_change"] = np.maximum(
                    load_marker["peak_abs_change"],
                    np.abs(np.asarray(
                        row["equivalent_drive_load_delta_from_phase_marker"],
                        dtype=np.float64)))
                row.update(
                    drive_load_phase_marker_label=load_marker["label"],
                    drive_load_phase_marker_frame=load_marker["frame"],
                    maximum_abs_drive_load_change_since_phase_marker=(
                        load_marker["peak_abs_change"].tolist()),
                )
                row["finite"] = bool(
                    row["finite"]
                    and motion.probe.bench._finite_tree(bridge.state)
                    and row["bucket_mpm_wrench_finite"]
                    and (not row["hencky_material_guard"]["enabled"]
                         or row["hencky_material_guard"]["safe"]))
                row["stopped"] = motion.stopped(row)
                row["hard_guard_checks"] = hard_guard_checks(
                    row, allow_truck=physics_truck)
                row["hard_guard"] = all(
                    row["hard_guard_checks"].values())
                row["advisory_checks"] = advisory_checks(row)
                row["advisory_active"] = any(
                    row["advisory_checks"].values())
                rows.append(row)
                append_json(output / "input.ndjson", {
                    **input_record,
                    "schema": "layered_47_manual_contact_input_v1",
                    "frame": len(rows),
                    "phase": row["phase"],
                    "requested_command_si": row["requested_command_si"],
                    "applied_command_si": row["applied_command_si"],
                    "command_projection_reason":
                        row["command_projection_reason"],
                    "training_eligible": False,
                })
                append_json(output / "physics.ndjson", row)
                if len(rows) % DECISION_OBSERVATION_STRIDE == 0:
                    append_json(output / "observation.ndjson", {
                        "frame": len(rows),
                        "phase": row["phase"],
                        "state": compact_state(bridge.state),
                        "terrain": terrain_observation(
                            fixture, bridge.state, cached_rock_vertices),
                        "training_eligible": False,
                    })
                if (len(rows) == 1 or len(rows) % 30 == 0
                        or not row["hard_guard"]
                        or row["advisory_active"]):
                    print(
                        "MANUAL_CONTACT "
                        f"frame={len(rows)} phase={row['phase']} "
                        f"soil={row['bucket_mpm_contact_samples']} "
                        f"rock={row['rock_bucket_contact_substeps']} "
                        "load=" + str([
                            round(value, 3) for value in
                            row["equivalent_drive_load"]]) + " "
                        "dload=" + str([
                            round(value, 3) for value in
                            row[
                                "equivalent_drive_load_delta_from_previous"]
                        ]) + " "
                        f"hard_safe={row['hard_guard']} "
                        f"velocity_notice={row['advisory_active']}",
                        flush=True)
                if latch_hard_fault(fault_frozen, row, len(rows)):
                    bridge.command = np.zeros(4, dtype=np.float64)
                    gate.armed = False
                    keyboard.clear()
                    progress(
                        "hard_guard_fault_frozen",
                        frame=fault_frozen["frame"],
                        failed_checks=fault_frozen["failed_checks"])
                    try:
                        fault_frozen["snapshot"] = write_snapshot("failure")
                    except Exception as error:
                        fault_frozen["snapshot_error"] = str(error)
                return row["hard_guard"]

            import pyglet
            import pyglet.display.xlib as pyglet_xlib
            if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
                pyglet_xlib.XlibScreen.is_primary = False
            from pyglet.window import key
            (display_assembly, display_model, display_state,
             display_names, to_pose) = display_helpers._display_runtime(
                 display)
            viewer = newton.viewer.ViewerGL(
                width=display["width"], height=display["height"],
                vsync=False, paused=True)
            viewer.set_model(display_model)
            viewer.picking_enabled = False
            viewer.renderer.set_title(
                "MANUAL LOADED TRANSPORT PILOT - NOT TRAINING DATA"
                if transport_mode else
                "MANUAL CONTACT PILOT - NOT TRAINING DATA")
            viewer.set_camera(
                wp.vec3(*display["camera_position_m"]),
                pitch=display["camera_pitch_deg"],
                yaw=display["camera_yaw_deg"])
            soil_surface = SoilSurface(
                fixture.state.particle_q,
                particle_spacing_m=float(base["mpm"]["particle_spacing_m"]),
                voxel_size_m=float(display["surface_voxel_m"]))
            display_helpers._prepare_rock_visuals(
                viewer, records, root, args.device)

            gate = InputGate("normal")
            session_budget = ManualSessionBudget(time.perf_counter())
            budget_notice = {"reason": None}
            keyboard = KeyboardFourAxisState()
            manual = keyboard
            focus = {"active": True}
            if args.manual_input == "gamepad":
                manager = pyglet.input.ControllerManager()
                controller = display_helpers._select_controller(
                    "gamepad", manager.get_controllers(), args.gamepad_index)
                controller.open()
                manual = GamepadFourAxisState(manual_config)
                controller.push_handlers(
                    on_stick_motion=lambda selected, stick, vector:
                        manual.set_stick(stick, vector.x, vector.y),
                    on_button_press=lambda selected, button:
                        manual.press_button(button))
                manager.push_handlers(
                    on_disconnect=lambda selected:
                        manual.disconnect() if selected is controller else None)
                report["gamepad_name"] = controller.name

            def marker_event(kind, label):
                nonlocal phase, fill_requested
                if kind == "phase":
                    load_marker.update(
                        label=label,
                        frame=len(rows) + 1,
                        value=np.asarray(
                            bridge.state["equivalent_drive_load"],
                            dtype=np.float64).copy(),
                        peak_abs_change=np.zeros(4, dtype=np.float64),
                    )
                event = {
                    "schema": "layered_47_manual_contact_annotation_v1",
                    "kind": kind,
                    "label": label,
                    "next_frame": len(rows) + 1,
                    "sim_time_s": len(rows) / HZ,
                    "wall_monotonic_s": time.monotonic(),
                    "training_eligible": False,
                    "equivalent_drive_load_baseline": (
                        load_marker["value"].tolist()
                        if kind == "phase" else None),
                    "drive_load_axis_labels": list(AXIS_LABELS),
                    "drive_load_axis_units": list(AXIS_UNITS),
                    "load_contact_thresholds_applied": False,
                }
                if kind == "phase":
                    phase = label
                    fill_requested = bool(fill_config and label == "off_soil_confirm")
                    phase_events.append(event)
                else:
                    event_annotations.append(event)
                append_json(output / "annotations.ndjson", event)

            phase_symbols = {
                key._1: phase_markers["1"],
                key._2: phase_markers["2"],
                key._3: phase_markers["3"],
                key._4: phase_markers["4"],
                key.R: phase_markers["R"],
            }
            event_symbols = {
                key._5: EVENT_MARKERS["5"],
                key._6: EVENT_MARKERS["6"],
                key._7: EVENT_MARKERS["7"],
                key._8: EVENT_MARKERS["8"],
            }

            def press(symbol, modifiers):
                nonlocal stop_reason
                names_by_key = {
                    key.J: "J", key.L: "L", key.I: "I", key.K: "K",
                    key.U: "U", key.O: "O", key.N: "N", key.M: "M",
                }
                if symbol in names_by_key:
                    keyboard.press(names_by_key[symbol])
                if symbol in phase_symbols:
                    marker_event("phase", phase_symbols[symbol])
                if symbol in event_symbols:
                    marker_event("event", event_symbols[symbol])
                if symbol in (key.X, key.ESCAPE):
                    stop_reason = "user_finished"
                if symbol == key.SPACE:
                    gate.armed = False
                    keyboard.clear()

            def release(symbol, modifiers):
                keyboard.release(key.symbol_string(symbol))

            def deactivated():
                focus["active"] = False
                gate.armed = False
                keyboard.clear()

            viewer.renderer.window.push_handlers(
                on_key_press=press,
                on_key_release=release,
                on_activate=lambda: focus.update(active=True),
                on_deactivate=deactivated)

            def ui(imgui):
                imgui.text(
                    "MANUAL LOADED TRANSPORT / NOT TRAINING DATA"
                    if transport_mode else
                    "MANUAL CONTACT PILOT / NOT TRAINING DATA")
                if fill_config:
                    from live_bucket_fill import display_lines
                    if phase != "off_soil_confirm":
                        imgui.text("FILL: press 4 after clearing soil")
                    else:
                        for line in display_lines(
                                live_fill.latest if live_fill else None, len(rows)):
                            imgui.text(line)
                if integral_policy["enabled"]:
                    effort = bridge.state["bucket_integral_effort"][3] / 1000
                    imgui.text(f"BUCKET INTEGRAL PILOT | assist {effort:.1f} kN")
                if active_curl_policy["enabled"]:
                    extra = bridge.state["bucket_active_curl_effort"][3] / 1000
                    imgui.text(f"ACTIVE CURL PILOT | extra {extra:.1f} kN")
                if hencky_policy["enabled"]:
                    imgui.text("HENCKY SOIL CANDIDATE | guarded manual pilot")
                if sweep_policy["enabled"]:
                    imgui.text("BUCKET PATH GUARD / BC DEMO CANDIDATE")
                imgui.text(
                    "Space pause/run | neutral sticks to arm | "
                    "X/Esc/Back finish")
                imgui.text(
                    "1 lift/transport | 2 align receiver | 3 unload/open | "
                    "4 settle/return | R recovery"
                    if transport_mode else
                    "1 penetrate | 2 curl/load | 3 lift/clear | "
                    "4 off-soil | R recovery")
                imgui.text(
                    "Press 1 before moving: marks the load baseline; "
                    "it does not move the excavator")
                imgui.text(
                    "5 jam | 6 spill | 7 load detached | 8 clear")
                imgui.text(
                    "Left: swing/boom | Right: bucket/arm"
                    if args.manual_input == "gamepad" else
                    "J/L swing | I/K boom | U/O arm | N/M bucket")
                imgui.text(
                    f"Input {args.manual_input} | phase {phase} | "
                    f"frames {len(rows)} | armed {gate.armed}")
                if budget_notice["reason"]:
                    imgui.text("TIME BUDGET: saved and paused; Space continues")
                    imgui.text("Neutral sticks to rearm; X saves and finishes")
                if fault_frozen["active"]:
                    imgui.text("FAULT FROZEN: physics stopped; viewer retained")
                    imgui.text(
                        "Failed hard checks: "
                        + ", ".join(fault_frozen["failed_checks"]))
                    imgui.text("Use X/Esc/Back only after inspection")
                imgui.text(
                    "GREEN MARKER selected entry: "
                    f"x={selected_cut['entry_xy_m'][0]:.2f}, "
                    f"y={selected_cut['entry_xy_m'][1]:.2f}, "
                    f"z={selected_cut['surface_z_m']:.2f}")
                if rows:
                    last = rows[-1]
                    imgui.text(
                        "Applied [swing, boom, arm, bucket]: " +
                        str([round(value, 4) for value in
                             last["applied_command_si"]]))
                    imgui.text(
                        f"Soil samples {last['bucket_mpm_contact_samples']} | "
                        f"rock substeps {last['rock_bucket_contact_substeps']} | "
                        f"guard {last['hard_guard']}")
                    if transport_mode:
                        imgui.text(
                            "Truck soil samples / rock substeps: "
                            f"{last.get('truck_mpm_contact_samples', 0)} / "
                            f"{last.get('rigid_contact_substeps', {}).get('rock_truck', 0)}")
                    imgui.text(
                        "Velocity notice (diagnostic only): "
                        + str(last["advisory_active"]))
                    imgui.text(
                        "particle/rock lin/rock ang: "
                        f"{last['particle_max_speed_m_s']:.3f} / "
                        f"{last['maximum_rock_linear_speed_m_s']:.3f} / "
                        f"{last['maximum_rock_angular_speed_rad_s']:.3f}")
                    imgui.text(
                        "External load [swing Nm | boom arm bucket N]")
                    imgui.text("now: " + str([
                        round(value, 2) for value in
                        last["equivalent_drive_load"]]))
                    imgui.text("delta/frame: " + str([
                        round(value, 2) for value in last[
                            "equivalent_drive_load_delta_from_previous"]]))
                    imgui.text(
                        f"delta/from {last['drive_load_phase_marker_label']}: "
                        + str([round(value, 2) for value in last[
                            "equivalent_drive_load_delta_from_phase_marker"]]))
                    imgui.text("peak abs/from marker: " + str([
                        round(value, 2) for value in last[
                            "maximum_abs_drive_load_change_since_phase_marker"]
                    ]))
                    imgui.text(
                        "No load threshold is being used to declare contact")
                    imgui.text(
                        "Projection: " +
                        str(last["command_projection_reason"] or "clear"))
                    if last["hencky_material_guard"]["enabled"]:
                        material = last["hencky_material_guard"]
                        imgui.text(
                            "Hencky guard: " + str(material["safe"]) +
                            f" | reject {material['rejection_count']}" +
                            " | Fnorm " + str(material["maximum_F_frobenius"]) +
                            " | Jp max " + str(material["maximum_Jp"]))

            viewer.register_ui_callback(ui, position="side")

            def signal_stop(*_):
                nonlocal stop_reason
                stop_reason = "signal_requested"

            prior_sigterm = signal.signal(signal.SIGTERM, signal_stop)
            draws = 0
            while viewer.is_running() and stop_reason is None:
                if fill_requested and live_fill is None:
                    from live_bucket_fill import LiveFillDisplay
                    live_fill = LiveFillDisplay(
                        fill_config, records, cached_rock_vertices,
                        output / "live-fill.ndjson")
                    contexts.callback(live_fill.close)
                if live_fill is not None:
                    live_fill.poll()
                    if fill_requested and live_fill.due(len(rows)):
                        fill_pose = kinematic_assembly.assemble(
                            np.asarray(bridge.state["mechanism_qpos"]),
                            include_meshes=False)[1]["bucket"]
                        live_fill.submit(
                            len(rows), fixture.state.particle_q.numpy(), fill_pose,
                            [rock["qpos"] for rock in bridge.state["rocks"]])
                        fill_requested = False
                if not fault_frozen["active"]:
                    budget_reason = session_budget.poll(
                        time.perf_counter(), len(rows), viewer.is_paused())
                    if budget_reason:
                        viewer._paused = True  # Same flag as Newton's Pause UI.
                        gate.armed = False
                        keyboard.clear()
                        budget_notice["reason"] = budget_reason
                        snapshot = write_snapshot("budget")
                        event = {"reason": budget_reason, "frame": len(rows),
                                 "snapshot": snapshot,
                                 "extra_physics_steps": 0}
                        report.setdefault("budget_pauses", []).append(event)
                        progress("budget_paused", **event)
                        motion.probe.bench._write_json(report_path, _clean(report))
                    elif not viewer.is_paused():
                        budget_notice["reason"] = None
                draw_with_selected_cut(
                    viewer, display_state, fixture, bridge.state, records,
                    display_assembly, display_names, to_pose, args.device,
                    display, selected_cut, len(rows), soil_surface)
                draws += 1
                if draws == 3:
                    ready = {
                        "paused": bool(viewer.is_paused()),
                        "physics_frames": len(rows),
                        "gamepad_name": report.get("gamepad_name"),
                        "bucket_integral_candidate": integral_policy.copy(),
                        "active_curl_candidate": active_curl_policy.copy(),
                        "hencky_update_candidate": hencky_policy.copy(),
                        "moving_bucket_sweep_candidate": sweep_policy.copy(),
                        "initial_state": compact_state(bridge.state),
                        "restore_exact": bool(restored["exact"]),
                        "legacy_push_shelf_cleanup": report[
                            "legacy_push_shelf_cleanup"],
                        "source_handoff": checkpoint_before,
                    }
                    motion.probe.bench._write_json(
                        output / "viewer-ready.json", _clean(ready))
                    progress("viewer_ready", **ready)
                if fault_frozen["active"]:
                    # Keep processing/rendering the window while guaranteeing
                    # that neither input nor the physics state can advance.
                    time.sleep(0.01)
                    continue
                if (args.ui_smoke or args.zero_probe_frames) and draws >= 3:
                    if args.zero_probe_frames:
                        for probe_index in range(args.zero_probe_frames):
                            projector.reason = None
                            projector.bucket_end_stop_projected = False
                            hard_safe = step(np.zeros(4), np.zeros(4), {
                                "wall_monotonic_s": time.monotonic(),
                                "sim_time_before_s": len(rows) / HZ,
                                "raw_controls": {"zero_probe": True},
                                "normalized_before_projection": [0.0] * 4,
                                "window_focused": focus["active"],
                                "connected": getattr(
                                    manual, "connected", True),
                                "armed": False,
                                "emergency_stop_latched": False,
                                "input_device": "system_zero_probe",
                                "probe_index": probe_index,
                            })
                            if not hard_safe:
                                break
                        if not fault_frozen["active"]:
                            stop_reason = "zero_probe_completed"
                    else:
                        stop_reason = "ui_smoke_paused_no_physics"
                    break
                paused = viewer.is_paused()
                action = manual.action(window_focused=focus["active"])
                requested = gate.command(
                    action, np.zeros(4), paused=paused,
                    focused=focus["active"],
                    connected=getattr(manual, "connected", True),
                    estop=getattr(
                        manual, "emergency_stop_latched", False))
                stop_reason = stop_reason or gate.stop_reason
                if stop_reason or not viewer.is_running():
                    break
                if not paused and not args.ui_smoke:
                    applied = projector.project(
                        bridge.state["mechanism_qpos"], requested)
                    step(requested, applied, {
                        "wall_monotonic_s": time.monotonic(),
                        "sim_time_before_s": len(rows) / HZ,
                        "raw_controls": manual.raw_controls(),
                        "normalized_before_projection": action,
                        "window_focused": focus["active"],
                        "connected": getattr(manual, "connected", True),
                        "armed": gate.armed,
                        "emergency_stop_latched": getattr(
                            manual, "emergency_stop_latched", False),
                        "input_device": args.manual_input,
                    })
            operator_exit_reason = stop_reason or "window_closed"
            if fault_frozen["active"]:
                stop_reason = "hard_guard_fault_frozen"
                report["fault_freeze"] = {
                    **fault_frozen,
                    "operator_exit_reason": operator_exit_reason,
                }
            else:
                stop_reason = operator_exit_reason

            # A normal operator finish gets a bounded zero-command tail so
            # post-hoc review can distinguish a stopped end state.  Emergency,
            # disconnect, signal and hard-fault paths never advance physics.
            if (stop_reason in {
                    "user_finished", "window_closed"}
                    and rows and not args.ui_smoke):
                for hold_index in range(FINAL_ZERO_HOLD_FRAMES):
                    projector.reason = None
                    projector.bucket_end_stop_projected = False
                    hard_safe = step(np.zeros(4), np.zeros(4), {
                        "wall_monotonic_s": time.monotonic(),
                        "sim_time_before_s": len(rows) / HZ,
                        "raw_controls": {"terminal_zero_hold": True},
                        "normalized_before_projection": [0.0] * 4,
                        "window_focused": focus["active"],
                        "connected": getattr(manual, "connected", True),
                        "armed": False,
                        "emergency_stop_latched": False,
                        "input_device": "system_safety_hold",
                        "hold_index": hold_index,
                    }, final_hold=True)
                    if not hard_safe:
                        break

            # A fault in the terminal zero tail is still latched and must not
            # permit another physics step. The operator had already requested
            # exit, so only the active-operation path above retains the view.
            if fault_frozen["active"] and "fault_freeze" not in report:
                stop_reason = "hard_guard_fault_frozen"
                report["fault_freeze"] = {
                    **fault_frozen,
                    "operator_exit_reason": operator_exit_reason,
                }

            if not fault_frozen["active"]:
                write_snapshot("final")
            summary = summarize_episode(
                rows, phase_events, event_annotations)
            checks = {
                "restore_exact": bool(restored["exact"]),
                "all_executed_frames_hard_safe":
                    summary["all_frames_hard_safe"],
                "source_handoff_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        checkpoint_path) == checkpoint_before,
                "source_handoff_metadata_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        checkpoint_metadata_path) == metadata_before,
                "original_source_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        source_before["path"]) == source_before,
                "input_bindings_unchanged": all(
                    motion.probe.bench.sha256(Path(path)) == sha256
                    for path, sha256 in bindings.items()),
                "no_scripted_excavation_motion": True,
                "truck_gate_unchanged": bool(
                    not transport_mode
                    or motion.probe.bench.sha256(audit_path) ==
                       bindings[str(audit_path)]),
                "training_not_granted":
                    report["scope"]["training_eligible"] is False,
                "smoke_or_probe_scope_obeyed": bool(
                    (not args.ui_smoke and not args.zero_probe_frames)
                    or (args.ui_smoke and not args.zero_probe_frames
                        and not rows)
                    or (args.zero_probe_frames
                        and len(rows) == args.zero_probe_frames)),
            }
            report.update(
                status=(
                    "manual_transport_pilot_fault_frozen"
                    if transport_mode and fault_frozen["active"] else
                    "manual_transport_zero_probe_completed"
                    if transport_mode and args.zero_probe_frames else
                    "manual_transport_ui_smoke_completed"
                    if transport_mode and args.ui_smoke else
                    "manual_transport_pilot_recorded"
                    if transport_mode else
                    "manual_contact_pilot_fault_frozen"
                    if fault_frozen["active"] else
                    "manual_contact_zero_probe_completed"
                    if args.zero_probe_frames else
                    "manual_contact_ui_smoke_completed" if args.ui_smoke else
                    "manual_contact_pilot_recorded"),
                stop_reason=stop_reason,
                episode_summary=summary,
                checks=checks,
                phase_events=phase_events,
                event_annotations=event_annotations,
                next_gate=(
                    "inspect_frozen_hard_fault_before_retry"
                    if fault_frozen["active"] else
                    "posthoc_review_before_training_eligibility"
                    if not args.ui_smoke and not args.zero_probe_frames else
                    "operator_review_then_manual_pilot"),
            )
            if live_fill is not None:
                live_fill.poll()
                report["live_bucket_fill"]["last_display_result"] = live_fill.latest
                report["live_bucket_fill"]["pending_at_exit"] = live_fill.future is not None
            bridge.close(kill=not all(checks.values()))
            bridge = None
    except (Exception, KeyboardInterrupt) as error:
        stop_reason = stop_reason or "implementation_or_runtime_failure"
        report.update(
            status=("manual_transport_pilot_failed" if transport_mode else
                    "manual_contact_pilot_failed"),
            next_gate="fix_or_review_failure_before_retry",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            })
    finally:
        if prior_sigterm is not None:
            signal.signal(signal.SIGTERM, prior_sigterm)
        if (report.get("status") in {
                "manual_contact_pilot_failed", "manual_transport_pilot_failed"}
                and not report.get("rejected_diagnostic_snapshots")):
            try:
                write_snapshot("failure")
            except Exception as error:
                report["snapshot_error"] = str(error)
        if bridge is not None:
            try:
                bridge.command = np.zeros(4)
                bridge.close(kill=True)
            except Exception as error:
                report.setdefault("cleanup_errors", {})["bridge"] = str(error)
        for name, resource in (("controller", controller),
                               ("viewer", viewer)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report.setdefault("cleanup_errors", {})[name] = str(error)
        report.update(
            stop_reason=stop_reason,
            episode_summary=summarize_episode(
                rows, phase_events, event_annotations),
            timing={
                "frames": len(rows),
                "simulation_seconds": len(rows) / HZ,
                "physics_wall_seconds": sum(
                    row["wall_seconds"] for row in rows),
                "total_wall_seconds": time.perf_counter() - started,
            })
        motion.probe.bench._write_json(report_path, _clean(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in (
            "config", "repository-root", "output", "candidate-report",
            "resume-report", "local-source-config", "checkpoint",
            "metadata", "approach-execution", "approach-report",
            "restore-audit", "manual-config", "display-config", "device",
            "mujoco-include", "mujoco-library-dir",
            "mechanism-header-dir", "controller-header-dir",
            "kinematics-python-dir", "kinematics-model",
            "kinematics-mesh-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--seed-auditor-source",
                        help="Archived producer of the historical seed audit")
    parser.add_argument(
        "--manual-input", choices=("gamepad", "keyboard"),
        default="gamepad")
    parser.add_argument("--gamepad-index", type=int, default=0)
    parser.add_argument(
        "--rigid-internal-hz", type=int, choices=(RIGID_INTERNAL_HZ,),
        default=RIGID_INTERNAL_HZ)
    parser.add_argument("--ui-smoke", action="store_true")
    parser.add_argument(
        "--bucket-integral-candidate", action="store_true",
        help="Opt-in D-120 bucket integral manual pilot; not stopped-qualified")
    parser.add_argument(
        "--active-curl-candidate", action="store_true",
        help="Opt-in D-136 error-based curl assist; requires integral and original prepoint")
    parser.add_argument(
        "--hencky-update-candidate", action="store_true",
        help="Opt-in D-128 Hencky soil update; guarded manual pilot only")
    parser.add_argument(
        "--moving-bucket-sweep-candidate", action="store_true",
        help="Opt-in D-133 bucket path guard for candidate demonstrations")
    parser.add_argument(
        "--zero-probe-frames", type=int, choices=(0, 3, 30), default=0)
    parser.add_argument(
        "--live-fill-config", help="Display-only mixed fill measurement on off-soil key 4")
    parser.add_argument('--cycle-checkpoint', action='store_true', help='Audited continuous second/third scoop prepoint')
    parser.add_argument('--cycle-save', action='store_true', help='Also save full native/controller/sweep frame state')
    parser.add_argument("--loaded-transport", action="store_true")
    parser.add_argument("--truck-config")
    parser.add_argument("--truck-repository-root")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result.get(key) for key in (
        "status", "stop_reason", "episode_summary", "checks",
        "next_gate", "timing", "failure")}, indent=2))
    return 0 if result.get("status") in {
        "manual_contact_ui_smoke_completed",
        "manual_contact_zero_probe_completed",
        "manual_contact_pilot_recorded",
        "manual_transport_ui_smoke_completed",
        "manual_transport_zero_probe_completed",
        "manual_transport_pilot_recorded",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
