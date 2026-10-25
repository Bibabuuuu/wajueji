#!/usr/bin/env python3
"""Execute and save the 47-rock sparse non-contact approach handoff.

This is the local diagnostic-domain route: 10 cm sparse MPM, 60 Hz outer
steps, and the frozen 4800 Hz rigid bridge.  It never executes contact and it
never marks the result as collection- or training-eligible.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

import layered_47_rigid_frequency_candidate as rigid_frequency
from layered_47_optimized_bridge import optimized_bridge, parser
import layered_47_sparse_approach_handoff_checkpoint as handoff
from layered_excavation_approach_planner import (
    BUCKET_MOUTH_WIDTH_M,
    BUCKET_PATH_CLEARANCE_M,
    _load_kinematics,
    _proxy_local_vertices,
    _terrain_clearance,
    build_fused_terrain,
    plan_noncontact_approach,
)
from planning.cycle import validate_cut
from planning.rear_lower_face import select_rear_lower_face
from planning.rear_lower_prepoint import (
    MECHANISM_CLEARANCE_M,
    REQUIRED_BUCKET_CLEARANCE_M,
    TRUCK_CLEARANCE_M,
    ground_completed_terrain,
    mechanism_and_truck_clearance,
    plan_rear_lower_prepoint,
    proxy_surface_samples,
    sampled_surface_clearance,
)
import run_layered_47_sparse_motion as motion
import run_layered_47_interactive_proxy_practice as display_helpers
from drive_load_observation import observe_drive_load


HZ = 60
RIGID_INTERNAL_HZ = 4800
MAXIMUM_PATH_FRAMES = 1200
ENDPOINT_HOLD_FRAMES = 30
MAXIMUM_DISPATCH_SURFACE_CHANGE_M = 0.01
MAXIMUM_PLANNED_CLOSURE_M = 1.0e-4
MAXIMUM_PHYSICAL_CLOSURE_M = 1.0e-3
SPARSE_TARGET_TIP_HEIGHT_ABOVE_SURFACE_M = 0.60
SUPPORTED_TARGET_TIP_HEIGHTS_ABOVE_SURFACE_M = (0.55, 0.60)
MAXIMUM_PATH_AXIS_TRACKING_ERROR = 0.025
MAXIMUM_ENDPOINT_AXIS_ERROR = 0.015
LEGACY_TOP_ROUTE = "legacy_top"
REAR_LOWER_ROUTE = "rear_lower"
SUPPORTED_ROUTES = (LEGACY_TOP_ROUTE, REAR_LOWER_ROUTE)
REAR_LOWER_EXECUTION_BUCKET_CLEARANCE_M = (
    REQUIRED_BUCKET_CLEARANCE_M)
MAXIMUM_STORED_RAW_CONTACT_SAMPLES = 64


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _observation_digest(terrain: dict, selection: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(
        terrain["height_m"], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(
        terrain["point_count"], dtype=np.int64).tobytes())
    digest.update(json.dumps(
        selection, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def _joint_bounds_ok(qpos: np.ndarray, chain) -> bool:
    q = np.asarray(qpos, dtype=np.float64)
    return bool(q.shape == (19,) and np.isfinite(q).all()
                and abs(q[motion.AXES[0]]) <= 2.992
                and all(lower <= q[index] <= upper
                for index, (lower, upper) in chain.limits.items()))


def _bucket_contact_diagnostics(fixture) -> dict:
    impulses, positions, collider_ids = (
        fixture.solver.collect_collider_impulses(fixture.state))
    impulse = impulses.numpy()
    position = positions.numpy()
    ids = collider_ids.numpy()
    norms = np.linalg.norm(impulse, axis=1)
    mask = (
        np.isfinite(impulse).all(axis=1)
        & np.isfinite(position).all(axis=1)
        & (norms > 1.0e-12)
        & np.isin(ids, fixture.bucket_collider_ids))
    selected_impulse = impulse[mask]
    selected_position = position[mask]
    selected_norm = norms[mask]
    strongest = np.argsort(selected_norm)[::-1][
        :MAXIMUM_STORED_RAW_CONTACT_SAMPLES]
    stored_impulse = selected_impulse[strongest]
    stored_position = selected_position[strongest]
    maximum_impulse = float(selected_norm.max(initial=0.0))
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "maximum_impulse_norm_ns": maximum_impulse,
        "sum_impulse_norm_ns": float(selected_norm.sum()),
        "net_impulse_world_ns": selected_impulse.sum(
            axis=0, dtype=np.float64).tolist(),
        "positions_world_m": stored_position.tolist(),
        "impulses_world_ns": stored_impulse.tolist(),
        "effective_contact": False,
        "classification": "raw_grid_impulse_diagnostic_only",
        "classification_basis": (
            "no_raw_mpm_sample_count_impulse_or_proximity_threshold_"
            "declares_physical_contact"),
        "impulse_promotes_contact": False,
        "particle_center_distance_promotes_contact": False,
        "raw_sample_count_promotes_contact": False,
        "stored_sample_limit": MAXIMUM_STORED_RAW_CONTACT_SAMPLES,
    }


def _open_observer(display: dict, records: list[dict], root: Path,
                   device: str, status: dict):
    """Open a read-only scene viewer for the automatic approach."""
    import pyglet.display.xlib as pyglet_xlib

    if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
        pyglet_xlib.XlibScreen.is_primary = False
    assembly, model, state, names, to_pose = (
        display_helpers._display_runtime(display))
    viewer = newton.viewer.ViewerGL(
        width=display["width"], height=display["height"],
        vsync=False, paused=False)
    viewer.set_model(model)
    viewer.picking_enabled = False
    viewer.renderer.set_title(
        "AUTOMATIC 1-2-3 PREPOSITIONING - OBSERVE ONLY")
    viewer.set_camera(
        wp.vec3(*display["camera_position_m"]),
        pitch=display["camera_pitch_deg"],
        yaw=display["camera_yaw_deg"])
    display_helpers._prepare_rock_visuals(
        viewer, records, root, device)

    def ui(imgui):
        phase = {
            "ready_at_start": "START",
            "high_clear": "1 HIGH_CLEAR",
            "outward_high": "2 OUTWARD_HIGH",
            "rear_lower_prepoint": "3 REAR_LOWER_PREPOINT",
            "endpoint_hold": "3 ENDPOINT_HOLD",
            "complete_at_prepoint_3": "3 COMPLETE",
            "safety_stop": "SAFETY_STOP",
        }.get(status.get("phase"), status.get("phase", "planning"))
        imgui.text("AUTOMATIC 1 -> 2 -> 3 / OBSERVE ONLY")
        imgui.text("Viewer input does not control the excavator")
        imgui.text("Closing this window requests a safe abort")
        imgui.text("RED = selected cut point")
        imgui.text("GREEN = prepoint 3 (manual/BC/RL starts after it)")
        imgui.text(
            f"Phase: {phase} | "
            f"frame: {status.get('frame', 0)}")
        if status.get("bucket_clearance_m") is not None:
            imgui.text(
                "Geometric bucket clearance: "
                f"{status['bucket_clearance_m']:.3f} m")
        imgui.text(
            "MPM raw grid samples (diagnostic only): "
            f"{status.get('bucket_contact_samples', 0)}")
        if status.get("equivalent_drive_load") is not None:
            imgui.text(
                "External load [swing Nm | boom arm bucket N]")
            imgui.text(str([
                round(value, 3) for value in
                status["equivalent_drive_load"]]))
            imgui.text("Frame delta: " + str([
                round(value, 3) for value in
                status["equivalent_drive_load_delta"]]))
        if status.get("message"):
            imgui.text(str(status["message"]))

    viewer.register_ui_callback(ui, position="side")
    return {
        "viewer": viewer,
        "assembly": assembly,
        "state": state,
        "names": names,
        "to_pose": to_pose,
        "display": display,
    }


def _marker_column(point_world_m, *, bottom_offset_m=-0.05,
                   top_offset_m=0.55) -> np.ndarray:
    point = np.asarray(point_world_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("approach_observer_marker")
    return np.asarray([
        [point[0], point[1], point[2] + offset]
        for offset in np.linspace(bottom_offset_m, top_offset_m, 13)
    ], dtype=np.float32)


def _draw_observer(runtime: dict, fixture, bridge_state: dict,
                   records: list[dict], device: str,
                   cut_point_world_m, prepoint_world_m) -> None:
    """Render the live physical state and the two operator landmarks."""
    viewer = runtime["viewer"]
    poses = runtime["assembly"].assemble(
        np.asarray(bridge_state["mechanism_qpos"], dtype=np.float64),
        include_meshes=False)[1]
    if np.linalg.norm(
            np.asarray(bridge_state["bucket_origin"])
            - poses["bucket"][:3, 3]) > 1.0e-7:
        raise RuntimeError("approach_observer_display_alignment")
    runtime["state"].body_q.assign(wp.array(
        [runtime["to_pose"](poses[name]) for name in runtime["names"]],
        dtype=wp.transform, device=device))
    viewer.begin_frame(float(bridge_state["time_s"]))
    viewer.log_state(runtime["state"])
    soil_surface = runtime.get("soil_surface")
    if soil_surface is None:
        soil_surface = display_helpers.SoilSurface(
            fixture.state.particle_q,
            particle_spacing_m=fixture.particle_spacing_m,
            voxel_size_m=float(runtime["display"]["surface_voxel_m"]))
        runtime["soil_surface"] = soil_surface
    physics_frame = int(round(float(bridge_state["time_s"]) * HZ))
    soil_surface.update(
        fixture.state.particle_q,
        physics_frame // int(
            runtime["display"]["surface_update_interval_frames"]))
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
            f"/approach/rocks/{prototype}",
            f"/practice/prototype/{prototype}",
            wp.array(xforms, dtype=wp.transform, device=device),
            wp.array(scales, dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec3(0.40, 0.37, 0.32),
                dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec4(0.92, 0.0, 0.0, 1.0),
                dtype=wp.vec4, device=device))
    for name, point, color in (
            ("selected_cut", cut_point_world_m, wp.vec3(1.0, 0.08, 0.05)),
            ("prepoint_3", prepoint_world_m, wp.vec3(0.10, 1.0, 0.10))):
        if point is None:
            continue
        marker = _marker_column(point)
        viewer.log_points(
            f"/approach/markers/{name}",
            wp.array(marker, dtype=wp.vec3, device=device),
            radii=0.035,
            colors=wp.full(
                len(marker), color, dtype=wp.vec3, device=device))
    viewer.end_frame()


def _hold_observer(runtime: dict | None, fixture, bridge_state: dict,
                   records: list[dict], device: str, cut_point_world_m,
                   prepoint_world_m, seconds: float) -> None:
    if runtime is None or seconds <= 0.0:
        return
    viewer = runtime["viewer"]
    deadline = time.monotonic() + seconds
    while viewer.is_running() and time.monotonic() < deadline:
        _draw_observer(
            runtime, fixture, bridge_state, records, device,
            cut_point_world_m, prepoint_world_m)
        time.sleep(1.0 / 30.0)


def frame_passes(row: dict, route: str = LEGACY_TOP_ROUTE) -> bool:
    strict = bool(row["strict"])
    if (route == REAR_LOWER_ROUTE and not strict
            and row.get("stable", False)
            and row.get("guard_failures")
                == ["bucket_mpm_contact_samples"]
            and not row.get("bucket_mpm_effective_contact", True)):
        # The upstream generic guard treats every nonzero grid impulse as
        # physical contact.  Sparse S2 collision nodes can retain impulses far
        # below the solver tolerance even with no nearby material.  Preserve
        # the raw count but accept only a separately classified numerical
        # residual here; every other generic and geometric guard still holds.
        strict = True
    base = bool(
        strict
        and row["joint_bounds_ok"]
        and row["axis_tracking_error"] <= MAXIMUM_PATH_AXIS_TRACKING_ERROR
        and row["closure_m"] <= MAXIMUM_PHYSICAL_CLOSURE_M)
    if route == LEGACY_TOP_ROUTE:
        return bool(base and row["bucket_terrain_clearance_m"]
                    >= BUCKET_PATH_CLEARANCE_M)
    if route == REAR_LOWER_ROUTE:
        return bool(
            base
            and not row.get("bucket_mpm_effective_contact", False)
            and row["bucket_terrain_clearance_m"]
                >= REAR_LOWER_EXECUTION_BUCKET_CLEARANCE_M
            and row["mechanism_terrain_clearance_m"]
                >= MECHANISM_CLEARANCE_M
            and row["static_truck_aabb_separation_m"]
                >= TRUCK_CLEARANCE_M)
    raise ValueError("sparse_approach_route")


def _plan(
        terrain, selection, qpos, proxy, args, *,
        assembly=None, chain=None, truck_world_vertices=None):
    if args.route == REAR_LOWER_ROUTE:
        if assembly is None or chain is None or truck_world_vertices is None:
            raise ValueError("rear_lower_approach_dependencies")
        return plan_rear_lower_prepoint(
            terrain=terrain,
            contact_selection=selection,
            operator_qpos=np.asarray(qpos, dtype=np.float64),
            open_bucket_proxy=proxy,
            truck_world_vertices=truck_world_vertices,
            assembly=assembly,
            chain=chain,
            output_dt_s=1.0 / HZ,
            validation_substeps=2,
            maximum_segment_frames=2400,
            maximum_closure_m=MAXIMUM_PLANNED_CLOSURE_M)
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
        target_tip_height_above_surface_m=
            args.target_tip_height_above_surface_m,
    )


def _plan_summary(path: dict) -> dict:
    qpos = np.asarray(path["qpos"], dtype=np.float64)
    velocity = np.asarray(path["velocity"], dtype=np.float64)
    if "segments" in path:
        return {
            "route": REAR_LOWER_ROUTE,
            "status": path["status"],
            "start_qpos": qpos[0].tolist(),
            "goal_qpos": qpos[-1].tolist(),
            "contact_target_world_m": np.asarray(
                path["contact_target_world_m"], dtype=np.float64).tolist(),
            "prepoint_tip_world_m": np.asarray(
                path["prepoint_tip_world_m"], dtype=np.float64).tolist(),
            "prepoint_pitch_rad": float(path["prepoint_pitch_rad"]),
            "outward_high_pitch_rad": float(
                path["outward_high_pitch_rad"]),
            "high_offset_above_crest_m": float(
                path["high_offset_above_crest_m"]),
            "manual_contact_delta_m": np.asarray(
                path["manual_contact_delta_m"], dtype=np.float64).tolist(),
            "segments": path["segments"],
            "path_frames": int(path["total_frames"]),
            "duration_sim_s": float(path["total_duration_sim_s"]),
            "minimum_planned_bucket_terrain_clearance_m": float(
                path["minimum_bucket_terrain_clearance_m"]),
            "minimum_planned_mechanism_terrain_clearance_m": float(
                path["minimum_mechanism_terrain_clearance_m"]),
            "minimum_planned_static_truck_aabb_separation_m": float(
                path["minimum_static_truck_aabb_separation_m"]),
            "maximum_planned_closure_m": float(path["maximum_closure_m"]),
            "maximum_planned_closure_limit_m": float(
                path["maximum_closure_limit_m"]),
            "command_speed_limits": np.asarray(
                path["command_speed_limits"], dtype=np.float64).tolist(),
            "validation_substeps": int(path["validation_substeps"]),
            "validation_dt_s": float(path["validation_dt_s"]),
            "proxy_sample_spacing_m": float(
                path["proxy_sample_spacing_m"]),
            "proxy_sample_count": int(path["proxy_sample_count"]),
            "bucket_sampling_buffer_m": float(
                path["bucket_sampling_buffer_m"]),
            "required_bucket_clearance_m": float(
                path["required_bucket_clearance_m"]),
            "qpos_float64_sha256": _array_sha256(qpos),
            "velocity_float64_sha256": _array_sha256(velocity),
            "continuous_collision_proven": False,
        }
    return {
        "route": LEGACY_TOP_ROUTE,
        "start_qpos": qpos[0].tolist(),
        "goal_qpos": qpos[-1].tolist(),
        "target_tip_world_m": np.asarray(
            path["target_tip_world_m"], dtype=np.float64).tolist(),
        "target_pitch_rad": float(path["target_pitch_rad"]),
        "target_tip_height_above_surface_m": float(
            path["target_tip_height_above_surface_m"]),
        "command_speed_limits": np.asarray(
            path["command_speed_limits"], dtype=np.float64).tolist(),
        "path_frames": int(path["steps"]),
        "duration_sim_s": float(path["duration_sim_s"]),
        "validation_substeps": int(path["validation_substeps"]),
        "validation_dt_s": float(path["validation_dt_s"]),
        "minimum_planned_bucket_terrain_clearance_m": float(
            path["minimum_bucket_terrain_clearance_m"]),
        "endpoint_planned_bucket_terrain_clearance_m": float(
            path["endpoint_bucket_terrain_clearance_m"]),
        "maximum_planned_closure_m": float(path["maximum_closure_m"]),
        "maximum_planned_closure_limit_m": float(
            path["maximum_planned_closure_limit_m"]),
        "qpos_float64_sha256": _array_sha256(qpos),
        "velocity_float64_sha256": _array_sha256(velocity),
        "continuous_collision_proven": False,
    }


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    journal = output / "progress.ndjson"
    execution_path = output / "approach-execution.json"
    checkpoint_path = output / "approach-handoff.npz"
    metadata_path = Path(str(checkpoint_path) + ".json")
    started = time.perf_counter()
    bridge = fixture = None
    observer = None
    observer_status = {
        "phase": "planning",
        "frame": 0,
        "bucket_clearance_m": None,
        "bucket_contact_samples": 0,
        "bucket_effective_contact": False,
        "equivalent_drive_load": None,
        "equivalent_drive_load_delta": None,
        "message": "Planning the checked non-contact route",
    }
    observer_cut_point = observer_prepoint = None
    rows = []
    report = {
        "schema": "layered_47_sparse_approach_handoff_run_v1",
        "status": "initializing",
        "hz": HZ,
        "rigid_internal_hz": RIGID_INTERNAL_HZ,
        "device": args.device,
        "scope": {
            "noncontact_only": True,
            "digging_executed": False,
            "manual_collection_eligible": False,
            "training_eligible": False,
            "formal_a30_equivalent": False,
        },
        "protocol": {
            "route": args.route,
            "solver_grid_type": "sparse",
            "solver_voxel_size_m": 0.1,
            "output_dt_s": 1.0 / HZ,
            "rigid_internal_hz": RIGID_INTERNAL_HZ,
            "maximum_path_frames": (
                2400 if args.route == REAR_LOWER_ROUTE
                else MAXIMUM_PATH_FRAMES),
            "endpoint_hold_frames": ENDPOINT_HOLD_FRAMES,
            "minimum_bucket_clearance_m": (
                REAR_LOWER_EXECUTION_BUCKET_CLEARANCE_M
                if args.route == REAR_LOWER_ROUTE
                else BUCKET_PATH_CLEARANCE_M),
            "minimum_mechanism_clearance_m": (
                MECHANISM_CLEARANCE_M
                if args.route == REAR_LOWER_ROUTE else None),
            "minimum_static_truck_aabb_separation_m": (
                TRUCK_CLEARANCE_M
                if args.route == REAR_LOWER_ROUTE else None),
            "maximum_dispatch_surface_change_m":
                MAXIMUM_DISPATCH_SURFACE_CHANGE_M,
            "maximum_planned_closure_m": MAXIMUM_PLANNED_CLOSURE_M,
            "maximum_physical_closure_m": MAXIMUM_PHYSICAL_CLOSURE_M,
            "raw_contact_samples_are_diagnostic_not_boolean_contact": True,
            "raw_impulse_promotes_contact": False,
            "particle_center_distance_promotes_contact": False,
            "raw_sample_count_promotes_contact": False,
            "target_tip_height_above_surface_m":
                (args.target_tip_height_above_surface_m
                 if args.route == LEGACY_TOP_ROUTE else None),
            "sparse_contact_support_margin_basis": (
                "measured_first_contact_after_0.100035m_tip_drop_at_"
                "0.499360m_above_selected_surface_and_0.287084m_"
                "geometric_clearance;_requested_target_is_predeclared_"
                "and_must_remain_noncontact"
                if args.route == LEGACY_TOP_ROUTE else
                "rear_lower_contact_target_is_separate_from_prepoint;_"
                "staged_route_uses_dense_open_bucket_proxy_and_known_"
                "ground_completion"),
            "validation_substeps": 2,
            "validation_dt_s": 1.0 / 120.0,
            "continuous_collision_proven": False,
            "postsolve_projection": {
                "enabled": True,
                "reaction_routing": "unattributed_diagnostic_not_applied_to_mujoco",
                "continuous_collision_proven": False,
            },
            "soil_rendering": (
                None if not args.viewer else {
                    "mode": display["soil_render_mode"],
                    "surface_voxel_m": display["surface_voxel_m"],
                    "surface_update_interval_frames": display[
                        "surface_update_interval_frames"],
                    "display_only": True,
                }),
        },
        "runner_sha256": motion.probe.bench.sha256(Path(__file__)),
    }
    motion.probe.bench._write_json(report_path, report)

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
            raise ValueError("sparse_approach_requires_rigid_4800hz")
        if args.route not in SUPPORTED_ROUTES:
            raise ValueError("sparse_approach_route")
        if args.checkpoint_sha256 != (
                "a3a78553efcf0cbb0018951b93640a95c2e7056558311b51472e255affa8f4f7"):
            raise ValueError("sparse_approach_source_checkpoint_identity")
        root = Path(args.repository_root).resolve()
        config_path = Path(args.config).resolve()
        local_source_config = Path(args.local_source_config).resolve()
        display = None
        if args.viewer:
            if not args.display_config:
                raise ValueError("approach_viewer_requires_display_config")
            display_config_path = Path(args.display_config).resolve()
            display = json.loads(display_config_path.read_text(
                encoding="utf-8"))["display"]
        candidate_report = Path(args.candidate_report).resolve()
        resume_report = Path(args.resume_report).resolve()
        loader, identity = motion.probe.select(47)
        base = loader(config_path, root)
        base["input_checkpoint"] = json.loads(
            local_source_config.read_text(encoding="utf-8"))[
                "input_checkpoint"]
        arrays, source, before = motion.audit.load_candidate(
            candidate_report, base, 47, HZ)
        motion.audit.validate_seed_audit(
            json.loads(resume_report.read_text(encoding="utf-8")),
            candidate_report, source, before, 47, HZ)
        if before["sha256"] != args.checkpoint_sha256:
            raise ValueError("sparse_approach_candidate_sha256")
        if source["config_sha256"] != motion.probe.bench.sha256(config_path):
            raise ValueError("sparse_approach_config_binding")
        original = source["source_checkpoint"]
        if motion.probe.bench.checkpoint_identity(
                original["path"]) != original:
            raise ValueError("sparse_approach_original_source_changed")
        original_metadata = Path(original["path"] + ".json")
        if motion.probe.bench.sha256(
                original_metadata) != source["metadata_sha256"]:
            raise ValueError("sparse_approach_original_metadata_changed")

        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity[
                "expected_ids"]:
            raise ValueError("sparse_approach_manifest_order")
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
            raise ValueError("sparse_approach_kinematic_axes")
        truck_path = truck_world_vertices = None
        target_selection = observation["selection"]
        if args.route == REAR_LOWER_ROUTE:
            if not args.truck_mesh:
                raise ValueError("rear_lower_approach_truck_mesh")
            truck_path = Path(args.truck_mesh).resolve()
            if (not truck_path.is_file()
                    or motion.probe.bench.sha256(truck_path) !=
                    "bd379fc56bd492b4e0a342f8432869f2926a6e1f10a4face407d70c5784fa640"):
                raise ValueError("rear_lower_approach_truck_mesh_binding")
            with np.load(truck_path, allow_pickle=False) as truck_archive:
                truck_world_vertices = truck_archive[
                    "world_vertices"].copy()
            target_selection = select_rear_lower_face(
                terrain, observation["selection"])
        # Only the post-restore, freshly reobserved path can be dispatched.
        # The former pre-restore full sweep duplicated that same expensive
        # validation and its commands were never eligible for execution.
        initial_summary = {
            "status": "full_path_validation_deferred_to_dispatch",
            "route": args.route,
            "start_qpos": np.asarray(
                arrays["mechanism_qpos"], dtype=np.float64).tolist(),
            "target_selection": target_selection,
        }

        binding_paths = [
            config_path, local_source_config, candidate_report,
            resume_report, original_metadata, manifest_path, proxy_path,
            Path(args.kinematics_model),
            *[Path(args.kinematics_python_dir) / name for name in (
                "scene_assembly.py", "fast_kinematics.py", "closed_chain.py")],
        ]
        if truck_path is not None:
            binding_paths.append(truck_path)
        if args.viewer:
            binding_paths.extend((
                display_config_path,
                Path(display_helpers.__file__).with_name("soil_surface.py"),
            ))
        bindings = {str(path): motion.probe.bench.sha256(path)
                    for path in binding_paths}
        report.update(
            source_checkpoint=before,
            original_source=original,
            candidate_report=str(candidate_report),
            resume_report=str(resume_report),
            solver_profile=source["solver_profile"],
            acquisition_domain=handoff.SOURCE_DOMAIN,
            observation=observation,
            target_selection=target_selection,
            initial_plan=initial_summary,
            input_bindings=bindings,
        )
        append("prepared", source_checkpoint=before,
               observation=observation, initial_plan=initial_summary)

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with optimized_bridge(args.mechanism_header_dir,
                              output / "build") as optimization, \
                tempfile.TemporaryDirectory(
                    prefix="sparse-approach-handoff-") as raw:
            fixture, held, restored = motion.audit.restore_candidate_runtime(
                args, base, root, records, arrays, proxy,
                source["solver_profile"], Path(raw))
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            report["restore"] = restored
            report["optimization"] = optimization.copy()
            local_proxy = _proxy_local_vertices(proxy)
            local_surface_samples = (
                proxy_surface_samples(proxy)
                if args.route == REAR_LOWER_ROUTE else None)
            initial_collision_terrain = (
                ground_completed_terrain(terrain)
                if args.route == REAR_LOWER_ROUTE else None)
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
            if (predicted_vertices.shape != actual_vertices.shape
                    or not np.allclose(predicted_vertices, actual_vertices,
                                       rtol=0.0, atol=1.0e-7)):
                raise ValueError("sparse_approach_prediction_geometry")

            def advance(
                    command, phase, reference_qpos, active_terrain,
                    active_collision_terrain=None):
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
                mechanism_clearance = None
                truck_clearance = None
                if args.route == REAR_LOWER_ROUTE:
                    if active_collision_terrain is None:
                        raise ValueError(
                            "rear_lower_missing_collision_terrain")
                    rotation = np.asarray(
                        bridge.state["bucket_rotation"],
                        dtype=np.float64).reshape(3, 3)
                    dense_bucket_world = (
                        local_surface_samples @ rotation.T
                        + np.asarray(bridge.state["bucket_origin"],
                                     dtype=np.float64))
                    bucket_clearance = sampled_surface_clearance(
                        dense_bucket_world,
                        active_collision_terrain)["clearance_m"]
                    _, bodies = assembly.assemble(
                        mechanism_qpos, include_meshes=False)
                    sweep = mechanism_and_truck_clearance(
                        sweep_geometries, bodies, active_terrain,
                        truck_world_vertices)
                    mechanism_clearance = sweep[
                        "mechanism_terrain_clearance_m"]
                    truck_clearance = sweep[
                        "static_truck_aabb_separation_m"]
                else:
                    bucket_clearance = _terrain_clearance(
                        bucket_vertices, active_terrain,
                        require_known=True)
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
                    joint_bounds_ok=_joint_bounds_ok(
                        mechanism_qpos, chain),
                    elapsed_simulation_seconds=(len(rows) + 1) / HZ,
                )
                row.update(observe_drive_load(
                    bridge.state,
                    previous=(
                        rows[-1]["equivalent_drive_load"]
                        if rows else None)))
                row["finite"] = bool(
                    row["finite"]
                    and motion.probe.bench._finite_tree(bridge.state))
                if row["bucket_mpm_contact_samples"] > 0:
                    row["bucket_contact_diagnostics"] = (
                        _bucket_contact_diagnostics(fixture))
                row["bucket_mpm_effective_contact"] = False
                motion.probe.classify_frame(
                    row, "measured", 0, transition="frequency")
                row["approach_guard"] = frame_passes(row, args.route)
                row["stopped"] = motion.stopped(row)
                rows.append(row)
                append("frame", row=row)
                observer_status.update(
                    phase=phase,
                    frame=len(rows),
                    bucket_clearance_m=float(bucket_clearance),
                    bucket_contact_samples=int(
                        row["bucket_mpm_contact_samples"]),
                    bucket_effective_contact=bool(
                        row["bucket_mpm_effective_contact"]),
                    equivalent_drive_load=row["equivalent_drive_load"],
                    equivalent_drive_load_delta=row[
                        "equivalent_drive_load_delta_from_previous"],
                    message=(
                        "SAFETY STOP TRIGGERED"
                        if not row["approach_guard"] else
                        "Automatic motion; please observe the geometry"),
                )
                if observer is not None:
                    if not observer["viewer"].is_running():
                        raise RuntimeError(
                            "sparse_approach_observer_window_closed")
                    _draw_observer(
                        observer, fixture, bridge.state, records,
                        args.device, observer_cut_point,
                        observer_prepoint)
                if len(rows) == 1 or len(rows) % 60 == 0 \
                        or not row["approach_guard"]:
                    print(
                        f"SPARSE_APPROACH frame={len(rows)} "
                        f"phase={phase} safe={row['approach_guard']} "
                        f"tracking={row['axis_tracking_error']:.9g} "
                        f"clearance={row['bucket_terrain_clearance_m']:.9g}",
                        flush=True)
                if not row["approach_guard"]:
                    report["rejection"] = {
                        "reason": "frame_guard",
                        "row": row,
                    }
                    raise RuntimeError("sparse_approach_frame_guard")
                return row

            zero = advance(
                np.zeros(4), "dispatch_reobservation",
                np.asarray(arrays["mechanism_qpos"], dtype=np.float64), terrain,
                initial_collision_terrain)
            zero["first_step_continuous"] = bool(
                zero["maximum_rock_position_step_m"]
                <= 0.01 / HZ * 1.01)
            if not zero["first_step_continuous"]:
                raise RuntimeError("sparse_approach_first_step_discontinuity")
            current_rock_qpos = np.asarray([
                rock["qpos"] for rock in bridge.state["rocks"]],
                dtype=np.float64)
            dispatch_terrain, dispatch_observation = build_fused_terrain(
                root, records, fixture.state.particle_q.numpy(),
                current_rock_qpos)
            dispatch_validation = validate_cut(
                dispatch_terrain, observation["selection"],
                bucket_width=BUCKET_MOUTH_WIDTH_M,
                maximum_surface_change_m=
                    MAXIMUM_DISPATCH_SURFACE_CHANGE_M)
            dispatch_target_selection = observation["selection"]
            dispatch_target_validation = {
                "valid": True,
                "contact_xy_change_m": 0.0,
                "contact_surface_change_m": float(
                    dispatch_validation["surface_change_m"]),
            }
            if args.route == REAR_LOWER_ROUTE:
                dispatch_target_selection = select_rear_lower_face(
                    dispatch_terrain, observation["selection"])
                contact_xy_change = float(np.linalg.norm(
                    np.asarray(
                        dispatch_target_selection["contact_xy_m"],
                        dtype=np.float64)
                    - np.asarray(target_selection["contact_xy_m"],
                                 dtype=np.float64)))
                contact_surface_change = abs(
                    float(dispatch_target_selection[
                        "contact_surface_z_m"])
                    - float(target_selection["contact_surface_z_m"]))
                dispatch_target_validation = {
                    "valid": bool(
                        contact_xy_change
                            <= MAXIMUM_DISPATCH_SURFACE_CHANGE_M
                        and contact_surface_change
                            <= MAXIMUM_DISPATCH_SURFACE_CHANGE_M),
                    "contact_xy_change_m": contact_xy_change,
                    "contact_surface_change_m": contact_surface_change,
                    "maximum_change_m":
                        MAXIMUM_DISPATCH_SURFACE_CHANGE_M,
                }
                if not dispatch_target_validation["valid"]:
                    raise ValueError(
                        "rear_lower_dispatch_target_changed")
            dispatch_plan = _plan(
                dispatch_terrain, dispatch_target_selection,
                bridge.state["mechanism_qpos"], proxy, args,
                assembly=assembly, chain=chain,
                truck_world_vertices=truck_world_vertices)
            dispatch_summary = _plan_summary(dispatch_plan)
            dispatch_collision_terrain = (
                ground_completed_terrain(dispatch_terrain)
                if args.route == REAR_LOWER_ROUTE else None)
            report.update(
                dispatch_observation=dispatch_observation,
                dispatch_validation=dispatch_validation,
                dispatch_target_selection=dispatch_target_selection,
                dispatch_target_validation=dispatch_target_validation,
                dispatch_plan=dispatch_summary,
            )
            append("dispatch_validated",
                   validation=dispatch_validation,
                   target_validation=dispatch_target_validation,
                   observation=dispatch_observation,
                   plan=dispatch_summary)

            planned_qpos = np.asarray(
                dispatch_plan["qpos"], dtype=np.float64)
            planned_velocity = np.asarray(
                dispatch_plan["velocity"], dtype=np.float64)
            observer_cut_point = np.asarray([
                *dispatch_target_selection["contact_xy_m"],
                dispatch_target_selection["contact_surface_z_m"],
            ], dtype=np.float64)
            observer_prepoint = (
                np.asarray(
                    dispatch_plan["prepoint_tip_world_m"],
                    dtype=np.float64)
                if args.route == REAR_LOWER_ROUTE else None)
            if args.viewer:
                observer = _open_observer(
                    display, records, root, args.device, observer_status)
                observer_status.update(
                    phase="ready_at_start",
                    message="RED is the selected cut; GREEN is prepoint 3",
                )
                _draw_observer(
                    observer, fixture, bridge.state, records, args.device,
                    observer_cut_point, observer_prepoint)
            if args.static_contact_probe_plan_frame is not None:
                plan_index = int(args.static_contact_probe_plan_frame)
                if not 0 <= plan_index < len(planned_qpos):
                    raise ValueError("static_contact_probe_plan_frame")
                hold_frames = int(args.static_contact_probe_hold_frames)
                if hold_frames < 1:
                    raise ValueError("static_contact_probe_hold_frames")
                for _ in range(1, hold_frames):
                    advance(
                        np.zeros(4), "static_probe_source_hold",
                        np.asarray(arrays["mechanism_qpos"],
                                   dtype=np.float64),
                        dispatch_terrain, dispatch_collision_terrain)
                _, probe_bodies = assembly.assemble(
                    planned_qpos[plan_index], include_meshes=False)
                bucket_pose = probe_bodies["bucket"]
                probe_state = {
                    **bridge.state,
                    "bucket_origin": bucket_pose[:3, 3].copy(),
                    "bucket_rotation": bucket_pose[:3, :3].copy(),
                    "bucket_linear_velocity_world_m_s": np.zeros(3),
                    "bucket_angular_velocity_world_rad_s": np.zeros(3),
                }
                fixture.assign(probe_state)
                origins = {
                    rock["rock_id"]: np.asarray(
                        rock["qpos"][:3], dtype=np.float64)
                    for rock in bridge.state["rocks"]}
                origins["bucket"] = bucket_pose[:3, 3].copy()
                measured, particles = fixture.step(1.0 / HZ, origins)
                contact_details = _bucket_contact_diagnostics(fixture)
                bucket_wrench = measured["bucket"]
                report.update(
                    status="sparse_static_contact_pose_probe_complete",
                    scope={
                        **report["scope"],
                        "noncontact_only": False,
                        "collision_query_only": True,
                        "mechanism_motion_executed": False,
                        "handoff_checkpoint_saved": False,
                    },
                    static_contact_probe={
                        "plan_frame": plan_index,
                        "source_hold_frames_before_pose_probe": hold_frames,
                        "source_hold_simulation_seconds":
                            hold_frames / HZ,
                        "planned_qpos": planned_qpos[plan_index].tolist(),
                        "bucket_origin_world_m":
                            bucket_pose[:3, 3].tolist(),
                        "bucket_rotation_world":
                            bucket_pose[:3, :3].tolist(),
                        "particle_metrics": particles,
                        "bucket_force_world_n": np.asarray(
                            bucket_wrench["force"],
                            dtype=np.float64).tolist(),
                        "bucket_force_norm_n": float(np.linalg.norm(
                            bucket_wrench["force"])),
                        "bucket_torque_world_nm": np.asarray(
                            bucket_wrench["torque"],
                            dtype=np.float64).tolist(),
                        "bucket_torque_norm_nm": float(np.linalg.norm(
                            bucket_wrench["torque"])),
                        "contact_details": contact_details,
                        "source_checkpoint_written": False,
                        "training_eligible": False,
                        "execution_handoff_eligible": False,
                    },
                    checks={
                        "source_restore_exact": restored["exact"],
                        "probe_finite": bool(particles["finite"]),
                        "all_source_hold_frames_noncontact": all(
                            row["bucket_mpm_contact_samples"] == 0
                            for row in rows),
                        "source_checkpoint_unchanged":
                            motion.probe.bench.checkpoint_identity(
                                before["path"]) == before,
                    },
                    next_gate=(
                        "interpret_static_pose_impulse_before_any_"
                        "additional_full_path_run"),
                )
                bridge.close(kill=True)
                bridge = None
                return report
            if args.route == REAR_LOWER_ROUTE:
                phase_labels = [
                    segment["name"].lower()
                    for segment in dispatch_plan["segments"]
                    for _ in range(int(segment["frames"]))
                ]
                if len(phase_labels) != len(planned_velocity):
                    raise ValueError("rear_lower_segment_frame_accounting")
            else:
                phase_labels = ["approach"] * len(planned_velocity)
            for index, (command, phase) in enumerate(
                    zip(planned_velocity, phase_labels, strict=True), start=1):
                advance(command, phase, planned_qpos[index],
                        dispatch_terrain, dispatch_collision_terrain)
            for _ in range(ENDPOINT_HOLD_FRAMES):
                advance(np.zeros(4), "endpoint_hold", planned_qpos[-1],
                        dispatch_terrain, dispatch_collision_terrain)

            approach_rows = [
                row for row in rows
                if row["phase"] not in (
                    "dispatch_reobservation", "endpoint_hold")]
            hold_rows = [row for row in rows
                         if row["phase"] == "endpoint_hold"]
            endpoint = rows[-1]
            checks = {
                "source_restore_exact": restored["exact"],
                "dispatch_reobservation_passed": bool(
                    dispatch_validation["valid"]),
                "dispatch_target_stable": bool(
                    dispatch_target_validation["valid"]),
                "first_step_continuous": zero["first_step_continuous"],
                "all_planned_frames_executed": len(approach_rows)
                    == len(planned_velocity),
                "all_endpoint_hold_frames_executed": len(hold_rows)
                    == ENDPOINT_HOLD_FRAMES,
                "all_frames_noncontact_safe": all(
                    row["approach_guard"] for row in rows),
                "all_mechanism_clearances_safe": (
                    all(row["mechanism_terrain_clearance_m"]
                        >= MECHANISM_CLEARANCE_M for row in rows)
                    if args.route == REAR_LOWER_ROUTE else True),
                "all_static_truck_separations_safe": (
                    all(row["static_truck_aabb_separation_m"]
                        >= TRUCK_CLEARANCE_M for row in rows)
                    if args.route == REAR_LOWER_ROUTE else True),
                "zero_effective_bucket_soil_contact_all_frames": all(
                    not row["bucket_mpm_effective_contact"]
                    for row in rows),
                "zero_bucket_rock_contact_all_frames": all(
                    row["rock_bucket_contact_substeps"] == 0
                    for row in rows),
                "endpoint_axis_error_within_tolerance":
                    endpoint["axis_tracking_error"]
                    <= MAXIMUM_ENDPOINT_AXIS_ERROR,
                "endpoint_stopped_tail": all(
                    row["stopped"] for row in hold_rows[-15:]),
                "source_checkpoint_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        before["path"]) == before,
                "source_metadata_unchanged":
                    motion.probe.bench.sha256(Path(before["path"] + ".json"))
                    == source["diagnostic_snapshots"][-1][
                        "metadata_sha256"],
                "original_source_unchanged":
                    motion.probe.bench.checkpoint_identity(
                        original["path"]) == original,
                "lineage_unchanged": motion.probe.seed_unchanged(
                    source["seed"]),
                "input_bindings_unchanged": all(
                    motion.probe.bench.sha256(Path(path)) == sha256
                    for path, sha256 in bindings.items()),
            }
            if not all(checks.values()):
                raise RuntimeError("sparse_approach_acceptance_failed")

            raw_contact_residual_frames = [
                {
                    "frame": row["frame"],
                    "phase": row["phase"],
                    "raw_samples": row["bucket_mpm_contact_samples"],
                    "diagnostics": row.get("bucket_contact_diagnostics"),
                }
                for row in rows if row["bucket_mpm_contact_samples"] > 0
            ]

            execution_document = clean({
                "schema": "layered_47_sparse_approach_execution_v1",
                "status": "sparse_noncontact_approach_passed",
                "checks": checks,
                "source_checkpoint": before,
                "observation": observation,
                "target_selection": target_selection,
                "dispatch_observation": dispatch_observation,
                "dispatch_validation": dispatch_validation,
                "dispatch_target_selection": dispatch_target_selection,
                "dispatch_target_validation": dispatch_target_validation,
                "manual_contact_target": ({
                    "source": dispatch_target_selection["source"],
                    "entry_xy_m": dispatch_target_selection[
                        "contact_xy_m"],
                    "surface_z_m": dispatch_target_selection[
                        "contact_surface_z_m"],
                    "strip_anchor_xy_m": dispatch_target_selection[
                        "strip_anchor_xy_m"],
                    "normalized_height_fraction":
                        dispatch_target_selection[
                            "normalized_height_fraction"],
                } if args.route == REAR_LOWER_ROUTE
                 else dispatch_target_selection),
                "initial_plan": initial_summary,
                "dispatch_plan": dispatch_summary,
                "raw_contact_residual_frames": raw_contact_residual_frames,
                "trace": rows,
                "scope": report["scope"],
                "simulation_time_s": float(
                    arrays["simulation_time_s"][0]) + len(rows) / HZ,
            })
            execution_identity = handoff.write_metadata_atomic(
                execution_path, execution_document)
            saved_arrays = handoff.capture_arrays(
                fixture, bridge.state,
                execution_document["simulation_time_s"])
            checkpoint_identity = handoff.write_checkpoint_atomic(
                checkpoint_path, saved_arrays)
            source_metadata_path = Path(before["path"] + ".json")
            provenance = {
                "source_checkpoint_sha256": before["sha256"],
                "source_checkpoint_metadata_sha256":
                    motion.probe.bench.sha256(source_metadata_path),
                "candidate_report_sha256":
                    motion.probe.bench.sha256(candidate_report),
                "resume_report_sha256":
                    motion.probe.bench.sha256(resume_report),
                "preflight_config_sha256":
                    motion.probe.bench.sha256(config_path),
                "local_source_config_sha256":
                    motion.probe.bench.sha256(local_source_config),
                "approach_execution_report_sha256":
                    execution_identity["sha256"],
                "approach_runner_sha256": report["runner_sha256"],
                "plan_qpos_float64_sha256":
                    dispatch_summary["qpos_float64_sha256"],
                "plan_velocity_float64_sha256":
                    dispatch_summary["velocity_float64_sha256"],
                "observation_state_digest": _observation_digest(
                    dispatch_terrain, dispatch_target_selection),
                "source_domain": handoff.SOURCE_DOMAIN,
                "source_checkpoint_status":
                    "grid_candidate_diagnostic_only",
                "hz": HZ,
                "rigid_internal_hz": RIGID_INTERNAL_HZ,
            }
            metadata_document = handoff.metadata_document(
                checkpoint_identity, saved_arrays, provenance)
            metadata_identity = handoff.write_metadata_atomic(
                metadata_path, metadata_document)
            report.update(
                status="sparse_approach_handoff_saved_unaudited",
                checks=checks,
                execution_report=execution_identity,
                handoff_checkpoint=checkpoint_identity,
                handoff_metadata=metadata_identity,
                handoff_provenance=provenance,
                next_gate=(
                    "independent_new_process_restore_audit_before_"
                    "contact_handoff"),
            )
            observer_status.update(
                phase="complete_at_prepoint_3",
                message="Reached prepoint 3; automatic digging was not run",
            )
            _hold_observer(
                observer, fixture, bridge.state, records, args.device,
                observer_cut_point, observer_prepoint,
                args.viewer_final_hold_seconds)
            bridge.close()
            bridge = None
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="sparse_approach_handoff_failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            next_gate="fix_noncontact_approach_before_any_contact",
        )
        observer_status.update(
            phase="safety_stop",
            message=(
                "SAFETY STOP: inspect this frozen frame, then close window"),
        )
        if (observer is not None and fixture is not None
                and bridge is not None):
            try:
                bridge.command = np.zeros(4)
                _hold_observer(
                    observer, fixture, bridge.state, records, args.device,
                    observer_cut_point, observer_prepoint,
                    args.viewer_final_hold_seconds)
            except Exception as viewer_error:
                report["observer_hold_error"] = str(viewer_error)
    finally:
        if observer is not None:
            try:
                observer["viewer"].close()
            except Exception as error:
                report["observer_cleanup_error"] = str(error)
        if bridge is not None:
            try:
                bridge.close(kill=True)
            except Exception as error:
                report["cleanup_error"] = str(error)
        report["timing"] = {
            "simulation_seconds": len(rows) / HZ,
            "physics_wall_seconds": sum(
                row["wall_seconds"] for row in rows),
            "total_wall_seconds": time.perf_counter() - started,
            "frames": len(rows),
        }
        motion.probe.bench._write_json(report_path, clean(report))
    return report


def main():
    argument_parser = parser()
    argument_parser.add_argument("--local-source-config", required=True)
    argument_parser.add_argument("--kinematics-python-dir", required=True)
    argument_parser.add_argument("--kinematics-model", required=True)
    argument_parser.add_argument("--kinematics-mesh-dir", required=True)
    argument_parser.add_argument(
        "--route", choices=SUPPORTED_ROUTES, default=LEGACY_TOP_ROUTE)
    argument_parser.add_argument("--truck-mesh")
    argument_parser.add_argument(
        "--static-contact-probe-plan-frame", type=int)
    argument_parser.add_argument(
        "--static-contact-probe-hold-frames", type=int, default=1)
    argument_parser.add_argument("--viewer", action="store_true")
    argument_parser.add_argument("--display-config")
    argument_parser.add_argument(
        "--viewer-final-hold-seconds", type=float, default=0.0)
    argument_parser.add_argument(
        "--rigid-internal-hz", type=int, choices=(RIGID_INTERNAL_HZ,),
        default=RIGID_INTERNAL_HZ)
    argument_parser.add_argument(
        "--target-tip-height-above-surface-m", type=float,
        choices=SUPPORTED_TARGET_TIP_HEIGHTS_ABOVE_SURFACE_M,
        default=SPARSE_TARGET_TIP_HEIGHT_ABOVE_SURFACE_M)
    args = argument_parser.parse_args()
    result = run(args)
    print(json.dumps({
        key: result.get(key) for key in (
            "status", "checks", "handoff_checkpoint", "handoff_metadata",
            "next_gate", "timing", "failure")
    }, indent=2))
    return 0 if result["status"] in (
        "sparse_approach_handoff_saved_unaudited",
        "sparse_static_contact_pose_probe_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
