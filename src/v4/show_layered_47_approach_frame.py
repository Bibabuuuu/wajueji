#!/usr/bin/env python3
"""Show a read-only reconstruction of one logged sparse-approach frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import newton
import numpy as np
import warp as wp

import run_layered_47_interactive_proxy_practice as display_helpers
import run_layered_47_sparse_approach_handoff as approach
import run_layered_47_sparse_motion as motion


def _logged_frame(progress_path: Path, frame: int) -> dict:
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        row = value.get("row")
        if value.get("event") == "frame" and row.get("frame") == frame:
            return row
    raise ValueError("approach_reconstruction_frame_not_found")


def _planned_closed_chain_qpos(report: dict, source_qpos: np.ndarray,
                               plan_index: int, chain) -> np.ndarray:
    """Reproduce the validated full-qpos path through the requested index."""
    if plan_index < 0:
        raise ValueError("approach_reconstruction_negative_plan_index")
    current = np.asarray(source_qpos, dtype=np.float64).copy()
    chain.q = current.copy()
    if plan_index == 0:
        return current
    generated = 0
    dt = 1.0 / approach.HZ
    limits = np.asarray(
        report["dispatch_plan"]["command_speed_limits"],
        dtype=np.float64)
    for segment in report["dispatch_plan"]["segments"]:
        start = current.copy()
        goal = np.asarray(segment["goal_qpos"], dtype=np.float64)
        steps = int(segment["frames"])
        delta = goal[chain.axes] - start[chain.axes]
        for index in range(1, steps + 1):
            s = index / steps
            desired = start[chain.axes] + (
                3.0 * s * s - 2.0 * s * s * s) * delta
            velocity = (desired - current[chain.axes]) / dt
            if np.any(np.abs(velocity) > limits + 1.0e-10):
                raise ValueError("approach_reconstruction_speed_budget")
            chain.q = current.copy()
            current = chain.advance(velocity, dt / 2.0)[0]
            current = chain.advance(velocity, dt / 2.0)[0]
            generated += 1
            if generated == plan_index:
                return current
        if not np.allclose(current, goal, rtol=0.0, atol=1.0e-5):
            raise ValueError("approach_reconstruction_segment_endpoint")
    raise ValueError("approach_reconstruction_plan_index")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--display-config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--kinematics-python-dir", required=True)
    parser.add_argument("--kinematics-model", required=True)
    parser.add_argument("--kinematics-mesh-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    row = _logged_frame(Path(args.progress), args.frame)
    display = json.loads(Path(args.display_config).read_text(
        encoding="utf-8"))["display"]
    root = Path(args.repository_root).resolve()
    manifest_paths = [
        Path(path) for path in report["input_bindings"]
        if path.endswith("scene-manifest.json")]
    if len(manifest_paths) != 1:
        raise ValueError("approach_reconstruction_manifest_binding")
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    _, identity = motion.probe.select(47)
    records = [record for record in manifest["rocks"]
               if record["layer_id"] in identity["layers"]]
    source_path = Path(report["source_checkpoint"]["path"])
    with np.load(source_path, allow_pickle=False) as archive:
        particles = archive["particle_q"].copy()
        rock_qpos = archive["rigid_qpos"].copy()
        source_mechanism_qpos = archive["mechanism_qpos"].copy()
    _, chain = approach._load_kinematics(
        args.kinematics_python_dir, args.kinematics_model,
        args.kinematics_mesh_dir)
    # Logged frame 1 is the zero-command dispatch reobservation.  Frame 2 is
    # planned path index 1, hence physical frame N maps to plan index N-1.
    mechanism_qpos = _planned_closed_chain_qpos(
        report, source_mechanism_qpos, args.frame - 1, chain)
    recorded_axes = np.asarray(row["axis_positions"], dtype=np.float64)
    active_axis_error = float(np.max(np.abs(
        mechanism_qpos[motion.AXES] - recorded_axes)))
    closure = float(np.max(np.abs(chain.residual(mechanism_qpos))))
    if (not math.isfinite(active_axis_error)
            or active_axis_error > approach.MAXIMUM_PATH_AXIS_TRACKING_ERROR
            or not math.isfinite(closure)
            or closure > approach.MAXIMUM_PLANNED_CLOSURE_M):
        raise ValueError("approach_reconstruction_kinematic_validation")
    cut = np.asarray([
        *report["dispatch_target_selection"]["contact_xy_m"],
        report["dispatch_target_selection"]["contact_surface_z_m"],
    ], dtype=np.float64)
    prepoint = np.asarray(
        report["dispatch_plan"]["prepoint_tip_world_m"], dtype=np.float64)
    raw_nodes = np.asarray(
        row.get("bucket_contact_diagnostics", {}).get(
            "positions_world_m", []), dtype=np.float32).reshape(-1, 3)

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    assembly, model, state, names, to_pose = (
        display_helpers._display_runtime(display))
    poses = assembly.assemble(
        mechanism_qpos, include_meshes=False)[1]
    state.body_q.assign(wp.array(
        [to_pose(poses[name]) for name in names],
        dtype=wp.transform, device=args.device))
    viewer = newton.viewer.ViewerGL(
        width=display["width"], height=display["height"],
        vsync=False, paused=True)
    viewer.set_model(model)
    viewer.picking_enabled = False
    viewer.renderer.set_title(
        f"CLOSED-CHAIN RECONSTRUCTION FRAME {args.frame} - READ ONLY")
    viewer.set_camera(
        wp.vec3(*display["camera_position_m"]),
        pitch=display["camera_pitch_deg"],
        yaw=display["camera_yaw_deg"])
    display_helpers._prepare_rock_visuals(
        viewer, records, root, args.device)

    diagnostics = row.get("bucket_contact_diagnostics", {})

    def ui(imgui):
        imgui.text(f"CLOSED-CHAIN FRAME {args.frame} / READ ONLY")
        imgui.text("Mechanism: validated planned full 19-DoF pose")
        imgui.text(
            f"Logged actuator tracking difference: {active_axis_error:.3e}")
        imgui.text(f"Closed-chain residual: {closure:.3e} m")
        imgui.text("Raw MPM nodes: exact logged frame")
        imgui.text("Soil and rocks: source snapshot (not evolved frame state)")
        imgui.text("RED = selected cut | GREEN = prepoint 3")
        imgui.text("MAGENTA = raw MPM grid nodes with residual impulse")
        imgui.text(
            f"Raw samples: {row['bucket_mpm_contact_samples']} | "
            f"effective: {row.get('bucket_mpm_effective_contact')}")
        imgui.text(
            "Maximum impulse: "
            f"{diagnostics.get('maximum_impulse_norm_ns', 0.0):.3e} N*s")
        particle_distance = diagnostics.get(
            "minimum_particle_center_distance_m")
        if particle_distance is not None:
            imgui.text(
                "Nearest particle-center distance (legacy diagnostic): "
                f"{particle_distance:.3f} m")
        else:
            imgui.text(
                "Particle-center proximity is not a contact criterion")

    viewer.register_ui_callback(ui, position="side")
    stride = int(display["particle_display_stride"])
    display_particles = particles[::stride]
    rock_groups = {
        prototype: [index for index, record in enumerate(records)
                    if record["prototype_id"] == prototype]
        for prototype in sorted({record["prototype_id"]
                                 for record in records})
    }
    try:
        while viewer.is_running():
            viewer.begin_frame(args.frame / approach.HZ)
            viewer.log_state(state)
            viewer.log_points(
                "/reconstruction/soil",
                wp.array(display_particles, dtype=wp.vec3,
                         device=args.device),
                radii=float(display["particle_radius_m"]),
                colors=wp.full(
                    len(display_particles), wp.vec3(0.58, 0.40, 0.23),
                    dtype=wp.vec3, device=args.device))
            for prototype, indices in rock_groups.items():
                xforms = [display_helpers._make_transform(
                    rock_qpos[index, :3], rock_qpos[index, 3:7])
                    for index in indices]
                scales = [wp.vec3(float(
                    records[index]["equivalent_diameter_m"]))
                    for index in indices]
                viewer.log_instances(
                    f"/reconstruction/rocks/{prototype}",
                    f"/practice/prototype/{prototype}",
                    wp.array(xforms, dtype=wp.transform, device=args.device),
                    wp.array(scales, dtype=wp.vec3, device=args.device),
                    wp.full(
                        len(indices), wp.vec3(0.40, 0.37, 0.32),
                        dtype=wp.vec3, device=args.device),
                    wp.full(
                        len(indices), wp.vec4(0.92, 0.0, 0.0, 1.0),
                        dtype=wp.vec4, device=args.device))
            for name, point, color in (
                    ("selected_cut", cut, wp.vec3(1.0, 0.08, 0.05)),
                    ("prepoint_3", prepoint, wp.vec3(0.10, 1.0, 0.10))):
                marker = approach._marker_column(point)
                viewer.log_points(
                    f"/reconstruction/markers/{name}",
                    wp.array(marker, dtype=wp.vec3, device=args.device),
                    radii=0.035,
                    colors=wp.full(
                        len(marker), color, dtype=wp.vec3,
                        device=args.device))
            if len(raw_nodes):
                viewer.log_points(
                    "/reconstruction/raw_mpm_nodes",
                    wp.array(raw_nodes, dtype=wp.vec3, device=args.device),
                    radii=0.025,
                    colors=wp.full(
                        len(raw_nodes), wp.vec3(1.0, 0.05, 0.75),
                        dtype=wp.vec3, device=args.device))
            viewer.end_frame()
            time.sleep(1.0 / 30.0)
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
