"""Read-only staged planning for a rear lower-face manual handoff.

The contact target remains an observation result.  This module only selects a
nearby non-contact prepoint and checks a high-clear -> outward-high -> lower
path.  It never steps physics and never creates excavation commands.
"""

from __future__ import annotations

import math

import numpy as np

from mixed_material_r0 import quaternion_matrix_wxyz
from planning.approach import sampled_approach
from planning.cycle_geometry import TIP, solve_pose


BUCKET_CLEARANCE_M = 0.15
BUCKET_SAMPLING_BUFFER_M = 0.02
SPARSE_MPM_CONTACT_SUPPORT_CLEARANCE_M = 0.30
REQUIRED_BUCKET_CLEARANCE_M = (
    SPARSE_MPM_CONTACT_SUPPORT_CLEARANCE_M + BUCKET_SAMPLING_BUFFER_M)
MECHANISM_CLEARANCE_M = 0.02
TRUCK_CLEARANCE_M = 0.02
PROXY_SAMPLE_SPACING_M = 0.025
KINEMATIC_COMMAND_LIMITS = np.asarray([0.12, 0.0225, 0.0225, 0.0225])


def ground_completed_terrain(terrain: dict, *, ground_height_m: float = 0.0):
    """Fuse the known rigid ground plane under an observed material surface."""
    height = np.asarray(terrain["height_m"], dtype=np.float64)
    count = np.asarray(terrain["point_count"], dtype=np.int64)
    if (not terrain.get("valid") or height.ndim != 2
            or count.shape != height.shape
            or not math.isfinite(ground_height_m)):
        raise ValueError("rear_prepoint_terrain_contract")
    completed = np.where(
        np.isfinite(height), np.maximum(height, ground_height_m),
        ground_height_m)
    return {
        **terrain,
        "height_m": completed,
        "point_count": np.maximum(count, 1),
        "ground_completion": {
            "height_m": float(ground_height_m),
            "basis": "known_rigid_ground_plane",
            "material_observation_unchanged": True,
        },
    }


def _axis_samples(lower: float, upper: float, spacing: float) -> np.ndarray:
    count = max(2, int(math.ceil((upper - lower) / spacing)) + 1)
    return np.linspace(lower, upper, count)


def _box_surface_samples(box: dict, spacing_m: float) -> np.ndarray:
    half = np.asarray(box["half_size_m"], dtype=np.float64)
    if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise ValueError("rear_prepoint_box_contract")
    samples = []
    for fixed_axis in range(3):
        free_axes = [axis for axis in range(3) if axis != fixed_axis]
        for sign in (-1.0, 1.0):
            for first in _axis_samples(
                    -half[free_axes[0]], half[free_axes[0]], spacing_m):
                for second in _axis_samples(
                        -half[free_axes[1]], half[free_axes[1]], spacing_m):
                    point = np.zeros(3, dtype=np.float64)
                    point[fixed_axis] = sign * half[fixed_axis]
                    point[free_axes] = [first, second]
                    samples.append(point)
    local = np.unique(np.round(samples, decimals=12), axis=0)
    rotation = quaternion_matrix_wxyz(box["quaternion_wxyz"])
    return (local @ rotation.T
            + np.asarray(box["position_bucket_m"], dtype=np.float64))


def _triangle_surface_samples(
        vertices: np.ndarray, faces: np.ndarray, spacing_m: float) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.isfinite(values).all()
            or np.any(triangles < 0) or np.any(triangles >= len(values))):
        raise ValueError("rear_prepoint_triangle_contract")
    samples = []
    for face in triangles:
        triangle = values[face]
        longest = max(
            np.linalg.norm(triangle[(index + 1) % 3] - triangle[index])
            for index in range(3))
        divisions = max(1, int(math.ceil(longest / spacing_m)))
        for first in range(divisions + 1):
            for second in range(divisions + 1 - first):
                samples.append(
                    triangle[0]
                    + first / divisions * (triangle[1] - triangle[0])
                    + second / divisions * (triangle[2] - triangle[0]))
    return np.asarray(samples, dtype=np.float64)


def proxy_surface_samples(
        proxy: dict,
        *,
        maximum_spacing_m: float = PROXY_SAMPLE_SPACING_M) -> np.ndarray:
    """Densely sample every solid face of the segmented open-bucket proxy."""
    if (not math.isfinite(maximum_spacing_m)
            or maximum_spacing_m <= 0.0):
        raise ValueError("rear_prepoint_sample_spacing_contract")
    parts = [
        _box_surface_samples(box, maximum_spacing_m)
        for box in proxy["shell_boxes"] + proxy["tooth_boxes"]
    ]
    parts.extend(
        _triangle_surface_samples(
            side["vertices_bucket_m"], side["faces"], maximum_spacing_m)
        for side in proxy["side_plate_meshes"])
    if not parts:
        raise ValueError("rear_prepoint_empty_proxy")
    return np.concatenate(parts, axis=0)


def sampled_surface_clearance(
        world_samples: np.ndarray, terrain: dict) -> dict:
    """Clearance of dense proxy samples against conservative cell maxima."""
    points = np.asarray(world_samples, dtype=np.float64)
    height = np.asarray(terrain["height_m"], dtype=np.float64)
    count = np.asarray(terrain["point_count"], dtype=np.int64)
    origin = np.asarray(terrain["origin_xy_m"], dtype=np.float64)
    resolution = float(terrain["resolution_m"])
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all() or height.ndim != 2
            or count.shape != height.shape or origin.shape != (2,)
            or not np.isfinite(height).all() or np.any(count < 1)
            or not math.isfinite(resolution) or resolution <= 0.0):
        raise ValueError("rear_prepoint_clearance_contract")
    cell = np.floor((points[:, :2] - origin) / resolution).astype(int)
    inside = ((cell[:, 0] >= 0) & (cell[:, 0] < height.shape[1])
              & (cell[:, 1] >= 0) & (cell[:, 1] < height.shape[0]))
    if not inside.all():
        raise ValueError("rear_prepoint_proxy_outside_observed_roi")
    surface = height[cell[:, 1], cell[:, 0]]
    clearance = points[:, 2] - surface
    index = int(np.argmin(clearance))
    return {
        "clearance_m": float(clearance[index]),
        "proxy_point_world_m": points[index].tolist(),
        "cell_surface_z_m": float(surface[index]),
    }


def _aabb_separation(
        first_lower: np.ndarray,
        first_upper: np.ndarray,
        second_lower: np.ndarray,
        second_upper: np.ndarray) -> float:
    gap = np.maximum(
        np.maximum(second_lower - first_upper, first_lower - second_upper),
        0.0)
    return float(np.linalg.norm(gap))


def _local_aabb_corners(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    return np.asarray([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ], dtype=np.float64)


def mechanism_and_truck_clearance(
        geometries: list[dict],
        bodies: dict,
        terrain: dict,
        truck_world_vertices: np.ndarray) -> dict:
    """Conservative moving-link terrain and full-truck AABB separation."""
    truck = np.asarray(truck_world_vertices, dtype=np.float64)
    if (truck.ndim != 2 or truck.shape[1] != 3 or not len(truck)
            or not np.isfinite(truck).all()):
        raise ValueError("rear_prepoint_truck_geometry_contract")
    truck_lower = truck.min(axis=0)
    truck_upper = truck.max(axis=0)
    mechanism_clearance = math.inf
    truck_clearance = math.inf
    for geometry in geometries:
        pose = np.asarray(bodies[geometry["body"]], dtype=np.float64)
        # A transformed local AABB encloses every transformed mesh vertex.
        # Using its eight corners for the truck test is therefore conservative
        # (never a larger separation) and avoids transforming the 58k-vertex
        # full mechanism mesh at every 120 Hz validation sample.
        local_corners = geometry.get("_body_local_aabb_corners")
        if local_corners is None:
            local_corners = _local_aabb_corners(
                geometry["body_local_vertices"])
            geometry["_body_local_aabb_corners"] = local_corners
        world_corners = (
            local_corners @ pose[:3, :3].T
            + pose[:3, 3])
        truck_clearance = min(
            truck_clearance,
            _aabb_separation(
                world_corners.min(axis=0), world_corners.max(axis=0),
                truck_lower, truck_upper))
        if geometry["mesh"] in ("bucket", "caterpillar", "chassis"):
            continue
        world = (
            geometry["body_local_vertices"] @ pose[:3, :3].T
            + pose[:3, 3])
        observed_clearance = _material_clearance(world, terrain)
        if observed_clearance is not None and observed_clearance < MECHANISM_CLEARANCE_M:
            # A link-wide world AABB couples the low hinge at one end with
            # soil beneath the high end. Refine using the enclosing rotated
            # local box against every observed solid terrain column. Both
            # shapes still enclose their geometry; no clearance is waived.
            observed_clearance = max(observed_clearance, oriented_terrain_separation(
                world_corners, pose[:3, :3], terrain))
        mechanism_clearance = min(
            mechanism_clearance, float(world[:, 2].min()))
        if observed_clearance is not None:
            mechanism_clearance = min(
                mechanism_clearance, observed_clearance)
    if math.isinf(mechanism_clearance) or math.isinf(truck_clearance):
        raise ValueError("rear_prepoint_missing_sweep_geometry")
    return {
        "mechanism_terrain_clearance_m": mechanism_clearance,
        "static_truck_aabb_separation_m": truck_clearance,
    }


def oriented_terrain_separation(corners, rotation, terrain):
    """Conservative SAT distance lower bound: oriented box to height columns."""
    corners = np.asarray(corners, dtype=float)
    height = np.asarray(terrain['height_m'], dtype=float)
    origin = np.asarray(terrain['origin_xy_m'], dtype=float)
    cell = float(terrain['resolution_m'])
    lower = np.floor((corners.min(0)[:2] - .02 - origin) / cell).astype(int)
    upper = np.floor((corners.max(0)[:2] + .02 - origin) / cell).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.array(height.shape[::-1]) - 1)
    ground = float(corners[:, 2].min())
    if np.any(lower > upper):
        return ground
    iy, ix = np.mgrid[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
    heights = height[iy, ix].ravel()
    known = np.isfinite(heights) & (heights > 0)
    if not known.any():
        return ground
    xy = origin + cell * (np.column_stack([ix.ravel(), iy.ravel()])[known] + .5)
    bottom = min(0., ground) - 1.
    centers = np.column_stack([xy, (heights[known] + bottom) / 2])
    half = np.column_stack([np.full((len(xy), 2), cell / 2), (heights[known] - bottom) / 2])
    axes = [*np.eye(3), *rotation.T]
    axes += [np.cross(a, b) for a in np.eye(3) for b in rotation.T]
    gaps = np.zeros(len(xy))
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis / norm
        projected = corners @ axis
        center = centers @ axis
        radius = half @ np.abs(axis)
        gaps = np.maximum(gaps, np.maximum(center - radius - projected.max(),
                                           projected.min() - center - radius))
    return min(ground, float(gaps.min()))


def _joint_bounds_ok(qpos: np.ndarray, chain) -> bool:
    q = np.asarray(qpos, dtype=np.float64)
    return bool(
        q.ndim == 1 and np.isfinite(q).all()
        and abs(q[int(chain.axes[0])]) <= 2.992
        and all(lower <= q[index] <= upper
                for index, (lower, upper) in chain.limits.items()))


def plan_rear_lower_prepoint(
        *,
        terrain: dict,
        contact_selection: dict,
        operator_qpos,
        open_bucket_proxy: dict,
        truck_world_vertices: np.ndarray,
        assembly,
        chain,
        output_dt_s: float = 1.0 / 60.0,
        validation_substeps: int = 2,
        maximum_segment_frames: int = 2400,
        maximum_closure_m: float = 1.0e-4,
        required_bucket_clearance_m: float =
            REQUIRED_BUCKET_CLEARANCE_M,
        transfer_profiles: tuple[tuple[float, float], ...] | None = None,
        prepoint_offsets: tuple[tuple[float, float], ...] | None = None) -> dict:
    """Select and validate a staged non-contact route to a rear prepoint."""
    contact_xy = np.asarray(contact_selection["contact_xy_m"], dtype=np.float64)
    anchor_xy = np.asarray(
        contact_selection["strip_anchor_xy_m"], dtype=np.float64)
    contact_z = float(contact_selection["contact_surface_z_m"])
    crest_z = float(contact_selection["crest_surface_z_m"])
    start = np.asarray(operator_qpos, dtype=np.float64)
    truck = np.asarray(truck_world_vertices, dtype=np.float64)
    if (contact_xy.shape != (2,) or anchor_xy.shape != (2,)
            or not np.isfinite(contact_xy).all()
            or not np.isfinite(anchor_xy).all()
            or not math.isfinite(contact_z) or not math.isfinite(crest_z)
            or start.ndim != 1 or not np.isfinite(start).all()
            or truck.ndim != 2 or truck.shape[1] != 3 or not len(truck)
            or not np.isfinite(truck).all()
            or not math.isfinite(maximum_closure_m)
            or maximum_closure_m <= 0.0
            or not math.isfinite(required_bucket_clearance_m)
            or required_bucket_clearance_m <= 0.0):
        raise ValueError("rear_prepoint_plan_contract")
    radial = contact_xy / np.linalg.norm(contact_xy)
    collision_terrain = ground_completed_terrain(terrain)
    local_samples = proxy_surface_samples(open_bucket_proxy)
    geometries, _ = assembly.assemble(start)
    required_bucket_clearance = float(required_bucket_clearance_m)
    rejections = []

    # The contact-to-prepoint offset and final pitch are user-confirmed.  Only
    # the non-contact high transfer pose varies.  Cheap IK screening therefore
    # rejects unreachable high poses before any dense full-path sweep.
    candidates = [
        # outward, upward, final pitch, high offset above crest, transfer pitch
        (0.30, 0.30, -0.32, 0.55, -0.35),
        (0.30, 0.30, -0.32, 0.525, -0.35),
        (0.30, 0.30, -0.32, 0.50, -0.32),
    ]
    if transfer_profiles is not None:
        profiles = np.asarray(transfer_profiles, dtype=np.float64)
        if (profiles.ndim != 2 or profiles.shape[1] != 2 or not 1 <= len(profiles) <= 8
                or not np.isfinite(profiles).all()
                or np.any((profiles[:, 0] < .50) | (profiles[:, 0] > 2.0))
                or np.any((profiles[:, 1] < -.8) | (profiles[:, 1] > -.20))):
            raise ValueError('rear_prepoint_transfer_profiles')
        candidates = [(0.30, 0.30, -0.32, float(h), float(p)) for h,p in profiles]
    if prepoint_offsets is not None:
        offsets = np.asarray(prepoint_offsets, dtype=np.float64)
        if (offsets.ndim != 2 or offsets.shape[1] != 2 or not 1 <= len(offsets) <= 3
                or not np.isfinite(offsets).all() or np.any((offsets < .30) | (offsets > .60))):
            raise ValueError('rear_prepoint_offsets')
        candidates = [(float(out), float(up), pitch, h, p)
                      for out, up in offsets for _, _, pitch, h, p in candidates]
    for (outward_m, upward_m, pitch_rad, high_offset_m,
         transfer_pitch_rad) in candidates:
        prepoint = np.r_[contact_xy + outward_m * radial,
                         contact_z + upward_m]
        stages = [
            ("HIGH_CLEAR", np.r_[anchor_xy, crest_z + high_offset_m],
             -0.32),
            # Keep the outward translation at the same high level.  A prior
            # diagonal outward/downward segment reached the sparse MPM
            # contact support shell despite large mesh-to-heightfield gap.
            ("OUTWARD_HIGH", np.r_[prepoint[:2],
                                    crest_z + high_offset_m],
             transfer_pitch_rad),
            ("REAR_LOWER_PREPOINT", prepoint, pitch_rad),
        ]
        try:
            goals = []
            seed = start.copy()
            for name, target, stage_pitch in stages:
                seed = solve_pose(
                    assembly, chain, seed, target, stage_pitch, TIP)
                if not _joint_bounds_ok(seed, chain):
                    raise ValueError("rear_prepoint_goal_joint_bounds")
                goals.append((name, target, stage_pitch, seed.copy()))

            minima = {
                "bucket_terrain_clearance_m": math.inf,
                "mechanism_terrain_clearance_m": math.inf,
                "static_truck_aabb_separation_m": math.inf,
            }
            maximum_closure = 0.0

            def advance(qpos, velocity, dt):
                chain.q = qpos.copy()
                return chain.advance(velocity, dt)[0]

            def check(qpos, _previous):
                nonlocal maximum_closure
                if not _joint_bounds_ok(qpos, chain):
                    raise ValueError("rear_prepoint_path_joint_bounds")
                _, bodies = assembly.assemble(qpos, include_meshes=False)
                bucket_pose = bodies["bucket"]
                bucket_world = (
                    local_samples @ bucket_pose[:3, :3].T
                    + bucket_pose[:3, 3])
                bucket = sampled_surface_clearance(
                    bucket_world, collision_terrain)
                if bucket["clearance_m"] < required_bucket_clearance:
                    raise ValueError(
                        "rear_prepoint_bucket_clearance:"
                        f"{bucket['clearance_m']:.9f}; stage={name}; "
                        f"point={bucket['proxy_point_world_m']}; surface={bucket['cell_surface_z_m']}")
                sweep = mechanism_and_truck_clearance(
                    geometries, bodies, terrain, truck)
                mechanism_clearance = sweep[
                    "mechanism_terrain_clearance_m"]
                truck_clearance = sweep[
                    "static_truck_aabb_separation_m"]
                if mechanism_clearance < MECHANISM_CLEARANCE_M:
                    raise ValueError(
                        "rear_prepoint_mechanism_clearance:"
                        f"{mechanism_clearance:.9f}")
                if truck_clearance < TRUCK_CLEARANCE_M:
                    raise ValueError(
                        "rear_prepoint_truck_clearance:"
                        f"{truck_clearance:.9f}")
                closure = float(np.max(np.abs(chain.residual(qpos))))
                if closure > maximum_closure_m:
                    raise ValueError(
                        f"rear_prepoint_closure:{closure:.9f}")
                minima["bucket_terrain_clearance_m"] = min(
                    minima["bucket_terrain_clearance_m"],
                    bucket["clearance_m"])
                minima["mechanism_terrain_clearance_m"] = min(
                    minima["mechanism_terrain_clearance_m"],
                    mechanism_clearance)
                minima["static_truck_aabb_separation_m"] = min(
                    minima["static_truck_aabb_separation_m"],
                    truck_clearance)
                maximum_closure = max(maximum_closure, closure)
                return {
                    **bucket,
                    "mechanism_terrain_clearance_m": mechanism_clearance,
                    "static_truck_aabb_separation_m": truck_clearance,
                    "closure_m": closure,
                }

            # Reject an obstructed endpoint before an expensive dense sweep.
            # This does not replace the subsequent whole-path validation.
            for name, _, _, goal in goals:
                check(goal, None)
            qpos = start.copy()
            segment_results = []
            all_qpos = []
            all_velocity = []
            for name, target, stage_pitch, goal in goals:
                path = sampled_approach(
                    qpos, goal, axes=chain.axes,
                    speed_limits=KINEMATIC_COMMAND_LIMITS,
                    advance=advance, check_pose=check,
                    dt=output_dt_s, max_steps=maximum_segment_frames,
                    validation_substeps=validation_substeps)
                diagnostics = path.pop("diagnostics")
                segment_results.append({
                    "name": name,
                    "target_tip_world_m": target.tolist(),
                    "target_pitch_rad": float(stage_pitch),
                    "frames": int(path["steps"]),
                    "duration_sim_s": float(path["duration_sim_s"]),
                    "minimum_bucket_terrain_clearance_m": min(
                        row["clearance_m"] for row in diagnostics),
                    "endpoint_bucket_terrain_clearance_m": diagnostics[-1][
                        "clearance_m"],
                    "goal_qpos": goal.tolist(),
                })
                all_qpos.append(path["qpos"] if not all_qpos
                                else path["qpos"][1:])
                all_velocity.append(path["velocity"])
                qpos = goal.copy()
            return {
                "status": "read_only_path_validated_not_execution_approved",
                "contact_target_world_m": np.r_[contact_xy, contact_z],
                "prepoint_tip_world_m": prepoint,
                "prepoint_pitch_rad": float(pitch_rad),
                "outward_high_pitch_rad": float(transfer_pitch_rad),
                "high_offset_above_crest_m": float(high_offset_m),
                "manual_contact_delta_m": np.r_[contact_xy, contact_z]
                - prepoint,
                "outward_offset_m": float(outward_m),
                "upward_offset_m": float(upward_m),
                "segments": segment_results,
                "qpos": np.concatenate(all_qpos, axis=0),
                "velocity": np.concatenate(all_velocity, axis=0),
                "total_frames": sum(row["frames"]
                                    for row in segment_results),
                "total_duration_sim_s": sum(row["duration_sim_s"]
                                             for row in segment_results),
                "minimum_bucket_terrain_clearance_m": minima[
                    "bucket_terrain_clearance_m"],
                "minimum_mechanism_terrain_clearance_m": minima[
                    "mechanism_terrain_clearance_m"],
                "minimum_static_truck_aabb_separation_m": minima[
                    "static_truck_aabb_separation_m"],
                "maximum_closure_m": maximum_closure,
                "bucket_clearance_limit_m": BUCKET_CLEARANCE_M,
                "bucket_sampling_buffer_m": BUCKET_SAMPLING_BUFFER_M,
                "sparse_mpm_contact_support_clearance_m":
                    SPARSE_MPM_CONTACT_SUPPORT_CLEARANCE_M,
                "required_bucket_clearance_m":
                    required_bucket_clearance,
                "proxy_sample_spacing_m": PROXY_SAMPLE_SPACING_M,
                "proxy_sample_count": int(len(local_samples)),
                "mechanism_clearance_limit_m": MECHANISM_CLEARANCE_M,
                "static_truck_clearance_limit_m": TRUCK_CLEARANCE_M,
                "maximum_closure_limit_m": float(maximum_closure_m),
                "command_speed_limits": KINEMATIC_COMMAND_LIMITS.copy(),
                "validation_substeps": int(validation_substeps),
                "validation_dt_s": float(output_dt_s / validation_substeps),
                "ground_completion": collision_terrain[
                    "ground_completion"],
                "rejected_candidate_count": len(rejections),
                "rejected_candidates": rejections,
                "continuous_collision_proven": False,
                "physics_stepped": False,
                "digging_executed": False,
                "training_eligible": False,
                "execution_approved": False,
            }
        except ValueError as error:
            rejections.append({
                "outward_offset_m": float(outward_m),
                "upward_offset_m": float(upward_m),
                "pitch_rad": float(pitch_rad),
                "outward_high_pitch_rad": float(transfer_pitch_rad),
                "high_offset_above_crest_m": float(high_offset_m),
                "reason": str(error),
            })
    raise ValueError(
        "rear_prepoint_no_valid_candidate:" + str(rejections))


def _material_clearance(vertices: np.ndarray, terrain: dict) -> float | None:
    """Local import boundary for the established conservative link check."""
    # This avoids making the generic observation module depend on v4 paths at
    # import time while retaining its already tested clearance semantics.
    from layered_excavation_approach_planner import _terrain_clearance
    return _terrain_clearance(
        vertices, terrain, padding_m=0.0, require_known=False)
