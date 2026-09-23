#!/usr/bin/env python3
"""Deterministic observation, target selection, and non-contact approach planning."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import numpy as np

from mixed_material_r0 import quaternion_matrix_wxyz
from planning.approach import sampled_approach
from planning.cycle import select_cut
from planning.cycle_geometry import TIP, solve_pose
from run_mixed_material_r3_a_probe import _read_obj


GRID_BOUNDS_XY_M = (3.5, 7.3, -1.5, 1.5)
GRID_RESOLUTION_M = 0.1
BUCKET_MOUTH_WIDTH_M = 0.6
BUCKET_PATH_CLEARANCE_M = 0.15
MECHANISM_PATH_CLEARANCE_M = 0.02
TARGET_TIP_HEIGHT_ABOVE_SURFACE_M = 0.40
TARGET_BUCKET_PITCH_RAD = -0.32
COMMAND_LIMITS_M_S = np.asarray([0.12, 0.0225, 0.0225, 0.0675])


def _grid_shape(
        bounds_xy_m: tuple[float, float, float, float],
        resolution_m: float) -> tuple[int, int]:
    xmin, xmax, ymin, ymax = bounds_xy_m
    if (not all(math.isfinite(value) for value in bounds_xy_m)
            or xmin >= xmax or ymin >= ymax
            or not math.isfinite(resolution_m) or resolution_m <= 0.0):
        raise ValueError("approach_grid_contract")
    return (
        math.ceil((ymax - ymin) / resolution_m - 1e-10),
        math.ceil((xmax - xmin) / resolution_m - 1e-10),
    )


def terrain_from_points(
        points: np.ndarray,
        *,
        bounds_xy_m: tuple[float, float, float, float] = GRID_BOUNDS_XY_M,
        resolution_m: float = GRID_RESOLUTION_M) -> dict:
    """Build the exact cell-max terrain consumed by the frozen selector."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("approach_point_shape")
    rows, columns = _grid_shape(bounds_xy_m, resolution_m)
    xmin, xmax, ymin, ymax = bounds_xy_m
    valid = np.isfinite(values).all(axis=1)
    valid &= ((values[:, 0] >= xmin) & (values[:, 0] < xmax)
              & (values[:, 1] >= ymin) & (values[:, 1] < ymax))
    selected = values[valid]
    column = np.floor((selected[:, 0] - xmin) / resolution_m).astype(int)
    row = np.floor((selected[:, 1] - ymin) / resolution_m).astype(int)
    cell_id = row * columns + column
    size = rows * columns
    height = np.full(size, -np.inf, dtype=np.float64)
    np.maximum.at(height, cell_id, selected[:, 2])
    count = np.bincount(cell_id, minlength=size)
    height[count == 0] = np.nan
    return {
        "valid": bool(np.count_nonzero(count)),
        "origin_xy_m": [xmin, ymin],
        "resolution_m": float(resolution_m),
        "height_m": height.reshape((rows, columns)),
        "point_count": count.reshape((rows, columns)),
    }


def rock_surface_points(
        repository_root: str | Path,
        records: list[dict],
        rigid_qpos: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    root = Path(repository_root).resolve()
    poses = np.asarray(rigid_qpos, dtype=np.float64)
    if poses.shape != (len(records), 7):
        raise ValueError("approach_rock_pose_shape")
    by_id = {}
    for record, pose in zip(records, poses, strict=True):
        vertices, _ = _read_obj(
            root / "assets/v4/mixed-material-r0" /
            f"{record['prototype_id']}.obj")
        local = (np.asarray(vertices, dtype=np.float64)
                 * float(record["equivalent_diameter_m"]))
        world = local @ quaternion_matrix_wxyz(pose[3:7]).T + pose[:3]
        by_id[record["rock_id"]] = world
    combined = np.concatenate(list(by_id.values()), axis=0)
    return combined, by_id


def build_fused_terrain(
        repository_root: str | Path,
        records: list[dict],
        particle_q: np.ndarray,
        rigid_qpos: np.ndarray) -> tuple[dict, dict]:
    particles = np.asarray(particle_q, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 3:
        raise ValueError("approach_particle_shape")
    rock_points, rock_points_by_id = rock_surface_points(
        repository_root, records, rigid_qpos)
    terrain = terrain_from_points(np.concatenate((particles, rock_points), axis=0))
    selection = select_cut(terrain, bucket_width=BUCKET_MOUTH_WIDTH_M)
    terrain_summary = {
        "bounds_xy_m": list(GRID_BOUNDS_XY_M),
        "resolution_m": GRID_RESOLUTION_M,
        "shape_yx": list(terrain["height_m"].shape),
        "known_cell_count": int(np.count_nonzero(terrain["point_count"])),
        "particle_point_count": int(len(particles)),
        "rock_surface_vertex_count": int(len(rock_points)),
        "visible_rock_ids": sorted(rock_points_by_id),
        "maximum_fused_height_m": float(np.nanmax(terrain["height_m"])),
    }
    return terrain, {"selection": selection, "terrain": terrain_summary}


def _proxy_local_vertices(proxy: dict) -> np.ndarray:
    values = []
    for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
        half = np.asarray(box["half_size_m"], dtype=np.float64)
        corners = np.asarray([
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)], dtype=np.float64)
        corners = (corners @ quaternion_matrix_wxyz(
            box["quaternion_wxyz"]).T
            + np.asarray(box["position_bucket_m"], dtype=np.float64))
        values.append(corners)
    values.extend(np.asarray(side["vertices_bucket_m"], dtype=np.float64)
                  for side in proxy["side_plate_meshes"])
    return np.concatenate(values, axis=0)


def _terrain_clearance(
        vertices: np.ndarray,
        terrain: dict,
        *,
        padding_m: float = 0.01,
        require_known: bool = True) -> float | None:
    points = np.asarray(vertices, dtype=np.float64)
    height = np.asarray(terrain["height_m"], dtype=np.float64)
    count = np.asarray(terrain["point_count"], dtype=np.int64)
    origin = np.asarray(terrain["origin_xy_m"], dtype=np.float64)
    resolution = float(terrain["resolution_m"])
    lower = points.min(axis=0) - padding_m
    upper = points.max(axis=0) + padding_m
    roi_upper = origin + resolution * np.asarray(
        [height.shape[1], height.shape[0]], dtype=np.float64)
    if (upper[0] < origin[0] or upper[1] < origin[1]
            or lower[0] >= roi_upper[0] or lower[1] >= roi_upper[1]):
        return None
    lower_xy = np.maximum(lower[:2], origin)
    upper_xy = np.minimum(upper[:2], np.nextafter(roi_upper, origin))
    start = np.floor((lower_xy - origin) / resolution).astype(int)
    end = np.floor((upper_xy - origin) / resolution).astype(int)
    patch = height[start[1]:end[1] + 1, start[0]:end[0] + 1]
    patch_count = count[start[1]:end[1] + 1, start[0]:end[0] + 1]
    if require_known and (not np.isfinite(patch).all()
                          or np.any(patch_count < 1)):
        raise ValueError("approach_unknown_bucket_corridor")
    finite = patch[np.isfinite(patch) & (patch_count > 0)]
    if not finite.size:
        return None
    return float(lower[2] - finite.max())


def _load_kinematics(
        python_directory: str | Path,
        model_path: str | Path,
        mesh_directory: str | Path):
    directory = str(Path(python_directory).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    Assembler = importlib.import_module("scene_assembly").Assembler
    install = importlib.import_module("fast_kinematics").install
    closed_chain = importlib.import_module("closed_chain")
    assembly = Assembler(Path(model_path).resolve(), Path(mesh_directory).resolve())
    install(assembly)
    return assembly, closed_chain.ClosedChain(assembly)


def plan_noncontact_approach(
        *,
        terrain: dict,
        selection: dict,
        operator_qpos: list[float],
        open_bucket_proxy: dict,
        kinematics_python_directory: str | Path,
        kinematics_model_path: str | Path,
        kinematics_mesh_directory: str | Path,
        output_dt_s: float = 1.0 / 120.0,
        maximum_path_frames: int = 2400,
        maximum_planned_closure_m: float = 1.0e-6,
        validation_substeps: int = 1,
        command_speed_limits: np.ndarray = COMMAND_LIMITS_M_S,
        target_tip_height_above_surface_m: float =
            TARGET_TIP_HEIGHT_ABOVE_SURFACE_M,
        target_bucket_pitch_rad: float = TARGET_BUCKET_PITCH_RAD) -> dict:
    if (not math.isfinite(maximum_planned_closure_m)
            or maximum_planned_closure_m <= 0.0):
        raise ValueError("approach_planned_closure_contract")
    if (not math.isfinite(target_tip_height_above_surface_m)
            or target_tip_height_above_surface_m <= 0.0
            or not math.isfinite(target_bucket_pitch_rad)):
        raise ValueError("approach_target_contract")
    speed_limits = np.asarray(command_speed_limits, dtype=np.float64)
    if (speed_limits.shape != COMMAND_LIMITS_M_S.shape
            or not np.isfinite(speed_limits).all()
            or np.any(speed_limits <= 0.0)
            or np.any(speed_limits > COMMAND_LIMITS_M_S)):
        raise ValueError("approach_command_speed_limits")
    assembly, chain = _load_kinematics(
        kinematics_python_directory,
        kinematics_model_path,
        kinematics_mesh_directory)
    start = np.asarray(operator_qpos, dtype=np.float64)
    xy = np.asarray(selection["entry_xy_m"], dtype=np.float64)
    target = np.asarray([
        xy[0], xy[1],
        float(selection["surface_z_m"])
        + target_tip_height_above_surface_m,
    ])
    goal = solve_pose(
        assembly, chain, start, target, target_bucket_pitch_rad, TIP)
    local_proxy = _proxy_local_vertices(open_bucket_proxy)
    geometries, _ = assembly.assemble(start)

    def advance(qpos, velocity, dt):
        chain.q = qpos.copy()
        return chain.advance(velocity, dt)[0]

    def check(qpos, _previous):
        _, bodies = assembly.assemble(qpos, include_meshes=False)
        bucket_pose = bodies["bucket"]
        bucket_world = (local_proxy @ bucket_pose[:3, :3].T
                        + bucket_pose[:3, 3])
        bucket_clearance = _terrain_clearance(
            bucket_world, terrain, require_known=True)
        if (bucket_clearance is None
                or bucket_clearance < BUCKET_PATH_CLEARANCE_M):
            raise ValueError(
                f"approach_bucket_clearance:{bucket_clearance!r}")
        mechanism_clearance = math.inf
        for geometry in geometries:
            if geometry["mesh"] == "bucket":
                continue
            pose = bodies[geometry["body"]]
            world = (geometry["body_local_vertices"] @ pose[:3, :3].T
                     + pose[:3, 3])
            clearance = _terrain_clearance(
                world, terrain, padding_m=0.0, require_known=False)
            if clearance is not None:
                mechanism_clearance = min(mechanism_clearance, clearance)
        if mechanism_clearance < MECHANISM_PATH_CLEARANCE_M:
            raise ValueError(
                f"approach_mechanism_clearance:{mechanism_clearance}")
        closure = float(np.max(np.abs(chain.residual(qpos))))
        if closure > maximum_planned_closure_m:
            raise ValueError(f"approach_closure:{closure}")
        return {
            "bucket_terrain_clearance_m": bucket_clearance,
            "mechanism_terrain_clearance_m": (
                None if math.isinf(mechanism_clearance)
                else mechanism_clearance),
            "closure_m": closure,
        }

    path = sampled_approach(
        start,
        goal,
        axes=chain.axes,
        speed_limits=speed_limits,
        advance=advance,
        check_pose=check,
        dt=output_dt_s,
        max_steps=maximum_path_frames,
        validation_substeps=validation_substeps,
    )
    diagnostics = path.pop("diagnostics")
    path["minimum_bucket_terrain_clearance_m"] = min(
        item["bucket_terrain_clearance_m"] for item in diagnostics)
    path["endpoint_bucket_terrain_clearance_m"] = diagnostics[-1][
        "bucket_terrain_clearance_m"]
    path["maximum_closure_m"] = max(
        item["closure_m"] for item in diagnostics)
    path["maximum_planned_closure_limit_m"] = float(
        maximum_planned_closure_m)
    path["target_tip_world_m"] = target
    path["target_pitch_rad"] = float(target_bucket_pitch_rad)
    path["command_speed_limits"] = speed_limits
    path["target_tip_height_above_surface_m"] = float(
        target_tip_height_above_surface_m)
    path["goal_qpos"] = goal
    return path
