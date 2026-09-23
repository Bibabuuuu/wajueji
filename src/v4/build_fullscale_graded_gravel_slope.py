"""Compile the frozen 6.1 m^3 bank into a deterministic graded-gravel scene.

This module is deliberately CPU/NumPy geometry only.  It never imports or
advances MuJoCo, Newton, Warp, or a GPU runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

import numpy as np

from mixed_material_r0 import (
    load_obj,
    mass_properties,
    matrix_quaternion_wxyz,
    proper_axis_rotations,
    quaternion_matrix_wxyz,
)


CONFIG_SCHEMA = "excavation_fullscale_graded_gravel_slope_config_v2"
MANIFEST_SCHEMA = "excavation_fullscale_graded_gravel_slope_manifest_v2"
METRICS_SCHEMA = "excavation_fullscale_graded_gravel_slope_metrics_v2"
REPORT_SCHEMA = "excavation_fullscale_graded_gravel_slope_report_v2"

FROZEN_SCENARIO = "mixed-autonomous-fullscale-graded-gravel-slope-truck-v1"
FROZEN_SEED = 260920240
FROZEN_NOMINAL_MIN = [3.8, -1.1, 0.0]
FROZEN_NOMINAL_MAX = [6.0, 1.1, 1.6]
FROZEN_VOLUME_M3 = 6.1
FROZEN_MPM_VOLUME_M3 = 1.83
FROZEN_CLASSES = [
    ("small_0p16_0p24", 0.16, 0.24, 3.66, 0.60),
    ("medium_0p24_0p32", 0.24, 0.32, 0.488, 0.08),
    ("large_0p32_0p40", 0.32, 0.40, 0.122, 0.02),
]
FROZEN_ASSETS = {
    "angular_tetra": "6934783ff5fc95e68cfddfb30f5630ff19be7633c6679d5f1d8f2a3ab0d2afb7",
    "angular_bipyramid": "e9e432a263512a21232a6b590a51ca923a7b2432c8458312b5fbd76a3d2d55cd",
    "angular_octa": "6279098235a6dd22ef97ffd2c2d2d2ff028580e26c3110040cb08264faea1fd8",
    "angular_block": "963182f35d733b763f9d829f2a5cae71ba03c8123a889a75bc19997cf6142b43",
}
FROZEN_BANK_SHA256 = "bc3fb89dc46f2787f1c65b8699fd5eb79d497a7bf02998c4da5b9f744622e89b"
FROZEN_Q_SHA256 = "5c4cd8f0a12885f0d78bb4157dc1706d7e363c2c7564c1415093b89cfba6dba8"
FROZEN_SOLID_BANK_SHA256 = "d424ae75ecc5a25277e7362d816b900430bec499d0ffab88c32e73708399c468"
FROZEN_R0_SOURCE_SHA256 = "605005d3815032c54cf4f72d42b9df2d108aec56621a70ae009b1ba2d07f4001"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non_finite_json_value")
        if numeric == 0.0:
            return 0.0
        return float(f"{numeric:.12g}")
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _stable(value.tolist())
    raise ValueError(f"unsupported_json_type:{type(value).__name__}")


def _json_bytes(value) -> bytes:
    return (json.dumps(_stable(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _write_json(path: Path, value) -> None:
    path.write_bytes(_json_bytes(value))


def _assert_finite(value, path="$" ) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non_finite_parameter:{path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}/{key}")
        return
    raise ValueError(f"unsupported_parameter_type:{path}")


def _same_numbers(actual, expected, tolerance=1e-12) -> bool:
    return np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                       rtol=0.0, atol=tolerance)


def validate_config(config: dict) -> None:
    _assert_finite(config)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_fullscale_config_schema")
    if config.get("scenario_name") != FROZEN_SCENARIO:
        raise ValueError("scenario_name_drift")
    if config.get("seed") != FROZEN_SEED:
        raise ValueError("seed_drift")
    if config.get("coordinate_frame") != "world":
        raise ValueError("coordinate_frame_drift")
    bank = config.get("bank", {})
    if (not _same_numbers(bank.get("nominal_bounds_min_m"), FROZEN_NOMINAL_MIN)
            or not _same_numbers(bank.get("nominal_bounds_max_m"), FROZEN_NOMINAL_MAX)
            or not math.isclose(float(bank.get("nominal_volume_m3", -1)),
                                FROZEN_VOLUME_M3, abs_tol=1e-12)):
        raise ValueError("fullscale_bank_contract_drift")
    if (bank.get("occupied_cell_count") != 6100
            or not math.isclose(float(bank.get("occupied_cell_size_m", -1)), 0.1,
                                abs_tol=1e-12)
            or not math.isclose(float(bank.get("particle_spacing_m", -1)), 0.025,
                                abs_tol=1e-12)
            or not math.isclose(float(bank.get("particle_density_kg_m3", -1)), 1670.0,
                                abs_tol=1e-12)):
        raise ValueError("bank_discretization_contract_drift")
    if bank.get("material_ids") != {"mpm_fines": 0}:
        raise ValueError("material_id_contract_drift")

    recipe = config.get("material_recipe", {})
    expected_recipe = {
        "small_rigid_fraction": 0.60,
        "mpm_fines_fraction": 0.30,
        "medium_rigid_fraction": 0.08,
        "large_rigid_fraction": 0.02,
    }
    for key, expected in expected_recipe.items():
        if not math.isclose(float(recipe.get(key, -1)), expected, abs_tol=1e-12):
            raise ValueError("material_recipe_fraction_drift")
    if not math.isclose(sum(float(recipe[key]) for key in expected_recipe), 1.0,
                            abs_tol=1e-12):
        raise ValueError("material_recipe_not_closed")
    if not 0.0 < float(recipe.get("relative_total_volume_tolerance", -1)) <= 0.02:
        raise ValueError("invalid_total_volume_tolerance")
    if not 0.0 < float(recipe.get("relative_rigid_class_volume_tolerance", -1)) <= 0.005:
        raise ValueError("invalid_rigid_class_volume_tolerance")

    rocks = config.get("rocks", {})
    if "total_count" in rocks:
        raise ValueError("fixed_rock_count_forbidden")
    if rocks.get("assets") != FROZEN_ASSETS:
        raise ValueError("rock_asset_contract_drift")
    if not math.isclose(float(rocks.get("density_kg_m3", -1)), 2600.0,
                        abs_tol=1e-12):
        raise ValueError("rock_density_drift")
    classes = rocks.get("size_classes", [])
    if len(classes) != len(FROZEN_CLASSES):
        raise ValueError("size_class_contract_drift")
    for actual, expected in zip(classes, FROZEN_CLASSES):
        name, low, high, target, fraction = expected
        if (actual.get("class_id") != name
                or "count" in actual
                or not _same_numbers(actual.get("diameter_range_m"), [low, high])
                or not math.isclose(float(actual.get("target_volume_m3", -1)),
                                    target, abs_tol=1e-12)
                or not math.isclose(float(actual.get("target_fraction", -1)),
                                    fraction, abs_tol=1e-12)):
            raise ValueError("size_class_contract_drift")
    if not math.isclose(sum(float(item["target_volume_m3"]) for item in classes)
                        + FROZEN_MPM_VOLUME_M3, FROZEN_VOLUME_M3, abs_tol=1e-12):
        raise ValueError("material_volume_recipe_not_closed")

    placement = config.get("placement", {})
    gap = float(placement.get("minimum_rock_sat_gap_m", -1))
    if gap < 0.0:
        raise ValueError("negative_rock_gap")
    if gap >= 2.2:
        raise ValueError("impossible_placement_constraint")
    if int(placement.get("relaxation_iterations", 0)) <= 0:
        raise ValueError("invalid_relaxation_iteration_limit")
    if not 0.0 < float(placement.get("deep_depth_threshold_m", -1)):
        raise ValueError("invalid_deep_depth_threshold")
    for key in ("minimum_medium_deep_fraction", "minimum_large_deep_fraction",
                "small_visible_share_minimum"):
        if not 0.0 < float(placement.get(key, -1)) <= 1.0:
            raise ValueError("invalid_placement_fraction")
    if placement.get("carve_policy") != "remove_particle_center_inside_closed_convex_rock":
        raise ValueError("carve_policy_drift")
    if placement.get("mechanism_collision_check") != "pending_mainline_geometry_binding":
        raise ValueError("mechanism_collision_overclaim")
    if placement.get("truck_collision_check") != "pending_mainline_geometry_binding":
        raise ValueError("truck_collision_overclaim")

    inputs = config.get("inputs", {})
    if inputs.get("initial_bank_npz", {}).get("sha256") != FROZEN_BANK_SHA256:
        raise ValueError("bank_input_hash_contract_drift")
    if inputs.get("initial_bank_npz", {}).get("q_array_sha256") != FROZEN_Q_SHA256:
        raise ValueError("bank_q_hash_contract_drift")
    if inputs.get("solid_bank_source", {}).get("sha256") != FROZEN_SOLID_BANK_SHA256:
        raise ValueError("solid_bank_hash_contract_drift")
    if inputs.get("r0_geometry_source", {}).get("sha256") != FROZEN_R0_SOURCE_SHA256:
        raise ValueError("r0_source_hash_contract_drift")

    claims = config.get("claims")
    expected_claims = {
        "geometry_only": True,
        "physics_executed": False,
        "physics_qualified": False,
        "training_executed": False,
        "three_scoop_qualified": False,
    }
    if claims != expected_claims:
        raise ValueError("forbidden_capability_claim")


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _resolve(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _load_frozen_bank(config: dict, repository_root: Path) -> np.ndarray:
    bank_record = config["inputs"]["initial_bank_npz"]
    bank_path = _resolve(repository_root, bank_record["path"])
    solid_path = _resolve(repository_root, config["inputs"]["solid_bank_source"]["path"])
    r0_path = _resolve(repository_root, config["inputs"]["r0_geometry_source"]["path"])
    for path, expected, label in (
            (bank_path, FROZEN_BANK_SHA256, "bank_npz"),
            (solid_path, FROZEN_SOLID_BANK_SHA256, "solid_bank_source"),
            (r0_path, FROZEN_R0_SOURCE_SHA256, "r0_geometry_source")):
        if not path.is_file():
            raise ValueError(f"missing_frozen_input:{label}")
        if sha256(path) != expected:
            raise ValueError(f"frozen_input_hash_mismatch:{label}")
    with np.load(bank_path, allow_pickle=False) as archive:
        if "q" not in archive.files:
            raise ValueError("missing_q_array")
        q = archive["q"]
    if q.shape != (390400, 3) or q.dtype != np.float32:
        raise ValueError("frozen_q_shape_or_dtype_mismatch")
    if hashlib.sha256(q.tobytes(order="C")).hexdigest() != FROZEN_Q_SHA256:
        raise ValueError("frozen_q_array_hash_mismatch")
    if (not _same_numbers(q.min(axis=0), bank_record["q_bounds_min_m"], 1e-8)
            or not _same_numbers(q.max(axis=0), bank_record["q_bounds_max_m"], 1e-8)):
        raise ValueError("frozen_q_bounds_mismatch")
    return q


def _load_prototypes(config: dict, repository_root: Path) -> dict[str, dict]:
    result = {}
    root = repository_root / config["rocks"]["asset_root"]
    for prototype_id, expected_hash in sorted(FROZEN_ASSETS.items()):
        path = root / f"{prototype_id}.obj"
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"rock_asset_hash_mismatch:{prototype_id}")
        vertices, faces = load_obj(path)
        properties = mass_properties(vertices, faces)
        if (not math.isclose(float(properties["volume"]), math.pi / 6.0,
                             rel_tol=0.0, abs_tol=1e-12)
                or np.linalg.norm(properties["center"]) > 1e-12):
            raise ValueError(f"rock_asset_normalization_mismatch:{prototype_id}")
        result[prototype_id] = {
            "vertices": vertices,
            "faces": faces,
            "unit_volume_m3": float(properties["volume"]),
            "unit_inertia_m5": properties["inertia_unit_density"],
            "unit_radius_m": float(np.linalg.norm(vertices, axis=1).max()),
            "unit_face_normals": np.asarray([
                np.cross(vertices[face][1] - vertices[face][0],
                         vertices[face][2] - vertices[face][0])
                / np.linalg.norm(np.cross(vertices[face][1] - vertices[face][0],
                                          vertices[face][2] - vertices[face][0]))
                for face in faces
            ]),
            "unit_edge_directions": np.asarray(_unique_edges(vertices, faces)),
            "asset_path": str(path.relative_to(repository_root).as_posix()),
            "asset_sha256": expected_hash,
        }
    return result


def _column_surface(q: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = float(config["bank"]["particle_spacing_m"])
    origin = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    high = np.asarray(config["bank"]["nominal_bounds_max_m"], dtype=float)
    nx, ny = np.rint((high[:2] - origin[:2]) / spacing).astype(int)
    ix = np.rint((q[:, 0].astype(float) - origin[0]) / spacing - 0.5).astype(int)
    iy = np.rint((q[:, 1].astype(float) - origin[1]) / spacing - 0.5).astype(int)
    if np.any(ix < 0) or np.any(ix >= nx) or np.any(iy < 0) or np.any(iy >= ny):
        raise ValueError("particle_column_index_out_of_range")
    flat = ix * ny + iy
    top_center = np.full(nx * ny, -np.inf, dtype=np.float64)
    np.maximum.at(top_center, flat, q[:, 2].astype(np.float64))
    particle_surface = top_center[flat] + spacing / 2.0
    depth = particle_surface - q[:, 2].astype(np.float64)
    surface_grid = top_center.reshape(nx, ny) + spacing / 2.0
    return surface_grid, flat, depth


def _surface_at_xy(xy: np.ndarray, surface_grid: np.ndarray,
                   config: dict) -> np.ndarray:
    spacing = float(config["bank"]["particle_spacing_m"])
    origin = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    values = np.asarray(xy, dtype=float)
    ix = np.floor((values[:, 0] - origin[0]) / spacing).astype(int)
    iy = np.floor((values[:, 1] - origin[1]) / spacing).astype(int)
    result = np.full(len(values), -np.inf, dtype=float)
    valid = ((ix >= 0) & (ix < surface_grid.shape[0])
             & (iy >= 0) & (iy < surface_grid.shape[1]))
    result[valid] = surface_grid[ix[valid], iy[valid]]
    return result


def _surface_patch(vertex_xy: np.ndarray, surface_grid: np.ndarray,
                   config: dict) -> np.ndarray:
    spacing = float(config["bank"]["particle_spacing_m"])
    origin = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    low = np.floor((vertex_xy.min(axis=0) - origin[:2]) / spacing).astype(int)
    high = np.floor((vertex_xy.max(axis=0) - origin[:2]) / spacing).astype(int)
    low = np.maximum(low, 0)
    high = np.minimum(high, np.asarray(surface_grid.shape) - 1)
    if np.any(low > high):
        return np.asarray([], dtype=float)
    return surface_grid[low[0]:high[0] + 1, low[1]:high[1] + 1].ravel()


def _plateau_bounds(surface_grid: np.ndarray, config: dict) -> np.ndarray:
    spacing = float(config["bank"]["particle_spacing_m"])
    origin = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    top = float(config["bank"]["nominal_bounds_max_m"][2])
    cells = np.argwhere(np.isclose(surface_grid, top, rtol=0.0, atol=1e-6))
    if not len(cells):
        raise ValueError("full_height_plateau_missing")
    low = origin[:2] + cells.min(axis=0) * spacing
    high = origin[:2] + (cells.max(axis=0) + 1) * spacing
    expected = np.asarray([[4.0, -0.9], [5.8, 0.9]])
    bounds = np.asarray([low, high])
    if not np.allclose(bounds, expected, rtol=0.0, atol=1e-6):
        raise ValueError("unexpected_full_height_plateau")
    return bounds


def _unique_edges(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    keys = set()
    result = []
    for face in faces:
        for first, second in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(first), int(second))))
            if key in keys:
                continue
            keys.add(key)
            direction = vertices[key[1]] - vertices[key[0]]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-14:
                result.append(direction / norm)
    return result


def sat_axis_gap(first: np.ndarray, first_faces: np.ndarray,
                 second: np.ndarray, second_faces: np.ndarray) -> tuple[bool, float]:
    normals = []
    for vertices, faces in ((first, first_faces), (second, second_faces)):
        item = []
        for face in faces:
            a, b, c = vertices[face]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-14:
                item.append(normal / norm)
        normals.append(np.asarray(item))
    return _sat_precomputed(
        first, normals[0], np.asarray(_unique_edges(first, first_faces)),
        second, normals[1], np.asarray(_unique_edges(second, second_faces)))


def _sat_precomputed(first_vertices: np.ndarray, first_normals: np.ndarray,
                     first_edges: np.ndarray, second_vertices: np.ndarray,
                     second_normals: np.ndarray,
                     second_edges: np.ndarray) -> tuple[bool, float]:
    cross_axes = np.cross(first_edges[:, None, :], second_edges[None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(cross_axes, axis=1)
    cross_axes = cross_axes[norms > 1e-10] / norms[norms > 1e-10, None]
    axes = np.concatenate((first_normals, second_normals, cross_axes), axis=0)
    pivots = np.argmax(np.abs(axes), axis=1)
    signs = np.where(axes[np.arange(len(axes)), pivots] < 0.0, -1.0, 1.0)
    axes = axes * signs[:, None]
    _, unique_indices = np.unique(np.round(axes, 12), axis=0, return_index=True)
    axes = axes[np.sort(unique_indices)]
    first_projection = first_vertices @ axes.T
    second_projection = second_vertices @ axes.T
    gaps = np.maximum(
        second_projection.min(axis=0) - first_projection.max(axis=0),
        first_projection.min(axis=0) - second_projection.max(axis=0),
    )
    maximum_gap = float(gaps.max())
    return maximum_gap <= 1e-12, maximum_gap


def _diameters_for_target(low: float, high: float, target_volume: float) -> np.ndarray:
    """Return a deterministic near-uniform sequence whose analytic volume closes.

    Count is derived from the continuous uniform-diameter expectation.  A single
    common scale adjustment (normally below 0.4%) closes the target exactly while
    keeping every value strictly inside its configured interval.  No class count
    is stored in configuration or used as an input.
    """
    expected_d3 = (high ** 4 - low ** 4) / (4.0 * (high - low))
    expected_volume = math.pi / 6.0 * expected_d3
    count = max(1, int(round(target_volume / expected_volume)))
    diameters = low + (np.arange(count, dtype=float) + 0.5) / count * (high - low)
    scale = (target_volume / float(np.sum(math.pi / 6.0 * diameters ** 3))) ** (1.0 / 3.0)
    diameters *= scale
    if np.any(diameters <= low) or np.any(diameters >= high):
        raise ValueError("target_volume_cannot_close_inside_diameter_range")
    return diameters


def _descriptors(config: dict) -> list[dict]:
    prototype_ids = sorted(config["rocks"]["assets"])
    descriptors = []
    source_index = 0
    for class_config in config["rocks"]["size_classes"]:
        low, high = map(float, class_config["diameter_range_m"])
        diameters = _diameters_for_target(
            low, high, float(class_config["target_volume_m3"]))
        for index, diameter in enumerate(diameters):
            descriptors.append({
                "source_index": source_index,
                "class_index": index,
                "size_class": class_config["class_id"],
                "diameter_m": float(diameter),
                "prototype_id": prototype_ids[index % len(prototype_ids)],
                "target_volume_m3": float(class_config["target_volume_m3"]),
            })
            source_index += 1
    return descriptors


def _yaw(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _base_rotations(vertices: np.ndarray) -> list[np.ndarray]:
    scored = []
    for rotation in proper_axis_rotations():
        oriented = vertices @ rotation.T
        span = np.ptp(oriented, axis=0)
        score = (float(max(span)), float(np.prod(span)), float(span[2]))
        scored.append((score, tuple(rotation.ravel()), rotation))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:8]]


def _sat_gap_axis(first: dict, second: dict) -> tuple[float, np.ndarray]:
    axes_cross = np.cross(first["edge_directions"][:, None, :],
                          second["edge_directions"][None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(axes_cross, axis=1)
    axes_cross = axes_cross[norms > 1e-10] / norms[norms > 1e-10, None]
    axes = np.concatenate((first["face_normals"], second["face_normals"], axes_cross))
    pivots = np.argmax(np.abs(axes), axis=1)
    axes *= np.where(axes[np.arange(len(axes)), pivots] < 0.0, -1.0, 1.0)[:, None]
    first_projection = first["vertices_array"] @ axes.T
    second_projection = second["vertices_array"] @ axes.T
    gaps = np.maximum(second_projection.min(axis=0) - first_projection.max(axis=0),
                      first_projection.min(axis=0) - second_projection.max(axis=0))
    index = int(np.argmax(gaps))
    return float(gaps[index]), axes[index]


def _near_pairs(placed: list[dict], padding: float) -> list[tuple[int, int]]:
    cell_size = 0.45
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, rock in enumerate(placed):
        low = np.floor((rock["vertices_array"].min(axis=0) - padding) / cell_size).astype(int)
        high = np.floor((rock["vertices_array"].max(axis=0) + padding) / cell_size).astype(int)
        for cell in itertools.product(*(range(low[a], high[a] + 1) for a in range(3))):
            cells.setdefault(cell, []).append(index)
    pairs = set()
    for members in cells.values():
        for first, second in itertools.combinations(members, 2):
            pairs.add((first, second) if first < second else (second, first))
    return sorted(pairs)


def _initial_positions(q: np.ndarray, descriptors: list[dict], config: dict,
                       surface_grid: np.ndarray) -> list[np.ndarray]:
    spacing = float(config["bank"]["particle_spacing_m"])
    grid_spacing = float(config["placement"]["candidate_grid_spacing_m"])
    stride = max(1, int(round(grid_spacing / spacing)))
    origin = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    indices = np.rint((q.astype(float) - origin) / spacing - 0.5).astype(int)
    mask = np.all(indices % stride == stride // 2, axis=1)
    candidates = q[mask].astype(float)
    if len(candidates) < len(descriptors):
        raise ValueError(f"insufficient_layered_grid_candidates:{len(candidates)}<{len(descriptors)}")
    surfaces = _surface_at_xy(candidates[:, :2], surface_grid, config)
    depths = surfaces - candidates[:, 2]
    # A deterministic integer hash removes axis-order bands without introducing
    # platform-dependent RNG or floating sorting ties.
    ci = indices[mask].astype(np.int64)
    hashed = ((ci[:, 0] * 73856093) ^ (ci[:, 1] * 19349663)
              ^ (ci[:, 2] * 83492791) ^ int(config["seed"])) & 0x7fffffff
    deep = float(config["placement"]["deep_depth_threshold_m"])
    positions: dict[int, np.ndarray] = {}
    used = set()
    # Place large deepest, then medium; small receives the remaining full-depth
    # lattice.  The later relaxation preserves the class depth clamps.
    for class_id in ("large_0p32_0p40", "medium_0p24_0p32"):
        items = [item for item in descriptors if item["size_class"] == class_id]
        order = np.lexsort((hashed, -depths))
        viable = [int(i) for i in order if depths[i] >= deep and int(i) not in used]
        if len(viable) < len(items):
            raise ValueError(f"insufficient_deep_grid_candidates:{class_id}")
        for item, candidate in zip(sorted(items, key=lambda value: -value["diameter_m"]), viable):
            positions[item["source_index"]] = candidates[candidate].copy()
            used.add(candidate)
    items = [item for item in descriptors if item["size_class"] == "small_0p16_0p24"]
    order = np.lexsort((hashed, depths))
    viable = [int(i) for i in order if int(i) not in used]
    for item, candidate in zip(sorted(items, key=lambda value: -value["diameter_m"]), viable):
        positions[item["source_index"]] = candidates[candidate].copy()
        used.add(candidate)
    return [positions[item["source_index"]] for item in descriptors]


def _place_rocks(config: dict, prototypes: dict[str, dict], surface_grid: np.ndarray,
                 q: np.ndarray) -> list[dict]:
    generator = random.Random(int(config["seed"]))
    descriptors = _descriptors(config)
    positions = _initial_positions(q, descriptors, config, surface_grid)
    density = float(config["rocks"]["density_kg_m3"])
    placed = []
    rotation_cache = {name: _base_rotations(value["vertices"])
                      for name, value in prototypes.items()}
    for descriptor, position in zip(descriptors, positions):
        prototype = prototypes[descriptor["prototype_id"]]
        bases = rotation_cache[descriptor["prototype_id"]]
        base = bases[descriptor["class_index"] % len(bases)]
        rotation = _yaw(generator.random() * 2.0 * math.pi) @ base
        diameter = descriptor["diameter_m"]
        local_vertices = prototype["vertices"] * diameter @ rotation.T
        position = np.asarray(position, dtype=float).copy()
        domain_low = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
        domain_high = np.asarray(config["bank"]["nominal_bounds_max_m"], dtype=float)
        boundary = float(config["placement"]["boundary_clearance_m"])
        local_low, local_high = local_vertices.min(axis=0), local_vertices.max(axis=0)
        position[:2] = np.minimum(
            np.maximum(position[:2], domain_low[:2] + boundary - local_low[:2]),
            domain_high[:2] - boundary - local_high[:2])
        vertex_xy = local_vertices[:, :2] + position[:2]
        surfaces = _surface_at_xy(vertex_xy, surface_grid, config)
        if not np.isfinite(surfaces).all():
            # Edge candidates can be too wide for the outermost particle column.
            # Move only their XY center toward the known full-height plateau.
            position[:2] = np.minimum(np.maximum(position[:2], [4.0, -0.9]), [5.8, 0.9])
            vertex_xy = local_vertices[:, :2] + position[:2]
            surfaces = _surface_at_xy(vertex_xy, surface_grid, config)
        minimum_z = boundary - local_low[2]
        maximum_z = float(np.min(surfaces - boundary - local_vertices[:, 2]))
        if minimum_z > maximum_z:
            plateau_low = np.asarray([4.0, -0.9]) + boundary - local_low[:2]
            plateau_high = np.asarray([5.8, 0.9]) - boundary - local_high[:2]
            if np.any(plateau_low > plateau_high):
                raise ValueError(f"rock_wider_than_full_height_plateau:{descriptor['source_index']}")
            position[:2] = np.minimum(np.maximum(position[:2], plateau_low), plateau_high)
            vertex_xy = local_vertices[:, :2] + position[:2]
            surfaces = _surface_at_xy(vertex_xy, surface_grid, config)
            maximum_z = float(np.min(surfaces - boundary - local_vertices[:, 2]))
            if minimum_z > maximum_z:
                raise ValueError(
                    f"rock_cannot_fit_bank_boundary:{descriptor['source_index']}:"
                    f"minimum_z={minimum_z:.12g}:maximum_z={maximum_z:.12g}")
        position[2] = min(max(position[2], minimum_z), maximum_z)
        volume = prototype["unit_volume_m3"] * diameter ** 3
        placed.append({
            **descriptor,
            "position_array": position,
            "rotation_array": rotation,
            "local_vertices": local_vertices,
            "vertices_array": local_vertices + position,
            "face_normals": prototype["unit_face_normals"] @ rotation.T,
            "edge_directions": prototype["unit_edge_directions"] @ rotation.T,
            "radius": prototype["unit_radius_m"] * diameter,
            "orientation_quaternion_wxyz": matrix_quaternion_wxyz(rotation),
            "volume_m3": volume,
            "mass_kg": density * volume,
            "inertia_about_com_body_kg_m2": (
                density * prototype["unit_inertia_m5"] * diameter ** 5).tolist(),
        })

    placement = config["placement"]
    required_gap = float(placement["minimum_rock_sat_gap_m"])
    boundary = float(placement["boundary_clearance_m"])
    tolerance = float(placement["relaxation_tolerance_m"])
    domain_low = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    domain_high = np.asarray(config["bank"]["nominal_bounds_max_m"], dtype=float)
    deep_threshold = float(placement["deep_depth_threshold_m"])
    converged_iteration = None
    last_violation = math.inf
    for iteration in range(int(placement["relaxation_iterations"])):
        corrections = np.zeros((len(placed), 3), dtype=float)
        maximum_violation = 0.0
        # Project each true transformed mesh into the occupied stepped bank.
        for index, rock in enumerate(placed):
            vertices = rock["vertices_array"]
            for axis in (0, 1):
                lower = domain_low[axis] + boundary - float(vertices[:, axis].min())
                upper = domain_high[axis] - boundary - float(vertices[:, axis].max())
                if lower > 0.0:
                    corrections[index, axis] += lower
                    maximum_violation = max(maximum_violation, lower)
                if upper < 0.0:
                    corrections[index, axis] += upper
                    maximum_violation = max(maximum_violation, -upper)
            projected_vertices = vertices.copy()
            projected_vertices[:, :2] += corrections[index, :2]
            ground = boundary - float(projected_vertices[:, 2].min())
            if ground > 0.0:
                corrections[index, 2] += ground
                maximum_violation = max(maximum_violation, ground)
            surfaces = _surface_at_xy(projected_vertices[:, :2], surface_grid, config)
            if not np.isfinite(surfaces).all():
                raise ValueError("rock_left_occupied_xy_domain_during_relaxation")
            top = float(np.max(projected_vertices[:, 2] - (surfaces - boundary)))
            if top > 0.0:
                corrections[index, 2] -= top
                maximum_violation = max(maximum_violation, top)
            if rock["size_class"] != "small_0p16_0p24":
                center_surface = float(_surface_at_xy(
                    rock["position_array"][None, :2], surface_grid, config)[0])
                depth_shortfall = (deep_threshold + boundary
                                   - (center_surface - rock["position_array"][2]))
                if depth_shortfall > 0.0:
                    corrections[index, 2] -= depth_shortfall
                    maximum_violation = max(maximum_violation, depth_shortfall)

        for first_index, second_index in _near_pairs(placed, required_gap):
            first, second = placed[first_index], placed[second_index]
            if float(np.linalg.norm(first["position_array"] - second["position_array"])) \
                    > first["radius"] + second["radius"] + required_gap:
                continue
            gap, axis = _sat_gap_axis(first, second)
            if gap + tolerance >= required_gap:
                continue
            amount = required_gap - gap
            direction = 1.0 if np.dot(second["position_array"] - first["position_array"], axis) >= 0 else -1.0
            delta = axis * direction * (0.51 * amount)
            corrections[first_index] -= delta
            corrections[second_index] += delta
            maximum_violation = max(maximum_violation, amount)

        # Damped simultaneous projection is deterministic and avoids order bias.
        corrections = np.clip(corrections, -0.03, 0.03)
        for index, rock in enumerate(placed):
            rock["position_array"] += 0.55 * corrections[index]
            rock["vertices_array"] = rock["local_vertices"] + rock["position_array"]
        last_violation = maximum_violation
        if os.environ.get("FULLSCALE_GRAVEL_DEBUG") and iteration % 20 == 0:
            print(f"relax iteration={iteration} max_violation={maximum_violation:.9g}",
                  file=__import__("sys").stderr, flush=True)
        if maximum_violation <= tolerance:
            converged_iteration = iteration + 1
            break
    if converged_iteration is None:
        raise ValueError(
            "layered_grid_sat_relaxation_failed:"
            f"rocks={len(placed)}:iterations={placement['relaxation_iterations']}:"
            f"last_max_violation_m={last_violation:.12g}:required_gap_m={required_gap:.12g}")

    for rock in placed:
        vertices = rock["vertices_array"]
        surfaces = _surface_at_xy(vertices[:, :2], surface_grid, config)
        relative = vertices[:, 2] - surfaces
        center_surface = float(_surface_at_xy(
            rock["position_array"][None, :2], surface_grid, config)[0])
        rock.update({
            "center_depth_below_local_surface_m": center_surface - rock["position_array"][2],
            "aabb_min_world_m": vertices.min(axis=0).tolist(),
            "aabb_max_world_m": vertices.max(axis=0).tolist(),
            "visible_vertex_count": int(np.count_nonzero(relative >= 0.0)),
            "maximum_vertex_height_above_local_surface_m": float(relative.max()),
            "minimum_vertex_height_above_local_surface_m": float(relative.min()),
            "grid_exposed": bool(relative.max() >= -0.75 * float(
                config["bank"]["particle_spacing_m"])),
            "placement_method": "deterministic_layered_grid_with_sat_relaxation",
            "relaxation_iterations": converged_iteration,
        })
    placed.sort(key=lambda item: item["source_index"])
    width = max(3, len(str(len(placed) - 1)))
    for index, rock in enumerate(placed):
        rock["rock_id"] = f"fullscale-rock-{index:0{width}d}"
    return placed


def _inside_convex(points: np.ndarray, vertices: np.ndarray,
                   faces: np.ndarray, tolerance=1e-10) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    center = vertices.mean(axis=0)
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, center - a) > 0.0:
            normal = -normal
        inside &= ((points - a) @ normal <= tolerance * max(1.0, np.linalg.norm(normal)))
        if not inside.any():
            break
    return inside


def _carve(q: np.ndarray, placed: list[dict], prototypes: dict[str, dict]) -> np.ndarray:
    removed = np.zeros(len(q), dtype=bool)
    for rock in placed:
        low = np.asarray(rock["aabb_min_world_m"])
        high = np.asarray(rock["aabb_max_world_m"])
        candidates = np.flatnonzero(
            (~removed) & np.all(q >= low - 1e-10, axis=1)
            & np.all(q <= high + 1e-10, axis=1))
        if not len(candidates):
            continue
        local = ((q[candidates].astype(float) - rock["position_array"])
                 @ rock["rotation_array"] / rock["diameter_m"])
        prototype = prototypes[rock["prototype_id"]]
        inside = _inside_convex(local, prototype["vertices"], prototype["faces"])
        removed[candidates[inside]] = True
    return removed


def _pair_metrics(placed: list[dict], prototypes: dict[str, dict]) -> tuple[int, float]:
    intersections = 0
    minimum_gap = math.inf
    candidates = []
    for first, second in itertools.combinations(placed, 2):
        lower_bound = (float(np.linalg.norm(first["position_array"] - second["position_array"]))
                       - first["radius"] - second["radius"])
        candidates.append((lower_bound, first, second))
    candidates.sort(key=lambda item: item[0])
    for lower_bound, first, second in candidates:
        if lower_bound >= minimum_gap:
            break
        intersects, gap = _sat_precomputed(
            first["vertices_array"], first["face_normals"], first["edge_directions"],
            second["vertices_array"], second["face_normals"], second["edge_directions"])
        intersections += int(intersects)
        minimum_gap = min(minimum_gap, gap)
    return intersections, minimum_gap


def _manifest(config: dict, placed: list[dict]) -> dict:
    records = []
    for rock in placed:
        records.append({
            "rock_id": rock["rock_id"],
            "prototype_id": rock["prototype_id"],
            "size_class": rock["size_class"],
            "equivalent_diameter_m": rock["diameter_m"],
            "uniform_linear_scale_from_unit_asset": rock["diameter_m"],
            "density_kg_m3": config["rocks"]["density_kg_m3"],
            "volume_m3": rock["volume_m3"],
            "mass_kg": rock["mass_kg"],
            "inertia_about_com_body_kg_m2": rock["inertia_about_com_body_kg_m2"],
            "position_world_m": rock["position_array"].tolist(),
            "orientation_quaternion_wxyz": rock["orientation_quaternion_wxyz"],
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "asset_path": f"{config['rocks']['asset_root']}/{rock['prototype_id']}.obj",
            "asset_sha256": config["rocks"]["assets"][rock["prototype_id"]],
            "aabb_min_world_m": rock["aabb_min_world_m"],
            "aabb_max_world_m": rock["aabb_max_world_m"],
            "visibility": {
                "classification": "grid_exposed" if rock["grid_exposed"] else "embedded",
                "visible_vertex_count": rock["visible_vertex_count"],
                "maximum_vertex_height_above_local_surface_m": (
                    rock["maximum_vertex_height_above_local_surface_m"]),
                "minimum_vertex_height_above_local_surface_m": (
                    rock["minimum_vertex_height_above_local_surface_m"]),
                "method": "transformed_mesh_vertices_relative_to_frozen_particle_column_surface",
            },
            "placement_diagnostics": {
                "method": rock["placement_method"],
                "relaxation_iterations": rock["relaxation_iterations"],
                "center_depth_below_local_surface_m": (
                    rock["center_depth_below_local_surface_m"]),
            },
        })
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "deterministic_geometry_candidate",
        "scenario_name": config["scenario_name"],
        "seed": config["seed"],
        "coordinate_frame": "world",
        "scope": "D-060 60/30/8/2 volume recipe and carved MPM bank geometry only; no physics",
        "rock_count": len(records),
        "rocks": records,
    }


def _metrics(config: dict, q: np.ndarray, depth: np.ndarray, removed: np.ndarray,
             placed: list[dict], prototypes: dict[str, dict]) -> dict:
    retained = ~removed
    particle_volume = float(config["bank"]["particle_spacing_m"]) ** 3
    particle_mass = float(config["bank"]["particle_density_kg_m3"]) * particle_volume
    intersections, minimum_gap = _pair_metrics(placed, prototypes)
    class_metrics = {}
    deep_threshold = float(config["placement"]["deep_depth_threshold_m"])
    for name, low, high, target, fraction in FROZEN_CLASSES:
        selected = [rock for rock in placed if rock["size_class"] == name]
        actual = sum(rock["volume_m3"] for rock in selected)
        exposed = [rock for rock in selected if rock["grid_exposed"]]
        depths = [rock["center_depth_below_local_surface_m"] for rock in selected]
        class_metrics[name] = {
            "count": len(selected),
            "estimated_count_from_uniform_distribution": (
                target / (math.pi / 6.0 * (high ** 4 - low ** 4) / (4.0 * (high - low)))),
            "target_fraction": fraction,
            "target_volume_m3": target,
            "analytic_rigid_volume_m3": actual,
            "relative_target_volume_error": abs(actual - target) / target,
            "mass_kg": sum(rock["mass_kg"] for rock in selected),
            "diameter_min_m": min(rock["diameter_m"] for rock in selected),
            "diameter_max_m": max(rock["diameter_m"] for rock in selected),
            "grid_exposed_count": len(exposed),
            "grid_exposed_volume_m3": sum(rock["volume_m3"] for rock in exposed),
            "grid_exposed_count_fraction": len(exposed) / len(selected),
            "center_depth_min_m": min(depths),
            "center_depth_median_m": float(np.median(depths)),
            "center_depth_max_m": max(depths),
            "center_at_least_0p45m_deep_count": sum(value >= deep_threshold for value in depths),
            "center_at_least_0p45m_deep_fraction": (
                sum(value >= deep_threshold for value in depths) / len(depths)),
        }
    all_vertices = np.concatenate([rock["vertices_array"] for rock in placed], axis=0)
    visible_count = sum(rock["grid_exposed"] for rock in placed)
    prototype_counts = Counter(rock["prototype_id"] for rock in placed)
    prototype_counts_by_class = {
        name: dict(sorted(Counter(rock["prototype_id"] for rock in placed
                                  if rock["size_class"] == name).items()))
        for name, *_ in FROZEN_CLASSES
    }
    rigid_volume = sum(rock["volume_m3"] for rock in placed)
    retained_mpm_volume = float(retained.sum()) * particle_volume
    combined_volume = rigid_volume + retained_mpm_volume
    target_total = float(config["bank"]["nominal_volume_m3"])
    mpm_error = abs(retained_mpm_volume - FROZEN_MPM_VOLUME_M3) / FROZEN_MPM_VOLUME_M3
    total_error = abs(combined_volume - target_total) / target_total
    visible_by_class = {name: class_metrics[name]["grid_exposed_count"]
                        for name, *_ in FROZEN_CLASSES}
    total_visible = sum(visible_by_class.values())
    medium_depth = class_metrics["medium_0p24_0p32"]["center_depth_median_m"]
    large_depth = class_metrics["large_0p32_0p40"]["center_depth_median_m"]
    checks = {
        "fullscale_nominal_volume_preserved": math.isclose(
            float(config["bank"]["nominal_volume_m3"]), 6.1, abs_tol=1e-12),
        "count_derived_from_volume_not_fixed": len(placed) != 240,
        "rigid_class_volume_targets": all(
            class_metrics[name]["relative_target_volume_error"]
            <= float(config["material_recipe"]["relative_rigid_class_volume_tolerance"])
            for name, *_ in FROZEN_CLASSES),
        "balanced_prototypes_per_class": all(
            max(counts.values()) - min(counts.values()) <= 1 and len(counts) == 4
            for counts in prototype_counts_by_class.values()),
        "particle_partition": int(retained.sum()) + int(removed.sum()) == len(q),
        "mpm_fines_nonempty": bool(retained.any()),
        "rock_rock_no_intersection": intersections == 0,
        "minimum_rock_sat_gap": minimum_gap + 1e-12
            >= float(config["placement"]["minimum_rock_sat_gap_m"]),
        "ground_penetration_zero": float(all_vertices[:, 2].min()) >= -1e-12,
        "medium_depth_bias": class_metrics["medium_0p24_0p32"][
            "center_at_least_0p45m_deep_fraction"]
            >= float(config["placement"]["minimum_medium_deep_fraction"]),
        "large_depth_bias": class_metrics["large_0p32_0p40"][
            "center_at_least_0p45m_deep_fraction"]
            >= float(config["placement"]["minimum_large_deep_fraction"]),
        "large_deeper_than_medium": large_depth >= medium_depth + float(
            config["placement"]["large_median_depth_margin_over_medium_m"]),
        "small_dominates_grid_exposure": total_visible > 0 and (
            visible_by_class["small_0p16_0p24"] == max(visible_by_class.values())
            and visible_by_class["small_0p16_0p24"] / total_visible
            >= float(config["placement"]["small_visible_share_minimum"])),
        "mpm_target_within_two_percent": mpm_error <= 0.02,
        "combined_equivalent_volume_within_tolerance": total_error
            <= float(config["material_recipe"]["relative_total_volume_tolerance"]),
        "retained_particle_rock_overlap_zero": True,
        "physics_not_executed": config["claims"]["physics_executed"] is False,
        "physics_not_qualified": config["claims"]["physics_qualified"] is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("compiled_scene_check_failed:" + ",".join(failed))
    return {
        "schema": METRICS_SCHEMA,
        "scenario_name": config["scenario_name"],
        "seed": config["seed"],
        "nominal_bank_bounds_min_m": config["bank"]["nominal_bounds_min_m"],
        "nominal_bank_bounds_max_m": config["bank"]["nominal_bounds_max_m"],
        "nominal_bank_volume_m3": config["bank"]["nominal_volume_m3"],
        "particle_mass_kg": particle_mass,
        "particle_equivalent_volume_m3": particle_volume,
        "original_particle_count": len(q),
        "original_particle_mass_kg": len(q) * particle_mass,
        "source_reported_initial_mass_kg": config["bank"]["source_initial_mass_kg"],
        "retained_particle_count": int(retained.sum()),
        "retained_particle_mass_kg": float(retained.sum()) * particle_mass,
        "carved_particle_count": int(removed.sum()),
        "carved_particle_mass_kg": float(removed.sum()) * particle_mass,
        "retained_mpm_target_volume_m3": FROZEN_MPM_VOLUME_M3,
        "retained_mpm_equivalent_volume_m3": retained_mpm_volume,
        "retained_mpm_relative_target_error": mpm_error,
        "rock_count": len(placed),
        "rock_total_analytic_solid_volume_m3": rigid_volume,
        "rock_total_mass_kg": sum(rock["mass_kg"] for rock in placed),
        "combined_equivalent_solid_volume_m3": combined_volume,
        "combined_relative_nominal_volume_error": total_error,
        "size_classes": class_metrics,
        "prototype_counts": dict(sorted(prototype_counts.items())),
        "prototype_counts_by_class": prototype_counts_by_class,
        "visible_rock_count": visible_count,
        "visibility_method": "true_mesh_vertices_within_0.75_particle_spacing_of_frozen_column_surface",
        "grid_exposed_count_by_class": visible_by_class,
        "rock_rock_sat_intersection_count": intersections,
        "minimum_rock_rock_sat_gap_m": minimum_gap,
        "minimum_rock_vertex_z_m": float(all_vertices[:, 2].min()),
        "combined_geometry_bounds_min_m": np.minimum(q[retained].min(axis=0),
                                                     all_vertices.min(axis=0)).tolist(),
        "combined_geometry_bounds_max_m": np.maximum(q[retained].max(axis=0),
                                                     all_vertices.max(axis=0)).tolist(),
        "placement_method": "deterministic_layered_grid_with_sat_relaxation",
        "placement_relaxation_iterations": max(rock["relaxation_iterations"]
                                                for rock in placed),
        "checks": checks,
        "all_geometry_checks_passed": all(checks.values()),
        "pending_checks": {
            "mechanism_initial_mesh_clearance": "pending_mainline_geometry_binding",
            "truck_initial_mesh_clearance": "pending_mainline_geometry_binding",
            "mpm_material_runtime": "not_executed",
        },
    }


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    return result.getvalue()


def _preview_svg(config: dict, q: np.ndarray, depth: np.ndarray,
                 removed: np.ndarray, placed: list[dict]) -> str:
    width, height = 1200, 760
    side_box = (70.0, 80.0, 500.0, 500.0)
    top_box = (650.0, 80.0, 500.0, 500.0)
    nominal_min = np.asarray(config["bank"]["nominal_bounds_min_m"], dtype=float)
    nominal_max = np.asarray(config["bank"]["nominal_bounds_max_m"], dtype=float)
    colors = {"small_0p16_0p24": "#6baed6", "medium_0p24_0p32": "#fd8d3c",
              "large_0p32_0p40": "#de2d26"}

    def sx(x):
        return side_box[0] + (x - nominal_min[0]) / (nominal_max[0] - nominal_min[0]) * side_box[2]
    def sz(z):
        return side_box[1] + side_box[3] - z / nominal_max[2] * side_box[3]
    def tx(x):
        return top_box[0] + (x - nominal_min[0]) / (nominal_max[0] - nominal_min[0]) * top_box[2]
    def ty(y):
        return top_box[1] + top_box[3] - (y - nominal_min[1]) / (nominal_max[1] - nominal_min[1]) * top_box[3]

    # Frozen bank side silhouette, reconstructed from the particle columns.
    spacing = float(config["bank"]["particle_spacing_m"])
    x_values = np.unique(q[:, 0].astype(float))
    surface = [float(q[np.isclose(q[:, 0], x, atol=spacing / 10), 2].max())
               + spacing / 2.0 for x in x_values]
    side_polygon = [(sx(nominal_min[0]), sz(0.0))]
    side_polygon += [(sx(x - spacing / 2), sz(z)) for x, z in zip(x_values, surface)]
    side_polygon += [(sx(x_values[-1] + spacing / 2), sz(0.0))]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in side_polygon)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1200" height="760" fill="#f7f5ef"/>',
        '<text x="600" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Full-scale 6.1 m³ graded gravel slope — geometry candidate</text>',
        f'<rect x="{side_box[0]}" y="{side_box[1]}" width="{side_box[2]}" height="{side_box[3]}" fill="#ffffff" stroke="#333"/>',
        f'<polygon points="{points}" fill="#8c6d31" opacity="0.85"/>',
        f'<text x="{side_box[0] + side_box[2]/2}" y="{side_box[1]-18}" text-anchor="middle" font-family="sans-serif" font-size="18">Side view (x-z), complete 2.2 × 1.6 m envelope</text>',
        f'<rect x="{top_box[0]}" y="{top_box[1]}" width="{top_box[2]}" height="{top_box[3]}" fill="#bdb76b" opacity="0.55" stroke="#333"/>',
        f'<text x="{top_box[0] + top_box[2]/2}" y="{top_box[1]-18}" text-anchor="middle" font-family="sans-serif" font-size="18">Top view (x-y), complete 2.2 × 2.2 m envelope</text>',
    ]
    for rock in placed:
        color = colors[rock["size_class"]]
        radius_side = rock["diameter_m"] / (nominal_max[0] - nominal_min[0]) * side_box[2] / 2
        radius_top = rock["diameter_m"] / (nominal_max[0] - nominal_min[0]) * top_box[2] / 2
        opacity = "0.82" if rock["grid_exposed"] else "0.18"
        dash = "" if rock["grid_exposed"] else ' stroke-dasharray="3 2"'
        lines.append(f'<circle cx="{sx(rock["position_array"][0]):.2f}" cy="{sz(rock["position_array"][2]):.2f}" r="{radius_side:.2f}" fill="{color}" fill-opacity="{opacity}" stroke="#222" stroke-width="0.6"{dash}/>')
        lines.append(f'<circle cx="{tx(rock["position_array"][0]):.2f}" cy="{ty(rock["position_array"][1]):.2f}" r="{radius_top:.2f}" fill="{color}" fill-opacity="{opacity}" stroke="#222" stroke-width="0.6"{dash}/>')
    lines += [
        '<text x="70" y="625" font-family="sans-serif" font-size="15">Brown: retained MPM fines; rocks use D-060 60/8/2% analytic volume recipe</text>',
        '<text x="70" y="650" font-family="sans-serif" font-size="15">Solid rock: grid-exposed true mesh; translucent dashed rock: embedded</text>',
        f'<circle cx="80" cy="684" r="8" fill="#6baed6"/><text x="96" y="690" font-family="sans-serif" font-size="15">{sum(r["size_class"] == "small_0p16_0p24" for r in placed)} small</text>',
        f'<circle cx="220" cy="684" r="8" fill="#fd8d3c"/><text x="236" y="690" font-family="sans-serif" font-size="15">{sum(r["size_class"] == "medium_0p24_0p32" for r in placed)} medium</text>',
        f'<circle cx="380" cy="684" r="8" fill="#de2d26"/><text x="396" y="690" font-family="sans-serif" font-size="15">{sum(r["size_class"] == "large_0p32_0p40" for r in placed)} large</text>',
        f'<text x="650" y="625" font-family="sans-serif" font-size="15">Original particles: {len(q)}; carved centers: {int(removed.sum())}</text>',
        '<text x="650" y="650" font-family="sans-serif" font-size="15">Mechanism/truck clearance and all physics remain pending.</text>',
        '</svg>',
    ]
    return "\n".join(lines) + "\n"


def _readme(config: dict, metrics: dict, report: dict) -> str:
    return f"""# 全尺寸分层碎石坡离线候选

状态：`deterministic_geometry_candidate`。本目录按D-060从冻结的6.1m³、390400粒子坡体编译出{metrics['rock_count']}块显式石体。小/中/大刚体石分别闭合到3.66/0.488/0.122m³；数量由确定性直径序列和目标体积反算，不使用固定240石口径。

真实凸石体采用R0四种冻结资产，每档尽量均衡；粒子中心落入石体才被carve，不额外删除粒子。石—石SAT相交为0，最小SAT间隙`{metrics['minimum_rock_rock_sat_gap_m']:.6f}m`。carve后保留MPM等效体积`{metrics['retained_mpm_equivalent_volume_m3']:.6f}m³`，刚体加MPM合计`{metrics['combined_equivalent_solid_volume_m3']:.6f}m³`，相对6.1m³误差`{metrics['combined_relative_nominal_volume_error']:.3%}`。

这只是CPU/NumPy离线几何和质量清单：`geometry_only=true`、`physics_executed=false`、`physics_qualified=false`。机构初始网格和实际装载车网格尚待主线绑定，因此二者净空未通过；MPM物性、正式N石动力学、沉降、铲运、装车、连续三铲和训练均未执行。

复现：

```bash
PYTHONPATH=src/v4 python3 src/v4/build_fullscale_graded_gravel_slope.py \\
  --config configs/v4/fullscale-graded-gravel-slope-v1.json \\
  --output-dir /new/nonexistent/output/path
```

配置SHA256：`{report['config_sha256']}`；编译器SHA256：`{report['implementation_sha256']}`。输出目录必须不存在，失败不会留下部分正式输出。
"""


def build(config_path: str | Path, output_dir: str | Path,
          repository_root: str | Path | None = None) -> dict:
    started = __import__("time").perf_counter()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_dir_exists:{output_dir}")
    repository = (Path(repository_root).resolve() if repository_root is not None
                  else Path(__file__).resolve().parents[2])
    config = load_config(config_path)
    q = _load_frozen_bank(config, repository)
    prototypes = _load_prototypes(config, repository)
    surface_grid, _, depth = _column_surface(q, config)
    _plateau_bounds(surface_grid, config)  # Frozen full-height shape sanity check.
    placed = _place_rocks(config, prototypes, surface_grid, q)
    removed = _carve(q, placed, prototypes)
    retained = ~removed
    material_id_all = np.full(len(q), config["bank"]["material_ids"]["mpm_fines"],
                              dtype=np.uint8)
    manifest = _manifest(config, placed)
    metrics = _metrics(config, q, depth, removed, placed, prototypes)

    # Recheck retained particle centers against every rock after carve.
    for rock in placed:
        low, high = np.asarray(rock["aabb_min_world_m"]), np.asarray(rock["aabb_max_world_m"])
        candidate = np.flatnonzero(retained & np.all(q >= low - 1e-10, axis=1)
                                   & np.all(q <= high + 1e-10, axis=1))
        if len(candidate):
            local = ((q[candidate].astype(float) - rock["position_array"])
                     @ rock["rotation_array"] / rock["diameter_m"])
            prototype = prototypes[rock["prototype_id"]]
            if _inside_convex(local, prototype["vertices"], prototype["faces"]).any():
                raise ValueError(f"retained_particle_rock_overlap:{rock['rock_id']}")

    npz_bytes = _deterministic_npz_bytes({
        "q": q[retained],
        "material_id": material_id_all[retained],
        "original_index": np.flatnonzero(retained).astype(np.int64),
        "removed_index": np.flatnonzero(removed).astype(np.int64),
    })
    preview = _preview_svg(config, q, depth, removed, placed)
    manifest_bytes = _json_bytes(manifest)
    metrics_bytes = _json_bytes(metrics)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "deterministic_geometry_candidate",
        "scenario_name": config["scenario_name"],
        "geometry_only": True,
        "physics_executed": False,
        "physics_qualified": False,
        "training_executed": False,
        "three_scoop_qualified": False,
        "config_sha256": sha256(config_path),
        "implementation_sha256": sha256(Path(__file__)),
        "inputs": {
            "initial_bank_npz_sha256": FROZEN_BANK_SHA256,
            "q_array_sha256": FROZEN_Q_SHA256,
            "solid_bank_source_sha256": FROZEN_SOLID_BANK_SHA256,
            "r0_geometry_source_sha256": FROZEN_R0_SOURCE_SHA256,
            "rock_asset_sha256": FROZEN_ASSETS,
        },
        "outputs": {
            "scene_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "scene_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "layered_bank_npz_sha256": hashlib.sha256(npz_bytes).hexdigest(),
            "preview_svg_sha256": hashlib.sha256(preview.encode("utf-8")).hexdigest(),
        },
        "checks": metrics["checks"],
        "pending_checks": metrics["pending_checks"],
        "elapsed_wall_seconds_diagnostic_only": __import__("time").perf_counter() - started,
    }
    # Runtime is diagnostic and would defeat byte determinism. Preserve it only as
    # an explicitly null field in the formal output; callers may time the build.
    report["elapsed_wall_seconds_diagnostic_only"] = None
    report_bytes = _json_bytes(report)
    readme = _readme(config, metrics, report)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-",
                                      dir=output_dir.parent))
    try:
        (temporary / "scene-manifest.json").write_bytes(manifest_bytes)
        (temporary / "scene-metrics.json").write_bytes(metrics_bytes)
        (temporary / "layered-bank.npz").write_bytes(npz_bytes)
        (temporary / "preview.svg").write_text(preview, encoding="utf-8", newline="\n")
        (temporary / "report.json").write_bytes(report_bytes)
        (temporary / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.config, args.output_dir)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
