"""Deterministic, geometry-only compiler for visible rock scene candidates.

The compiler consumes explicit measured/reviewed/synthetic parameters and the
frozen R0 convex assets.  It does not import or advance MuJoCo, Newton or Warp.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable

import numpy as np

from mixed_material_r0 import load_obj, mass_properties, quaternion_matrix_wxyz


CONFIG_SCHEMA = "excavation_scene_reconstruction_config_v1"
MANIFEST_SCHEMA = "excavation_scene_reconstruction_manifest_v1"
METRICS_SCHEMA = "excavation_scene_reconstruction_metrics_v1"
REPORT_SCHEMA = "excavation_scene_reconstruction_report_v1"
INDEX_SCHEMA = "excavation_scene_reconstruction_index_v1"
ALLOWED_SOURCE_STATUSES = {
    "measured", "human_reviewed_estimate", "synthetic_assumption", "unknown"
}
SCENE_CLASSES = {
    "ordinary-small-rock-dominant", "sparse-large-rock", "jam-boundary"
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_json_value(value):
    """Remove platform-specific floating-point tail noise before serialization."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non_finite_json_value")
        if numeric == 0.0:
            return 0.0
        return float(f"{numeric:.10g}")
    if isinstance(value, list):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported_json_type:{type(value).__name__}")


def _json_bytes(value) -> bytes:
    return (json.dumps(_stable_json_value(value), indent=2, sort_keys=True,
                       allow_nan=False)
            + "\n").encode("utf-8")


def write_json(path: str | Path, value) -> None:
    Path(path).write_bytes(_json_bytes(value))


def _canonical_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_finite(value, path: str = "$") -> None:
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


def _cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_area(open_polygon: np.ndarray) -> float:
    following = np.roll(open_polygon, -1, axis=0)
    return 0.5 * float(np.sum(
        open_polygon[:, 0] * following[:, 1]
        - open_polygon[:, 1] * following[:, 0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray,
                        c: np.ndarray, d: np.ndarray) -> bool:
    def orientation(p, q, r):
        return _cross2(q - p, r - p)

    values = (orientation(a, b, c), orientation(a, b, d),
              orientation(c, d, a), orientation(c, d, b))
    return values[0] * values[1] < -1e-14 and values[2] * values[3] < -1e-14


def _validate_polygon(raw_polygon) -> np.ndarray:
    polygon = np.asarray(raw_polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 4:
        raise ValueError("invalid_region_polygon")
    if not np.array_equal(polygon[0], polygon[-1]):
        raise ValueError("region_polygon_not_closed")
    open_polygon = polygon[:-1]
    if len(open_polygon) < 3 or abs(_polygon_area(open_polygon)) <= 1e-12:
        raise ValueError("degenerate_region_polygon")
    edge_count = len(open_polygon)
    for first in range(edge_count):
        for second in range(first + 1, edge_count):
            if second in {first, (first + 1) % edge_count}:
                continue
            if first == 0 and second == edge_count - 1:
                continue
            if _segments_intersect(
                    open_polygon[first], open_polygon[(first + 1) % edge_count],
                    open_polygon[second], open_polygon[(second + 1) % edge_count]):
                raise ValueError("self_intersecting_region_polygon")
    return open_polygon


def _validate_config(config: dict) -> None:
    _assert_finite(config)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_scene_reconstruction_schema")
    if config.get("coordinate_frame") != "world":
        raise ValueError("unsupported_coordinate_frame")
    sources = config.get("sources", {})
    if not sources:
        raise ValueError("missing_parameter_sources")
    for source_id, source in sources.items():
        if source.get("status") not in ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"unknown_source_status:{source_id}")
        if not source.get("source_id") or not source.get("measurement_method"):
            raise ValueError(f"incomplete_source_record:{source_id}")
        if not isinstance(source.get("human_reviewed"), bool):
            raise ValueError(f"invalid_source_review_flag:{source_id}")
    required_parameter_sources = {
        "/region", "/surface", "/placement", "/scene_classes",
        "/rocks/assets", "/rocks/density_kg_m3", "/rocks/prototype_mix",
    }
    parameter_sources = config.get("parameter_sources", {})
    if set(parameter_sources) != required_parameter_sources:
        raise ValueError("incomplete_parameter_source_map")
    if any(source_id not in sources for source_id in parameter_sources.values()):
        raise ValueError("unknown_parameter_source_reference")

    _validate_polygon(config["region"]["polygon_xy_m"])
    if float(config["region"]["boundary_safety_distance_m"]) < 0.0:
        raise ValueError("negative_boundary_safety_distance")
    base_quaternion = np.asarray(
        config["rocks"]["orientation_base_quaternion_wxyz"], dtype=float)
    if base_quaternion.shape != (4,) or not math.isclose(
            float(np.linalg.norm(base_quaternion)), 1.0, abs_tol=1e-12):
        raise ValueError("orientation_base_quaternion_not_unit")
    if float(config["rocks"]["density_kg_m3"]) <= 0.0:
        raise ValueError("nonpositive_rock_density")
    assets = config["rocks"].get("assets", {})
    mix = config["rocks"].get("prototype_mix", {})
    if not assets or set(assets) != set(mix):
        raise ValueError("asset_mix_mismatch")
    if any(len(value) != 64 for value in assets.values()):
        raise ValueError("invalid_asset_sha256")
    if any(float(value) < 0.0 for value in mix.values()) or not math.isclose(
            sum(map(float, mix.values())), 1.0, abs_tol=1e-12):
        raise ValueError("invalid_prototype_mix")

    placement = config["placement"]
    positive_fields = (
        "ground_clearance_m", "minimum_rock_gap_m",
        "narrow_phase_neighbor_margin_m", "maximum_attempts_per_rock",
        "strata_columns", "strata_rows", "local_density_radius_m",
    )
    if any(float(placement[key]) <= 0.0 for key in positive_fields):
        raise ValueError("nonpositive_placement_parameter")
    coverage_resolution = placement["coverage_grid_resolution_xy"]
    if len(coverage_resolution) != 2 or any(int(value) < 10 for value in coverage_resolution):
        raise ValueError("invalid_coverage_grid_resolution")
    total_spatial_weight = float(placement["background_weight"])
    for cluster in placement["clusters"]:
        if any(float(value) <= 0.0 for value in cluster["sigma_xy_m"]):
            raise ValueError("invalid_cluster_sigma")
        if float(cluster["weight"]) < 0.0:
            raise ValueError("invalid_cluster_weight")
        total_spatial_weight += float(cluster["weight"])
    if not math.isclose(total_spatial_weight, 1.0, abs_tol=1e-12):
        raise ValueError("spatial_weights_do_not_sum_to_one")

    scene_classes = config.get("scene_classes", [])
    if {item.get("scene_class") for item in scene_classes} != SCENE_CLASSES:
        raise ValueError("required_scene_classes_missing")
    all_seeds: list[int] = []
    admission = config["admission"]
    regular_with_limit = 0
    regular_total = 0
    for item in scene_classes:
        if item.get("primary_constraint") != "rock_count":
            raise ValueError("unsupported_primary_constraint")
        count = int(item["rock_count"])
        if count <= 0 or sum(int(entry["count"]) for entry in item["diameter_bins"]) != count:
            raise ValueError("diameter_bin_count_mismatch")
        if len(item.get("seeds", [])) != 3:
            raise ValueError("each_scene_class_requires_three_seeds")
        all_seeds.extend(int(value) for value in item["seeds"])
        descriptors = []
        for entry in item["diameter_bins"]:
            low, high = float(entry["minimum_m"]), float(entry["maximum_m"])
            if low <= 0.0 or high < low or int(entry["count"]) <= 0:
                raise ValueError("invalid_diameter_bin")
            descriptors.extend([0.5 * (low + high)] * int(entry["count"]))
        small_fraction = sum(
            1 for value in descriptors
            if value <= float(admission["small_maximum_diameter_m"]) + 1e-12
        ) / count
        if item["scene_class"] != "jam-boundary":
            regular_total += len(item["seeds"])
            if small_fraction + 1e-12 < float(admission["minimum_small_count_fraction"]):
                raise ValueError("regular_scene_small_fraction_below_minimum")
            large_count = sum(
                1 for value in descriptors
                if float(admission["large_minimum_diameter_m"]) <= value
                <= float(admission["large_maximum_diameter_m"])
            )
            if large_count > int(admission["maximum_large_count"]):
                raise ValueError("regular_scene_large_count_above_maximum")
            if any(value >= 0.27 - 1e-12 for value in descriptors):
                regular_with_limit += len(item["seeds"])
            if any(value >= float(admission["jam_minimum_diameter_m"]) for value in descriptors):
                raise ValueError("jam_rock_leaked_into_regular_scene")
        elif not any(
                value >= float(admission["jam_minimum_diameter_m"])
                for value in descriptors):
            raise ValueError("jam_scene_missing_jam_boundary_rock")
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("duplicate_scene_seed")
    if regular_with_limit / regular_total > float(
            admission["maximum_regular_scene_fraction_with_0p27m"]) + 1e-12:
        raise ValueError("regular_0p27m_scene_fraction_above_maximum")


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _surface_height(surface: dict, x: float, y: float) -> float:
    origin_x, origin_y = map(float, surface["origin_xy_m"])
    value = (float(surface["base_height_m"])
             + float(surface["slope_dz_dx"]) * (x - origin_x)
             + float(surface["slope_dz_dy"]) * (y - origin_y))
    for bump in surface["bumps"]:
        center_x, center_y = map(float, bump["center_xy_m"])
        sigma_x, sigma_y = map(float, bump["sigma_xy_m"])
        exponent = -0.5 * (
            ((x - center_x) / sigma_x) ** 2
            + ((y - center_y) / sigma_y) ** 2)
        value += float(bump["amplitude_m"]) * math.exp(exponent)
    return value


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = map(float, point)
    inside = False
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = map(float, first)
        x2, y2 = map(float, second)
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _point_segment_distance(point: np.ndarray, first: np.ndarray,
                            second: np.ndarray) -> float:
    delta = second - first
    scale = float(np.dot(point - first, delta) / np.dot(delta, delta))
    closest = first + min(1.0, max(0.0, scale)) * delta
    return float(np.linalg.norm(point - closest))


def _boundary_clearance(points: np.ndarray, polygon: np.ndarray) -> float:
    minimum = math.inf
    for point in points:
        if not _point_in_polygon(point, polygon):
            return -math.inf
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            minimum = min(minimum, _point_segment_distance(point, first, second))
    return minimum


def _uniform_quaternion(generator: random.Random) -> np.ndarray:
    first, second, third = generator.random(), generator.random(), generator.random()
    x = math.sqrt(1.0 - first) * math.sin(2.0 * math.pi * second)
    y = math.sqrt(1.0 - first) * math.cos(2.0 * math.pi * second)
    z = math.sqrt(first) * math.sin(2.0 * math.pi * third)
    w = math.sqrt(first) * math.cos(2.0 * math.pi * third)
    result = np.array([w, x, y, z], dtype=float)
    if result[0] < 0.0:
        result = -result
    return result


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    value = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    value /= np.linalg.norm(value)
    if value[0] < 0.0:
        value = -value
    return value


def _load_prototypes(config: dict, repository_root: Path) -> dict[str, dict]:
    result = {}
    asset_root = repository_root / config["rocks"]["asset_root"]
    for prototype_id, expected_hash in sorted(config["rocks"]["assets"].items()):
        path = asset_root / f"{prototype_id}.obj"
        if sha256(path) != expected_hash:
            raise ValueError(f"asset_sha256_mismatch:{prototype_id}")
        vertices, faces = load_obj(path)
        properties = mass_properties(vertices, faces)
        equivalent_diameter = (6.0 * properties["volume"] / math.pi) ** (1.0 / 3.0)
        if not math.isclose(equivalent_diameter, 1.0, abs_tol=1e-12):
            raise ValueError(f"asset_not_unit_equivalent_diameter:{prototype_id}")
        if float(np.linalg.norm(properties["center"])) > 1e-12:
            raise ValueError(f"asset_not_com_centered:{prototype_id}")
        result[prototype_id] = {
            "path": path,
            "path_repository": str(path.relative_to(repository_root).as_posix()),
            "sha256": expected_hash,
            "vertices": vertices,
            "faces": faces,
            "unit_volume_m3": float(properties["volume"]),
            "unit_inertia_m5": properties["inertia_unit_density"],
            "unit_bounding_radius_m": float(np.linalg.norm(vertices, axis=1).max()),
        }
    return result


def _unique_edges(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    keys: set[tuple[int, int]] = set()
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


def _sat_axis_gap(first: np.ndarray, first_faces: np.ndarray,
                  second: np.ndarray, second_faces: np.ndarray) -> tuple[bool, float]:
    axes: list[np.ndarray] = []
    for vertices, faces in ((first, first_faces), (second, second_faces)):
        for face in faces:
            a, b, c = vertices[face]
            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm > 1e-14:
                axes.append(normal / norm)
    for first_edge in _unique_edges(first, first_faces):
        for second_edge in _unique_edges(second, second_faces):
            axis = np.cross(first_edge, second_edge)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-10:
                axes.append(axis / norm)
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for axis in axes:
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis = -axis
        unique.setdefault(tuple(np.round(axis, 12)), axis)
    maximum_separating_gap = -math.inf
    for axis in unique.values():
        first_projection = first @ axis
        second_projection = second @ axis
        gap = max(float(second_projection.min() - first_projection.max()),
                  float(first_projection.min() - second_projection.max()))
        maximum_separating_gap = max(maximum_separating_gap, gap)
    return maximum_separating_gap <= 1e-12, maximum_separating_gap


def _diameter_descriptors(scene_config: dict, generator: random.Random) -> list[dict]:
    descriptors = []
    for bin_config in scene_config["diameter_bins"]:
        count = int(bin_config["count"])
        low, high = float(bin_config["minimum_m"]), float(bin_config["maximum_m"])
        for index in range(count):
            fraction = (index + 0.5) / count
            descriptors.append({
                "diameter_m": low + (high - low) * fraction,
                "bin_id": bin_config["bin_id"],
            })
    generator.shuffle(descriptors)
    return descriptors


def _prototype_assignments(config: dict, count: int,
                           generator: random.Random) -> list[str]:
    weights = config["rocks"]["prototype_mix"]
    exact = {key: float(value) * count for key, value in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = count - sum(counts.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    assignments = [key for key in sorted(counts) for _ in range(counts[key])]
    generator.shuffle(assignments)
    return assignments


def _background_candidate(generator: random.Random, polygon: np.ndarray,
                          placement: dict, sequence_index: int) -> np.ndarray:
    columns, rows = int(placement["strata_columns"]), int(placement["strata_rows"])
    cell_count = columns * rows
    cell = sequence_index % cell_count
    column, row = cell % columns, cell // columns
    minimum = polygon.min(axis=0)
    maximum = polygon.max(axis=0)
    fraction = np.array([
        (column + generator.random()) / columns,
        (row + generator.random()) / rows,
    ])
    return minimum + fraction * (maximum - minimum)


def _candidate_xy(generator: random.Random, polygon: np.ndarray, placement: dict,
                  sequence_index: int) -> np.ndarray:
    choice = generator.random()
    background_weight = float(placement["background_weight"])
    if choice < background_weight:
        return _background_candidate(generator, polygon, placement, sequence_index)
    choice -= background_weight
    for cluster in placement["clusters"]:
        if choice < float(cluster["weight"]):
            center = np.asarray(cluster["center_xy_m"], dtype=float)
            sigma = np.asarray(cluster["sigma_xy_m"], dtype=float)
            return np.array([
                generator.gauss(float(center[0]), float(sigma[0])),
                generator.gauss(float(center[1]), float(sigma[1])),
            ])
        choice -= float(cluster["weight"])
    raise AssertionError("unreachable_spatial_weight_selection")


def _support_position_z(local_vertices: np.ndarray, xy: np.ndarray,
                        surface: dict, clearance: float) -> float:
    required = [
        _surface_height(surface, float(xy[0] + vertex[0]),
                        float(xy[1] + vertex[1])) - float(vertex[2])
        for vertex in local_vertices
    ]
    return max(required) + clearance


def _build_scene(config: dict, scene_config: dict, seed: int,
                 prototypes: dict[str, dict]) -> tuple[dict, dict]:
    generator = random.Random(seed)
    polygon = _validate_polygon(config["region"]["polygon_xy_m"])
    boundary_requirement = float(config["region"]["boundary_safety_distance_m"])
    placement = config["placement"]
    surface = config["surface"]
    density = float(config["rocks"]["density_kg_m3"])
    descriptors = _diameter_descriptors(scene_config, generator)
    assignments = _prototype_assignments(config, len(descriptors), generator)
    items = []
    for source_index, (descriptor, prototype_id) in enumerate(zip(descriptors, assignments)):
        item = dict(descriptor)
        item.update({"prototype_id": prototype_id, "source_index": source_index})
        items.append(item)
    items.sort(key=lambda item: (-item["diameter_m"], item["source_index"]))

    placed: list[dict] = []
    attempts_total = 0
    maximum_attempts = int(placement["maximum_attempts_per_rock"])
    base_quaternion = np.asarray(
        config["rocks"]["orientation_base_quaternion_wxyz"], dtype=float)
    for placement_index, item in enumerate(items):
        prototype = prototypes[item["prototype_id"]]
        diameter = float(item["diameter_m"])
        quaternion = _quaternion_multiply(base_quaternion, _uniform_quaternion(generator))
        rotation = quaternion_matrix_wxyz(quaternion)
        local_vertices = prototype["vertices"] * diameter @ rotation.T
        radius = prototype["unit_bounding_radius_m"] * diameter
        accepted = None
        for attempt in range(maximum_attempts):
            attempts_total += 1
            xy = _candidate_xy(
                generator, polygon, placement,
                placement_index * maximum_attempts + attempt)
            z = _support_position_z(
                local_vertices, xy, surface, float(placement["ground_clearance_m"]))
            position = np.array([xy[0], xy[1], z], dtype=float)
            world_vertices = local_vertices + position
            boundary_clearance = _boundary_clearance(world_vertices[:, :2], polygon)
            if boundary_clearance + 1e-12 < boundary_requirement:
                continue
            minimum_surface_clearance = min(
                float(vertex[2]) - _surface_height(
                    surface, float(vertex[0]), float(vertex[1]))
                for vertex in world_vertices)
            if minimum_surface_clearance < float(placement["ground_clearance_m"]) - 1e-10:
                continue
            rejected = False
            pair_gaps = []
            narrow_checks = 0
            for previous in placed:
                center_distance = float(np.linalg.norm(position - previous["position_array"]))
                sphere_gap = center_distance - radius - previous["bounding_radius"]
                if sphere_gap > float(placement["narrow_phase_neighbor_margin_m"]):
                    continue
                narrow_checks += 1
                intersects, sat_gap = _sat_axis_gap(
                    world_vertices, prototype["faces"], previous["vertices_array"],
                    prototypes[previous["prototype_id"]]["faces"])
                pair_gaps.append(sat_gap)
                if intersects or sat_gap + 1e-12 < float(placement["minimum_rock_gap_m"]):
                    rejected = True
                    break
            if rejected:
                continue
            accepted = {
                "position_array": position,
                "vertices_array": world_vertices,
                "boundary_clearance": boundary_clearance,
                "surface_clearance": minimum_surface_clearance,
                "bounding_radius": radius,
                "quaternion": quaternion,
                "narrow_checks_on_accept": narrow_checks,
                "accept_pair_gaps": pair_gaps,
                "attempt_count": attempt + 1,
            }
            break
        if accepted is None:
            raise ValueError(
                f"scene_placement_failed:{scene_config['scene_class']}:"
                f"seed={seed}:rock={placement_index}:attempts={maximum_attempts}")
        volume = prototype["unit_volume_m3"] * diameter ** 3
        mass = density * volume
        inertia_body = density * prototype["unit_inertia_m5"] * diameter ** 5
        inertia_world = rotation @ inertia_body @ rotation.T
        world_vertices = accepted["vertices_array"]
        placed.append({
            **accepted,
            "rock_id": f"{scene_config['scene_class']}-{seed}-rock-{placement_index:03d}",
            "prototype_id": item["prototype_id"],
            "size_bucket": item["bin_id"],
            "equivalent_diameter_m": diameter,
            "uniform_linear_scale_from_unit_asset": diameter,
            "density_kg_m3": density,
            "volume_m3": volume,
            "mass_kg": mass,
            "inertia_about_com_body_kg_m2": inertia_body.tolist(),
            "inertia_about_com_world_kg_m2": inertia_world.tolist(),
            "position_world_m": accepted["position_array"].tolist(),
            "orientation_quaternion_wxyz": accepted["quaternion"].tolist(),
            "linear_velocity_world_mps": [0.0, 0.0, 0.0],
            "angular_velocity_world_radps": [0.0, 0.0, 0.0],
            "asset_path": prototype["path_repository"],
            "asset_sha256": prototype["sha256"],
            "parameter_source": config["parameter_sources"]["/scene_classes"],
            "asset_source": config["parameter_sources"]["/rocks/assets"],
            "bounding_volume": {
                "type": "sphere_and_world_aabb",
                "sphere_radius_m": radius,
                "aabb_min_world_m": world_vertices.min(axis=0).tolist(),
                "aabb_max_world_m": world_vertices.max(axis=0).tolist(),
            },
            "placement_diagnostics": {
                "attempt_count": accepted["attempt_count"],
                "boundary_clearance_m": accepted["boundary_clearance"],
                "minimum_vertex_surface_clearance_m": accepted["surface_clearance"],
                "narrow_phase_checks_on_accept": accepted["narrow_checks_on_accept"],
            },
        })
    manifest_rocks = []
    for rock in placed:
        manifest_rocks.append({
            key: value for key, value in rock.items()
            if not key.endswith("_array") and key not in {
                "vertices_array", "bounding_radius", "quaternion",
                "boundary_clearance", "surface_clearance", "accept_pair_gaps",
                "narrow_checks_on_accept", "attempt_count",
            }
        })
    scene_id = f"{scene_config['scene_class']}-seed-{seed}"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "geometry_reconstruction_candidate",
        "scope": "geometry only; not physics qualified or training ready",
        "scene_id": scene_id,
        "scene_class": scene_config["scene_class"],
        "seed": seed,
        "coordinate_frame": config["coordinate_frame"],
        "region_polygon_xy_m": config["region"]["polygon_xy_m"],
        "surface_model": config["surface"],
        "parameter_sources": config["parameter_sources"],
        "rock_count": len(manifest_rocks),
        "placement_attempts_total": attempts_total,
        "rocks": manifest_rocks,
    }
    metrics = _compute_metrics(config, manifest, placed, prototypes)
    return manifest, metrics


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(item[0]), float(item[1])) for item in points})
    if len(ordered) <= 1:
        return np.asarray(ordered, dtype=float)
    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))
    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _coverage_fraction(config: dict, placed: list[dict]) -> float:
    polygon = _validate_polygon(config["region"]["polygon_xy_m"])
    minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
    columns, rows = map(int, config["placement"]["coverage_grid_resolution_xy"])
    hulls = [_convex_hull_2d(rock["vertices_array"][:, :2]) for rock in placed]
    x_values = minimum[0] + (np.arange(columns) + 0.5) / columns * (
        maximum[0] - minimum[0])
    y_values = minimum[1] + (np.arange(rows) + 0.5) / rows * (
        maximum[1] - minimum[1])
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    def vectorized_inside(target_polygon: np.ndarray) -> np.ndarray:
        inside = np.zeros(len(points), dtype=bool)
        x, y = points[:, 0], points[:, 1]
        for first, second in zip(
                target_polygon, np.roll(target_polygon, -1, axis=0)):
            crosses = (first[1] > y) != (second[1] > y)
            intersection_x = ((second[0] - first[0]) * (y - first[1])
                              / (second[1] - first[1] + 1e-300) + first[0])
            inside ^= crosses & (x < intersection_x)
        return inside

    inside_region = vectorized_inside(polygon)
    covered = np.zeros(len(points), dtype=bool)
    for hull in hulls:
        covered |= vectorized_inside(hull)
    sample_count = int(inside_region.sum())
    if not sample_count:
        raise ValueError("coverage_grid_has_no_region_samples")
    return int((covered & inside_region).sum()) / sample_count


def _compute_metrics(config: dict, manifest: dict, placed: list[dict],
                     prototypes: dict[str, dict]) -> dict:
    diameters = np.asarray([rock["equivalent_diameter_m"] for rock in placed])
    volumes = np.asarray([rock["volume_m3"] for rock in placed])
    centers = np.asarray([rock["position_world_m"] for rock in placed])
    admission = config["admission"]
    small_mask = diameters <= float(admission["small_maximum_diameter_m"]) + 1e-12
    large_mask = ((diameters >= float(admission["large_minimum_diameter_m"]) - 1e-12)
                  & (diameters <= float(admission["large_maximum_diameter_m"]) + 1e-12))
    jam_mask = diameters >= float(admission["jam_minimum_diameter_m"]) - 1e-12
    nearest = []
    sat_gaps = []
    intersections = 0
    for first_index, first in enumerate(placed):
        distances = [
            float(np.linalg.norm(centers[first_index] - centers[second_index]))
            for second_index in range(len(placed)) if second_index != first_index
        ]
        nearest.append(min(distances) if distances else None)
    for first, second in itertools.combinations(placed, 2):
        intersects, gap = _sat_axis_gap(
            first["vertices_array"], prototypes[first["prototype_id"]]["faces"],
            second["vertices_array"], prototypes[second["prototype_id"]]["faces"])
        intersections += int(intersects)
        sat_gaps.append(gap)
    bin_counts: dict[str, int] = {}
    prototype_counts: dict[str, int] = {}
    for rock in placed:
        bin_counts[rock["size_bucket"]] = bin_counts.get(rock["size_bucket"], 0) + 1
        prototype_counts[rock["prototype_id"]] = prototype_counts.get(
            rock["prototype_id"], 0) + 1
    density_radius = float(config["placement"]["local_density_radius_m"])
    local_neighbor_counts = [
        sum(1 for second_index in range(len(placed))
            if first_index != second_index
            and np.linalg.norm(centers[first_index, :2] - centers[second_index, :2])
            <= density_radius)
        for first_index in range(len(placed))
    ]
    regular = manifest["scene_class"] != "jam-boundary"
    large_volume_fraction = float(volumes[large_mask].sum() / volumes.sum())
    checks = {
        "rock_count_preserved": len(placed) == manifest["rock_count"],
        "minimum_small_count_fraction": (
            not regular or float(np.mean(small_mask)) + 1e-12
            >= float(admission["minimum_small_count_fraction"])),
        "maximum_regular_large_count": (
            not regular or int(large_mask.sum()) <= int(admission["maximum_large_count"])),
        "maximum_regular_large_volume_fraction": (
            not regular or large_volume_fraction
            <= float(admission["maximum_large_volume_fraction"]) + 1e-12),
        "jam_isolation": (not bool(jam_mask.any()) if regular else bool(jam_mask.any())),
        "region_and_boundary_clearance": min(
            rock["boundary_clearance"] for rock in placed)
            >= float(config["region"]["boundary_safety_distance_m"]) - 1e-10,
        "surface_support": min(rock["surface_clearance"] for rock in placed)
            >= float(config["placement"]["ground_clearance_m"]) - 1e-10,
        "convex_narrow_phase_no_intersection": intersections == 0,
        "minimum_rock_gap": (not sat_gaps or min(sat_gaps)
            >= float(config["placement"]["minimum_rock_gap_m"]) - 1e-10),
        "finite_values": True,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError("scene_admission_failed:" + ",".join(failed))
    return {
        "schema": METRICS_SCHEMA,
        "scene_id": manifest["scene_id"],
        "scene_class": manifest["scene_class"],
        "seed": manifest["seed"],
        "rock_count": len(placed),
        "diameter_histogram_count": dict(sorted(bin_counts.items())),
        "diameter_count_fraction": {
            key: value / len(placed) for key, value in sorted(bin_counts.items())
        },
        "diameter_d50_m": float(np.quantile(diameters, 0.5)),
        "diameter_d95_m": float(np.quantile(diameters, 0.95)),
        "diameter_max_m": float(diameters.max()),
        "small_count": int(small_mask.sum()),
        "small_count_fraction": float(np.mean(small_mask)),
        "large_count": int(large_mask.sum()),
        "large_equivalent_volume_fraction": large_volume_fraction,
        "jam_boundary_count": int(jam_mask.sum()),
        "prototype_counts": dict(sorted(prototype_counts.items())),
        "total_mass_kg": float(sum(rock["mass_kg"] for rock in placed)),
        "total_equivalent_solid_volume_m3": float(volumes.sum()),
        "projected_surface_coverage_fraction": _coverage_fraction(config, placed),
        "projected_surface_coverage_method": (
            "deterministic cell-center union estimate of transformed convex XY hulls"
        ),
        "coverage_grid_resolution_xy": config["placement"]["coverage_grid_resolution_xy"],
        "nearest_neighbor_center_distance_m": {
            "minimum": min(value for value in nearest if value is not None),
            "mean": float(np.mean([value for value in nearest if value is not None])),
            "per_rock": nearest,
        },
        "minimum_sat_separating_axis_gap_m": min(sat_gaps) if sat_gaps else None,
        "minimum_boundary_clearance_m": min(
            rock["boundary_clearance"] for rock in placed),
        "minimum_vertex_surface_clearance_m": min(
            rock["surface_clearance"] for rock in placed),
        "local_density_radius_m": density_radius,
        "local_neighbor_count": {
            "minimum": min(local_neighbor_counts),
            "maximum": max(local_neighbor_counts),
            "mean": float(np.mean(local_neighbor_counts)),
            "per_rock": local_neighbor_counts,
        },
        "placement_attempts_total": manifest["placement_attempts_total"],
        "admission_checks": checks,
        "all_admission_checks_passed": all(checks.values()),
    }


def _svg_points(points: np.ndarray, mapping) -> str:
    return " ".join(f"{mapping(point)[0]:.3f},{mapping(point)[1]:.3f}" for point in points)


def _preview_svg(config: dict, manifest: dict, placed: list[dict]) -> str:
    polygon = _validate_polygon(config["region"]["polygon_xy_m"])
    minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
    width, height = maximum - minimum
    top_x, top_y, top_w, top_h = 55.0, 55.0, 670.0, 570.0
    side_x, side_y, side_w, side_h = 770.0, 55.0, 370.0, 570.0
    z_samples = [
        _surface_height(config["surface"], float(x), 0.0)
        for x in np.linspace(minimum[0], maximum[0], 120)
    ]
    z_values = z_samples + [float(vertex[2]) for rock in placed
                            for vertex in rock["vertices_array"]]
    z_min, z_max = min(z_values) - 0.04, max(z_values) + 0.04

    def top_map(point):
        return (top_x + (float(point[0]) - minimum[0]) / width * top_w,
                top_y + (maximum[1] - float(point[1])) / height * top_h)

    def side_map(point):
        return (side_x + (float(point[0]) - minimum[0]) / width * side_w,
                side_y + (z_max - float(point[1])) / (z_max - z_min) * side_h)

    colors = {
        "small_le_0p16": "#4c78a8", "medium_0p16_0p22": "#f2cf5b",
        "large_0p22_0p27": "#f58518", "jam_gt_0p285": "#d62728",
    }
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680">',
        '<rect width="1200" height="680" fill="#fbfbf8"/>',
        f'<text x="55" y="28" font-family="sans-serif" font-size="18">{manifest["scene_id"]}</text>',
        '<text x="55" y="49" font-family="sans-serif" font-size="12">top view; world X/Y in metres</text>',
        '<text x="770" y="49" font-family="sans-serif" font-size="12">side view; X/Z in metres</text>',
        f'<polygon points="{_svg_points(polygon, top_map)}" fill="#ece8dc" stroke="#333" stroke-width="2"/>',
    ]
    for rock in placed:
        hull = _convex_hull_2d(rock["vertices_array"][:, :2])
        color = colors.get(rock["size_bucket"], "#777")
        lines.append(
            f'<polygon points="{_svg_points(hull, top_map)}" fill="{color}" '
            'fill-opacity="0.72" stroke="#222" stroke-width="0.8"/>')
    surface_points = np.asarray([
        [x, _surface_height(config["surface"], float(x), 0.0)]
        for x in np.linspace(minimum[0], maximum[0], 120)
    ])
    lines.append(
        f'<polyline points="{_svg_points(surface_points, side_map)}" fill="none" '
        'stroke="#6b5b35" stroke-width="2"/>')
    for rock in placed:
        hull = _convex_hull_2d(rock["vertices_array"][:, [0, 2]])
        color = colors.get(rock["size_bucket"], "#777")
        lines.append(
            f'<polygon points="{_svg_points(hull, side_map)}" fill="{color}" '
            'fill-opacity="0.62" stroke="#222" stroke-width="0.7"/>')
    legend_y = 646
    for index, (key, color) in enumerate(colors.items()):
        x = 55 + index * 220
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="16" height="12" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 22}" y="{legend_y - 1}" font-family="sans-serif" '
            f'font-size="11">{key}</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def _scene_readme(manifest: dict, metrics: dict) -> str:
    checks = "\n".join(
        f"- `{key}`: {'passed' if value else 'failed'}"
        for key, value in metrics["admission_checks"].items())
    return f"""# {manifest['scene_id']}

Status: `geometry_reconstruction_candidate`.

This directory is a deterministic, geometry-only candidate. It is not an
official-scene reconstruction, physics qualification, or training input.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_reconstruction.py \\
  --config configs/v4/scene-reconstruction-v1.json \\
  --output-dir reports/v4/scene-reconstruction-2026-09-19-v1
```

## Admission

{checks}

The next interface is the R0 asset ID, body-frame inertia, world pose and
coordinate-frame mapping recorded in `scene-manifest.json`. The R1/R2 mainline
must repeat collision and support checks in its actual physics representation.
"""


def _root_readme(index: dict) -> str:
    scene_lines = "\n".join(
        f"- `{item['scene_id']}`: {item['status']}"
        for item in index["scenes"])
    return f"""# Scene reconstruction v1

Status: `geometry_reconstruction_candidate`.

Nine deterministic geometry candidates were compiled from explicitly marked
synthetic parameters and the frozen R0 convex assets. They are not an official
scene reconstruction and have no physics or training qualification.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_reconstruction.py \\
  --config configs/v4/scene-reconstruction-v1.json \\
  --output-dir reports/v4/scene-reconstruction-2026-09-19-v1
```

The output directory must not already exist. Generate into two fresh temporary
directories and compare files to verify byte-for-byte determinism.

## Scenes

{scene_lines}

## Boundary

The compiler only establishes explicit provenance, stable IDs, R0-compatible
mass/inertia, surface support, boundary clearance and convex SAT non-intersection.
R1/R2 must independently qualify the actual collision proxies, settling and
rock/bucket dynamics before any scene may be called physics-qualified.
"""


def build(config_path: str | Path, output_dir: str | Path,
          repository_root: str | Path | None = None) -> dict:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_directory_exists:{output_dir}")
    root = (Path(repository_root).resolve() if repository_root is not None
            else Path(__file__).resolve().parents[2])
    config = load_config(config_path)
    prototypes = _load_prototypes(config, root)
    config_hash = sha256(config_path)
    generator_hash = sha256(Path(__file__))
    input_hash = _canonical_hash({
        key: config[key] for key in (
            "coordinate_frame", "sources", "parameter_sources", "region",
            "surface", "rocks", "placement", "scene_classes", "admission")
    })
    asset_hashes = {
        key: prototypes[key]["sha256"] for key in sorted(prototypes)
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        scene_records = []
        for scene_config in config["scene_classes"]:
            for seed in scene_config["seeds"]:
                manifest, metrics = _build_scene(
                    config, scene_config, int(seed), prototypes)
                scene_directory = temporary / manifest["scene_id"]
                scene_directory.mkdir()
                manifest_path = scene_directory / "scene-manifest.json"
                metrics_path = scene_directory / "scene-metrics.json"
                preview_path = scene_directory / "preview.svg"
                write_json(manifest_path, manifest)
                write_json(metrics_path, metrics)
                placed = []
                for rock in manifest["rocks"]:
                    prototype = prototypes[rock["prototype_id"]]
                    rotation = quaternion_matrix_wxyz(
                        rock["orientation_quaternion_wxyz"])
                    vertices = (prototype["vertices"]
                                * rock["uniform_linear_scale_from_unit_asset"]
                                @ rotation.T
                                + np.asarray(rock["position_world_m"]))
                    placed.append({**rock, "vertices_array": vertices})
                preview_path.write_text(
                    _preview_svg(config, manifest, placed), encoding="utf-8",
                    newline="\n")
                report = {
                    "schema": REPORT_SCHEMA,
                    "status": "geometry_reconstruction_candidate",
                    "scope": "geometry only; not official reconstruction, physics or training",
                    "scene_id": manifest["scene_id"],
                    "scene_class": manifest["scene_class"],
                    "seed": manifest["seed"],
                    "input_sha256": input_hash,
                    "config_sha256": config_hash,
                    "generator_sha256": generator_hash,
                    "asset_sha256": asset_hashes,
                    "scene_manifest_sha256": sha256(manifest_path),
                    "scene_metrics_sha256": sha256(metrics_path),
                    "preview_sha256": sha256(preview_path),
                    "all_admission_checks_passed": metrics["all_admission_checks_passed"],
                    "boundaries": [
                        "synthetic parameters do not reproduce an official scene",
                        "no underground material or mechanical properties inferred",
                        "R1/R2 collision, settling and dynamics must be requalified",
                    ],
                }
                report_path = scene_directory / "report.json"
                write_json(report_path, report)
                (scene_directory / "README.md").write_text(
                    _scene_readme(manifest, metrics), encoding="utf-8", newline="\n")
                scene_records.append({
                    "scene_id": manifest["scene_id"],
                    "scene_class": manifest["scene_class"],
                    "seed": manifest["seed"],
                    "status": report["status"],
                    "report_sha256": sha256(report_path),
                    "all_admission_checks_passed": report["all_admission_checks_passed"],
                })
        index = {
            "schema": INDEX_SCHEMA,
            "status": "geometry_reconstruction_candidate",
            "scope": "nine deterministic geometry candidates; no physics or training",
            "input_sha256": input_hash,
            "config_sha256": config_hash,
            "generator_sha256": generator_hash,
            "asset_sha256": asset_hashes,
            "scene_count": len(scene_records),
            "scenes": scene_records,
            "all_scenes_passed": all(
                item["all_admission_checks_passed"] for item in scene_records),
        }
        if len(scene_records) != 9 or not index["all_scenes_passed"]:
            raise ValueError("scene_reconstruction_index_not_admitted")
        write_json(temporary / "report.json", index)
        (temporary / "README.md").write_text(
            _root_readme(index), encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
        return index
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    index = build(args.config, args.output_dir, args.repository_root)
    print(json.dumps({
        "status": index["status"],
        "scene_count": index["scene_count"],
        "all_scenes_passed": index["all_scenes_passed"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
