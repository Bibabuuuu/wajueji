"""R0 geometry, rock-asset and deterministic-scene contract builder.

This module is deliberately geometry-only.  It does not import MuJoCo,
Newton or Warp and cannot advance physics or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np


CONFIG_SCHEMA = "excavation_mixed_material_r0_config_v1"
REPORT_SCHEMA = "excavation_mixed_material_r0_report_v1"
GAUGE_SCHEMA = "excavation_bucket_aperture_gauge_v1"
SCENE_SCHEMA = "excavation_mixed_material_scene_manifest_v1"
ASSET_SCHEMA = "excavation_rigid_rock_asset_manifest_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line_number, raw in enumerate(
            Path(path).read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"invalid_obj_vertex:{line_number}")
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            ids = [int(value.split("/", 1)[0]) for value in fields[1:]]
            ids = [value - 1 if value > 0 else len(vertices) + value for value in ids]
            if len(ids) < 3:
                raise ValueError(f"invalid_obj_face:{line_number}")
            for index in range(1, len(ids) - 1):
                faces.append([ids[0], ids[index], ids[index + 1]])
    result_vertices = np.asarray(vertices, dtype=float)
    result_faces = np.asarray(faces, dtype=int)
    if (result_vertices.ndim != 2 or result_vertices.shape[1:] != (3,)
            or result_faces.ndim != 2 or result_faces.shape[1:] != (3,)
            or not len(result_faces) or not np.isfinite(result_vertices).all()
            or result_faces.min() < 0 or result_faces.max() >= len(result_vertices)):
        raise ValueError("invalid_obj_mesh")
    return result_vertices, result_faces


def quaternion_matrix_wxyz(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("invalid_quaternion")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-15:
        raise ValueError("invalid_quaternion")
    w, x, y, z = value / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_quaternion_wxyz(rotation) -> list[float]:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid_rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            scale / 4.0,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1]
                              - matrix[2, 2]) * 2.0
            q = np.array([
                (matrix[2, 1] - matrix[1, 2]) / scale,
                scale / 4.0,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0]
                              - matrix[2, 2]) * 2.0
            q = np.array([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                scale / 4.0,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0]
                              - matrix[1, 1]) * 2.0
            q = np.array([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                scale / 4.0,
            ])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q.tolist()


def connected_components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    parent = np.arange(vertex_count)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[1]), int(face[2]))
    groups: dict[int, list[int]] = {}
    for index in range(vertex_count):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(group, dtype=int) for group in sorted(
        groups.values(), key=lambda value: (-len(value), value[0]))]


def _cluster_spans(values: np.ndarray, split_gap: float) -> list[list[float]]:
    ordered = np.sort(np.asarray(values, dtype=float))
    if not len(ordered):
        return []
    spans = []
    start = previous = float(ordered[0])
    for item in ordered[1:]:
        current = float(item)
        if current - previous > split_gap:
            spans.append([start, previous])
            start = current
        previous = current
    spans.append([start, previous])
    return spans


def analyze_bucket(bucket_obj: str | Path, inventory_contract: str | Path,
                   config: dict) -> dict:
    bucket_config = config["bucket"]
    if sha256(bucket_obj) != bucket_config["source_sha256"]:
        raise ValueError("bucket_mesh_sha256_mismatch")
    if sha256(inventory_contract) != bucket_config["inventory_contract_sha256"]:
        raise ValueError("inventory_contract_sha256_mismatch")
    contract = json.loads(Path(inventory_contract).read_text(encoding="utf-8"))["bucket"]
    for key in ("y_limits_m", "reference_prism_volume_m3"):
        expected = (bucket_config["interior_y_limits_m"] if key == "y_limits_m"
                    else bucket_config[key])
        if not np.allclose(contract[key], expected, rtol=0, atol=1e-12):
            raise ValueError("inventory_contract_value_mismatch:" + key)
    if not (np.allclose(contract["polygon_xz_m"][0],
                        bucket_config["mouth_endpoints_xz_m"][0], rtol=0, atol=1e-12)
            and np.allclose(contract["polygon_xz_m"][-1],
                            bucket_config["mouth_endpoints_xz_m"][1],
                            rtol=0, atol=1e-12)):
        raise ValueError("inventory_mouth_endpoints_mismatch")

    vertices, faces = load_obj(bucket_obj)
    # MuJoCo's refquat is a reference-frame correction.  The qualified
    # assembler applies its inverse before exposing bucket-body coordinates.
    ref_rotation = quaternion_matrix_wxyz(bucket_config["mesh_refquat_wxyz"])
    corrected = vertices @ ref_rotation
    components = connected_components(len(vertices), faces)
    component_metrics = []
    for component in components:
        values = corrected[component]
        component_metrics.append({
            "vertex_count": int(len(component)),
            "bounds_min_m": values.min(axis=0).tolist(),
            "bounds_max_m": values.max(axis=0).tolist(),
        })

    y_min, y_max = map(float, bucket_config["interior_y_limits_m"])
    negative_candidates = [item for item in component_metrics
                           if item["vertex_count"] >= 100
                           and item["bounds_max_m"][1] <= y_min + 1e-4]
    positive_candidates = [item for item in component_metrics
                           if item["vertex_count"] >= 100
                           and item["bounds_min_m"][1] >= y_max - 1e-4]
    if not negative_candidates or not positive_candidates:
        raise ValueError("side_plate_components_not_found")
    negative_plate = min(negative_candidates,
                         key=lambda item: abs(item["bounds_max_m"][1] - y_min))
    positive_plate = min(positive_candidates,
                         key=lambda item: abs(item["bounds_min_m"][1] - y_max))

    mouth = np.asarray(bucket_config["mouth_endpoints_xz_m"], dtype=float)
    chord = mouth[1] - mouth[0]
    mouth_height = float(np.linalg.norm(chord))
    chord_axis = chord / mouth_height
    candidates = []
    for component in components:
        values = corrected[component]
        if np.ptp(values[:, 1]) >= 0.5:
            projection = (values[:, [0, 2]] - mouth[1]) @ chord_axis
            candidates.append((float(projection.max()), component, projection))
    if not candidates:
        raise ValueError("cutting_edge_component_not_found")
    _, cutting_component, projection = max(candidates, key=lambda item: item[0])
    cutting_vertices = corrected[cutting_component]
    tip_mask = projection >= projection.max() - float(bucket_config["tooth_tip_band_m"])
    tooth_spans = _cluster_spans(
        cutting_vertices[tip_mask, 1], float(bucket_config["tooth_cluster_gap_m"]))
    if len(tooth_spans) != 5:
        raise ValueError("unexpected_tooth_count")
    tooth_gaps = [tooth_spans[index + 1][0] - tooth_spans[index][1]
                  for index in range(len(tooth_spans) - 1)]
    inner_width = y_max - y_min
    d_pass = min(inner_width, mouth_height)
    return {
        "source_path_anchor": bucket_config["source_path_anchor"],
        "source_sha256": bucket_config["source_sha256"],
        "inventory_contract_path_anchor": bucket_config["inventory_contract_path_anchor"],
        "inventory_contract_sha256": bucket_config["inventory_contract_sha256"],
        "coordinate_frame": bucket_config["coordinate_frame"],
        "mesh_refquat_wxyz": bucket_config["mesh_refquat_wxyz"],
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "connected_component_count": int(len(components)),
        "corrected_bounds_min_m": corrected.min(axis=0).tolist(),
        "corrected_bounds_max_m": corrected.max(axis=0).tolist(),
        "inner_side_plate_y_limits_m": [y_min, y_max],
        "inner_clear_width_m": inner_width,
        "side_plate_thickness_m": {
            "negative_y": (negative_plate["bounds_max_m"][1]
                           - negative_plate["bounds_min_m"][1]),
            "positive_y": (positive_plate["bounds_max_m"][1]
                           - positive_plate["bounds_min_m"][1]),
        },
        "mouth_endpoints_xz_m": mouth.tolist(),
        "mouth_chord_m": mouth_height,
        "limiting_spherical_gauge_m": d_pass,
        "limiting_dimension": "inner_clear_width_m",
        "tooth_tip_band_m": bucket_config["tooth_tip_band_m"],
        "tooth_tip_spans_y_m": tooth_spans,
        "tooth_tip_widths_m": [high - low for low, high in tooth_spans],
        "tooth_tip_gaps_m": tooth_gaps,
        "reference_prism_volume_m3": bucket_config["reference_prism_volume_m3"],
        "rigid_contact_constraint": {
            "current_mjcf_bucket_geom_count": 1,
            "current_geom_source": "bucket.obj",
            "mujoco_mesh_collision_semantics": "convex_hull",
            "open_cavity_rigid_contact_qualified": False,
            "r1_requirement": (
                "build and validate an open-cavity convex decomposition or an "
                "equivalent non-convex rigid-contact representation; the current "
                "single mesh must not be used for rock-in-bucket qualification"
            ),
        },
    }


def _orient_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    oriented = faces.copy()
    for index, face in enumerate(oriented):
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0 - center) < 0:
            oriented[index, [1, 2]] = oriented[index, [2, 1]]
    return oriented


def _prototype_sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tetra_vertices = np.array([
        [-0.72, -0.46, -0.34], [0.80, -0.31, -0.28],
        [-0.05, 0.76, -0.23], [0.09, -0.03, 0.91],
    ])
    tetra_faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])

    bipyramid_vertices = np.array([
        [-0.78, -0.38, 0.0], [0.15, -0.62, 0.0],
        [0.83, 0.14, 0.0], [-0.27, 0.71, 0.0],
        [0.04, -0.02, 0.66], [-0.08, 0.05, -0.53],
    ])
    bipyramid_faces = np.array([
        [4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0],
        [5, 1, 0], [5, 2, 1], [5, 3, 2], [5, 0, 3],
    ])

    octa_vertices = np.array([
        [0.91, 0.02, -0.03], [-0.73, -0.02, 0.04],
        [0.03, 0.61, 0.02], [-0.02, -0.54, -0.05],
        [0.08, -0.03, 0.48], [-0.06, 0.04, -0.42],
    ])
    octa_faces = np.array([
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
        [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
    ])

    cube = np.array(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float)
    affine = np.array([[1.45, 0.18, -0.08], [0.0, 0.86, 0.11],
                       [0.07, -0.04, 0.62]])
    block_vertices = cube @ affine.T
    # itertools ordering: 000,001,010,011,100,101,110,111.
    block_faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return {
        "angular_tetra": (tetra_vertices, tetra_faces),
        "angular_bipyramid": (bipyramid_vertices, bipyramid_faces),
        "angular_octa": (octa_vertices, octa_faces),
        "angular_block": (block_vertices, block_faces),
    }


def mass_properties(vertices: np.ndarray, faces: np.ndarray) -> dict:
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for face in faces:
        a, b, c = vertices[face]
        signed_volume = float(np.dot(a, np.cross(b, c)) / 6.0)
        total = a + b + c
        volume += signed_volume
        first += signed_volume * total / 4.0
        second += signed_volume * (
            np.outer(a, a) + np.outer(b, b) + np.outer(c, c)
            + np.outer(total, total)) / 20.0
    if volume <= 1e-12:
        raise ValueError("nonpositive_mesh_volume")
    center = first / volume
    inertia_origin = np.trace(second) * np.eye(3) - second
    inertia_center = inertia_origin - volume * (
        np.dot(center, center) * np.eye(3) - np.outer(center, center))
    inertia_center = (inertia_center + inertia_center.T) / 2.0
    if np.linalg.eigvalsh(inertia_center).min() <= 0:
        raise ValueError("invalid_mesh_inertia")
    return {"volume": volume, "center": center, "inertia_unit_density": inertia_center}


def _closed_convex_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict:
    edges: dict[tuple[int, int], int] = {}
    maximum_plane_violation = 0.0
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        maximum_plane_violation = max(
            maximum_plane_violation, float(np.max((vertices - a) @ normal)))
        for first, second in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(first), int(second))))
            edges[key] = edges.get(key, 0) + 1
    closed = bool(edges) and all(count == 2 for count in edges.values())
    convex = maximum_plane_violation <= 1e-10
    if not closed or not convex:
        raise ValueError("prototype_not_closed_convex")
    return {
        "closed_two_manifold": closed,
        "convex": convex,
        "maximum_face_plane_violation_m": maximum_plane_violation,
    }


def normalized_prototypes(prototype_ids) -> dict[str, dict]:
    sources = _prototype_sources()
    result = {}
    target_volume = math.pi / 6.0
    for prototype_id in prototype_ids:
        if prototype_id not in sources:
            raise ValueError("unknown_prototype:" + prototype_id)
        vertices, faces = sources[prototype_id]
        faces = _orient_faces(vertices, faces)
        initial = mass_properties(vertices, faces)
        vertices = vertices - initial["center"]
        vertices *= (target_volume / initial["volume"]) ** (1.0 / 3.0)
        properties = mass_properties(vertices, faces)
        if np.linalg.norm(properties["center"]) > 1e-12:
            raise ValueError("prototype_center_normalization_failed")
        metrics = _closed_convex_metrics(vertices, faces)
        result[prototype_id] = {
            "vertices": vertices,
            "faces": faces,
            "unit_volume_m3": float(properties["volume"]),
            "unit_inertia_m5": properties["inertia_unit_density"],
            "equivalent_diameter_m": (
                6.0 * properties["volume"] / math.pi) ** (1.0 / 3.0),
            "bounds_min_m": vertices.min(axis=0),
            "bounds_max_m": vertices.max(axis=0),
            "bounding_radius_m": float(np.linalg.norm(vertices, axis=1).max()),
            **metrics,
        }
    return result


def write_obj(path: str | Path, prototype_id: str,
              vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# deterministic R0 closed convex rock proxy", "o " + prototype_id]
    lines += ["v " + " ".join(f"{value:.17g}" for value in vertex)
              for vertex in vertices]
    lines += ["f " + " ".join(str(int(value) + 1) for value in face)
              for face in faces]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_assets(asset_dir: str | Path, config: dict,
                 prototypes: dict[str, dict]) -> dict:
    output = Path(asset_dir)
    output.mkdir(parents=True, exist_ok=True)
    density = float(config["rocks"]["density_kg_m3"])
    records = []
    repository_root = Path(config["rocks"]["repository_asset_root"])
    for prototype_id in config["rocks"]["prototype_ids"]:
        prototype = prototypes[prototype_id]
        path = output / f"{prototype_id}.obj"
        write_obj(path, prototype_id, prototype["vertices"], prototype["faces"])
        records.append({
            "prototype_id": prototype_id,
            "visual_mesh": str((repository_root / path.name).as_posix()),
            "collision_proxy": str((repository_root / path.name).as_posix()),
            "visual_collision_mesh_identical": True,
            "sha256": sha256(path),
            "vertex_count": int(len(prototype["vertices"])),
            "triangle_count": int(len(prototype["faces"])),
            "unit_equivalent_diameter_m": prototype["equivalent_diameter_m"],
            "unit_volume_m3": prototype["unit_volume_m3"],
            "unit_mass_kg_at_configured_density": density * prototype["unit_volume_m3"],
            "unit_inertia_kg_m2_at_configured_density": (
                density * prototype["unit_inertia_m5"]).tolist(),
            "bounds_min_m": prototype["bounds_min_m"].tolist(),
            "bounds_max_m": prototype["bounds_max_m"].tolist(),
            "unit_bounding_radius_m": prototype["bounding_radius_m"],
            "closed_two_manifold": prototype["closed_two_manifold"],
            "convex": prototype["convex"],
            "maximum_face_plane_violation_m": (
                prototype["maximum_face_plane_violation_m"]),
        })
    return {
        "schema": ASSET_SCHEMA,
        "density_kg_m3": density,
        "density_status": "explicit_simulation_assumption_not_official_material_truth",
        "visual_collision_policy": config["rocks"]["visual_collision_policy"],
        "fracture_enabled": config["rocks"]["fracture_enabled"],
        "assets": records,
    }


def proper_axis_rotations() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            for column, row in enumerate(permutation):
                rotation[row, column] = signs[column]
            if np.linalg.det(rotation) > 0.5:
                rotations.append(rotation)
    rotations.sort(key=lambda value: tuple(value.ravel()))
    if len(rotations) != 24:
        raise AssertionError("expected_24_axis_rotations")
    return rotations


def build_gauge(bucket: dict, config: dict,
                prototypes: dict[str, dict]) -> dict:
    rotations = proper_axis_rotations()
    mouth = np.asarray(bucket["mouth_endpoints_xz_m"], dtype=float)
    chord = mouth[1] - mouth[0]
    chord /= np.linalg.norm(chord)
    width_axis = np.array([0.0, 1.0, 0.0])
    chord_axis = np.array([chord[0], 0.0, chord[1]])
    width_limit = float(bucket["inner_clear_width_m"])
    chord_limit = float(bucket["mouth_chord_m"])
    pose_results = []
    summaries = []
    for diameter in config["gauge"]["equivalent_diameters_m"]:
        for prototype_id in config["rocks"]["prototype_ids"]:
            vertices = prototypes[prototype_id]["vertices"] * float(diameter)
            passed = 0
            for pose_index, rotation in enumerate(rotations):
                oriented = vertices @ rotation.T
                width_span = float(np.ptp(oriented @ width_axis))
                chord_span = float(np.ptp(oriented @ chord_axis))
                fits = bool(width_span <= width_limit + 1e-12
                            and chord_span <= chord_limit + 1e-12)
                passed += int(fits)
                pose_results.append({
                    "prototype_id": prototype_id,
                    "equivalent_diameter_m": float(diameter),
                    "pose_id": f"axis24-{pose_index:02d}",
                    "orientation_quaternion_wxyz": matrix_quaternion_wxyz(rotation),
                    "projected_width_m": width_span,
                    "projected_mouth_chord_m": chord_span,
                    "passes_zero_thickness_aperture": fits,
                })
            summaries.append({
                "prototype_id": prototype_id,
                "equivalent_diameter_m": float(diameter),
                "passing_pose_count": passed,
                "tested_pose_count": len(rotations),
                "all_tested_poses_pass": passed == len(rotations),
                "any_tested_pose_passes": passed > 0,
            })
    sphere_gauges = []
    for diameter in config["gauge"]["equivalent_diameters_m"]:
        sphere_gauges.append({
            "diameter_m": float(diameter),
            "passes": bool(float(diameter) <= min(width_limit, chord_limit) + 1e-12),
            "width_clearance_m": width_limit - float(diameter),
            "mouth_chord_clearance_m": chord_limit - float(diameter),
        })
    return {
        "schema": GAUGE_SCHEMA,
        "method": (
            "zero-thickness rectangular mouth aperture in the qualified bucket-body "
            "frame; convex prototype projection over 24 proper signed-axis poses"
        ),
        "scope_limit": (
            "geometric preflight only; it does not prove swept passage through a "
            "validated open-cavity rigid collision decomposition"
        ),
        "inner_clear_width_m": width_limit,
        "mouth_chord_m": chord_limit,
        "limiting_spherical_gauge_m": min(width_limit, chord_limit),
        "sphere_gauges": sphere_gauges,
        "prototype_pose_summaries": summaries,
        "prototype_pose_results": pose_results,
    }


def _uniform_quaternion(generator: random.Random) -> list[float]:
    first, second, third = generator.random(), generator.random(), generator.random()
    x = math.sqrt(1.0 - first) * math.sin(2.0 * math.pi * second)
    y = math.sqrt(1.0 - first) * math.cos(2.0 * math.pi * second)
    z = math.sqrt(first) * math.sin(2.0 * math.pi * third)
    w = math.sqrt(first) * math.cos(2.0 * math.pi * third)
    value = np.array([w, x, y, z])
    if value[0] < 0:
        value = -value
    return value.tolist()


def build_scene_manifest(config: dict, prototypes: dict[str, dict]) -> dict:
    density = float(config["rocks"]["density_kg_m3"])
    prototype_ids = list(config["rocks"]["prototype_ids"])
    scenes = []
    for scene_config in config["scenes"]:
        generator = random.Random(int(scene_config["seed"]))
        rocks = []
        columns = int(scene_config["grid_columns"])
        spacing = float(scene_config["grid_spacing_m"])
        origin = np.asarray(scene_config["grid_origin_xy_m"], dtype=float)
        for index, diameter in enumerate(scene_config["equivalent_diameters_m"]):
            prototype_id = prototype_ids[index % len(prototype_ids)]
            prototype = prototypes[prototype_id]
            scale = float(diameter)
            radius = prototype["bounding_radius_m"] * scale
            row, column = divmod(index, columns)
            position = [float(origin[0] + column * spacing),
                        float(origin[1] + row * spacing),
                        float(radius + scene_config["ground_clearance_m"])]
            volume = prototype["unit_volume_m3"] * scale ** 3
            mass = density * volume
            inertia = density * prototype["unit_inertia_m5"] * scale ** 5
            rocks.append({
                "rock_id": f"{scene_config['scene_id']}-rock-{index:03d}",
                "prototype_id": prototype_id,
                "size_bucket": scene_config["size_bucket"],
                "equivalent_diameter_m": scale,
                "uniform_linear_scale_from_unit_asset": scale,
                "density_kg_m3": density,
                "volume_m3": volume,
                "mass_kg": mass,
                "inertia_about_com_kg_m2": inertia.tolist(),
                "position_world_m": position,
                "orientation_quaternion_wxyz": _uniform_quaternion(generator),
                "linear_velocity_world_mps": [0.0, 0.0, 0.0],
                "angular_velocity_world_radps": [0.0, 0.0, 0.0],
                "bounding_sphere_radius_m": radius,
            })
        minimum_pair_clearance = math.inf
        for first, second in itertools.combinations(rocks, 2):
            distance = float(np.linalg.norm(
                np.asarray(first["position_world_m"])
                - np.asarray(second["position_world_m"])))
            clearance = (distance - first["bounding_sphere_radius_m"]
                         - second["bounding_sphere_radius_m"])
            minimum_pair_clearance = min(minimum_pair_clearance, clearance)
        if len(rocks) < 2:
            minimum_pair_clearance = None
        ground_clearances = [rock["position_world_m"][2]
                             - rock["bounding_sphere_radius_m"] for rock in rocks]
        no_overlap = (minimum_pair_clearance is None or minimum_pair_clearance >= -1e-12)
        ground_clear = min(ground_clearances) >= -1e-12
        if not no_overlap or not ground_clear:
            raise ValueError("scene_initial_overlap")
        scenes.append({
            "scene_id": scene_config["scene_id"],
            "seed": int(scene_config["seed"]),
            "size_bucket": scene_config["size_bucket"],
            "rock_count": len(rocks),
            "equivalent_diameter_d95_m": float(np.quantile(
                np.asarray(scene_config["equivalent_diameters_m"], dtype=float), 0.95)),
            "total_rock_mass_kg": sum(rock["mass_kg"] for rock in rocks),
            "minimum_bounding_sphere_pair_clearance_m": minimum_pair_clearance,
            "minimum_ground_clearance_m": min(ground_clearances),
            "initial_overlap_check": "passed_conservative_bounding_spheres",
            "rocks": rocks,
        })
    return {
        "schema": SCENE_SCHEMA,
        "coordinate_frame": "world",
        "deterministic_generation": True,
        "mass_inertia_rule": "m scales as s^3; inertia about COM scales as s^5",
        "scenes": scenes,
    }


def build(config_path: str | Path, bucket_obj: str | Path,
          inventory_contract: str | Path, asset_dir: str | Path) -> tuple[dict, dict, dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_r0_config_schema")
    prototypes = normalized_prototypes(config["rocks"]["prototype_ids"])
    bucket = analyze_bucket(bucket_obj, inventory_contract, config)
    assets = build_assets(asset_dir, config, prototypes)
    gauge = build_gauge(bucket, config, prototypes)
    scenes = build_scene_manifest(config, prototypes)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "scope": "R0 geometry and asset contract only; no physics and no training",
        "config_sha256": sha256(config_path),
        "bucket": bucket,
        "asset_manifest": assets,
        "gauge_summary": {
            "schema": gauge["schema"],
            "method": gauge["method"],
            "scope_limit": gauge["scope_limit"],
            "inner_clear_width_m": gauge["inner_clear_width_m"],
            "mouth_chord_m": gauge["mouth_chord_m"],
            "limiting_spherical_gauge_m": gauge["limiting_spherical_gauge_m"],
            "sphere_gauges": gauge["sphere_gauges"],
            "prototype_pose_summaries": gauge["prototype_pose_summaries"],
        },
        "scene_summary": [{key: scene[key] for key in (
            "scene_id", "seed", "size_bucket", "rock_count",
            "total_rock_mass_kg", "minimum_bounding_sphere_pair_clearance_m",
            "equivalent_diameter_d95_m", "minimum_ground_clearance_m",
            "initial_overlap_check")}
            for scene in scenes["scenes"]],
        "admission": {
            "appearance_collision_dimensions_consistent": True,
            "all_prototypes_closed_convex": True,
            "mass_and_inertia_recomputable": True,
            "deterministic_scene_manifests": True,
            "initial_overlap_checks_passed": True,
            "bucket_rigid_contact_open_cavity_qualified": False,
            "mechanism_rock_load_envelope_qualified": False,
            "mechanism_load_note": (
                "scene masses are reported but no official rock density or qualified "
                "mechanism rock-load envelope exists; R1 must not infer capacity"
            ),
            "r0_complete": True,
            "r1_physics_started": False,
            "training_started": False,
        },
    }
    return report, gauge, scenes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bucket-obj", type=Path, required=True)
    parser.add_argument("--inventory-contract", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report, gauge, scenes = build(
        args.config, args.bucket_obj, args.inventory_contract, args.asset_dir)
    write_json(args.output_dir / "bucket-gauge-report.json", gauge)
    write_json(args.output_dir / "scene-manifest.json", scenes)
    write_json(args.output_dir / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
