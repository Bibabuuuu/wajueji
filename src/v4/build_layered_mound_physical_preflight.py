#!/usr/bin/env python3
"""Build a non-intersecting 65-rock geometry from the approved visual intent.

The approved preview is treated as a style/layout target, not as trusted
collision geometry.  Lower layers may use the wider mound cross-section at
their placement height.  No physics step is executed.
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
import shutil
import tempfile
import zipfile

import numpy as np

from audit_layered_mound_preview_geometry import _carve, _physical_rocks
from build_fullscale_graded_gravel_slope import _sat_precomputed, _unique_edges, sha256
from preview_layered_mound_existing_viewer import _load_config, _mound_mask, build_rock_records


CONFIG_SCHEMA = "excavation_layered_mound_physical_preflight_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_physical_preflight_report_v1"
LAYERS = ("lower", "lower_middle", "middle", "upper", "top")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return stream.getvalue()


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict, Path]:
    repository = Path(repository_root).resolve()
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "approved_visual_intent_to_nonintersecting_geometry_only"):
        raise ValueError("layered_mound_physical_config_schema_or_status")
    source = value.get("visual_config", {})
    visual_path = repository / source.get("path", "")
    if (not visual_path.is_file() or sha256(visual_path) != source.get("sha256")
            or source.get("required_review_status") != "user_approved"):
        raise ValueError("layered_mound_physical_visual_binding")
    visual = _load_config(visual_path)
    if visual.get("visual_review", {}).get("status") != "user_approved":
        raise ValueError("layered_mound_physical_visual_review")
    packing = value.get("packing", {})
    expected_keys = {
        "method", "minimum_sat_gap_m", "boundary_margin_m",
        "local_candidate_count", "total_candidate_count",
        "accepted_candidate_pool_size", "candidate_sigma_xy_m",
        "nearest_neighbor_score_weight", "seed_base",
        "maximum_target_xy_shift_m", "maximum_mean_target_xy_shift_m"}
    if (set(packing) != expected_keys
            or packing.get("method") !=
            "deterministic_stage_footprint_cluster_preserving_rejection"
            or not 0.0 < float(packing["minimum_sat_gap_m"]) <= 0.005
            or not 0.0 < float(packing["boundary_margin_m"]) <= 0.01
            or not 0 < int(packing["local_candidate_count"]) <
            int(packing["total_candidate_count"])
            or int(packing["accepted_candidate_pool_size"]) <= 0
            or np.asarray(packing["candidate_sigma_xy_m"]).shape != (2,)):
        raise ValueError("layered_mound_physical_packing_contract")
    if value.get("construction") != {
            "carve_policy": "remove_particle_center_inside_closed_convex_rock_only",
            "activation_policy": "six_height_stages_from_approved_visual_config",
            "rock_density_kg_m3": 2600.0}:
        raise ValueError("layered_mound_physical_construction_contract")
    if value.get("scope") != {
            "geometry_only": True, "physics_step": False,
            "mpm_or_gpu_run": False, "visual_style_preserved": True,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("layered_mound_physical_scope")
    return value, visual, visual_path


def _stage_footprint(visual: dict, height: float) -> np.ndarray:
    envelope = visual["mound_envelope"]
    outer = np.asarray(envelope["outer_xy_bounds_m"], dtype=float)
    plateau = np.asarray(envelope["plateau_xy_bounds_m"], dtype=float)
    base = float(envelope["base_height_m"])
    top = float(envelope["plateau_height_m"])
    cosine = np.clip(2.0 * (height - base) / (top - base) - 1.0, -1.0, 1.0)
    distance = math.acos(float(cosine)) / math.pi
    return np.vstack((
        plateau[0] - distance * (plateau[0] - outer[0]),
        plateau[1] + distance * (outer[1] - plateau[1]),
    ))


def _prepare_rocks(visual: dict) -> tuple[list[dict], dict]:
    physical, prototypes = _physical_rocks(visual)
    visual_records = {item["id"]: item for item in build_rock_records(visual)}
    work_zone = np.asarray(visual["work_zone_xy_m"], dtype=float)
    stage_specs = {item["rock_layer"]: item for item in visual["construction_stages"]
                   if item["rock_layer"] is not None}
    for rock in physical:
        record = visual_records[rock["rock_id"]]
        local_vertices = rock["vertices"] - np.asarray(rock["position_world_m"])
        face_normals = []
        for face in rock["faces"]:
            a, b, c = local_vertices[face]
            normal = np.cross(b - a, c - a)
            face_normals.append(normal / np.linalg.norm(normal))
        height = float(stage_specs[rock["layer_id"]]["placement_surface_height_m"])
        bounds = _stage_footprint(visual, height)
        original = np.asarray(rock["position_world_m"], dtype=float)
        uv = (original[:2] - work_zone[0]) / (work_zone[1] - work_zone[0])
        rock.update({
            "local_vertices": local_vertices,
            "face_normals": np.asarray(face_normals),
            "edge_directions": np.asarray(_unique_edges(local_vertices, rock["faces"])),
            "original_position": original,
            "target_xy": bounds[0] + uv * (bounds[1] - bounds[0]),
            "stage_footprint_xy": bounds,
            "horizontal_aabb_area": float(np.prod(np.ptp(local_vertices[:, :2], axis=0))),
            "color_rgb": record["color_rgb"],
        })
    return physical, prototypes


def _sat(first_vertices: np.ndarray, first: dict,
         second_vertices: np.ndarray, second: dict) -> tuple[bool, float]:
    return _sat_precomputed(
        first_vertices, first["face_normals"], first["edge_directions"],
        second_vertices, second["face_normals"], second["edge_directions"])


def _pack(rocks: list[dict], config: dict) -> list[dict]:
    packing = config["packing"]
    margin = float(packing["boundary_margin_m"])
    required_gap = float(packing["minimum_sat_gap_m"])
    sigma = np.asarray(packing["candidate_sigma_xy_m"], dtype=float)
    placed: list[dict] = []
    for layer_index, layer in enumerate(LAYERS):
        group = [rock for rock in rocks if rock["layer_id"] == layer]
        ordered = sorted(group, key=lambda rock: (-rock["horizontal_aabb_area"], rock["rock_id"]))
        for rock_index, rock in enumerate(ordered):
            rng = np.random.default_rng(int(packing["seed_base"]) + layer_index * 100 + rock_index)
            local = rock["local_vertices"]
            bounds = rock["stage_footprint_xy"]
            low = bounds[0] - local[:, :2].min(axis=0) + margin
            high = bounds[1] - local[:, :2].max(axis=0) - margin
            if np.any(high <= low):
                raise ValueError("layered_mound_physical_rock_larger_than_footprint:" + rock["rock_id"])
            target = np.clip(rock["target_xy"], low, high)
            candidates = []
            for attempt in range(int(packing["total_candidate_count"])):
                if attempt == 0:
                    xy = target.copy()
                elif attempt < int(packing["local_candidate_count"]):
                    xy = target + rng.normal(0.0, sigma)
                else:
                    xy = rng.uniform(low, high)
                xy = np.clip(xy, low, high)
                position = rock["original_position"].copy()
                position[:2] = xy
                vertices = local + position
                nearest_gap = 10.0
                valid = True
                for other in placed:
                    other_vertices = other["vertices"]
                    if (np.any(vertices.max(axis=0) + required_gap < other_vertices.min(axis=0))
                            or np.any(other_vertices.max(axis=0) + required_gap < vertices.min(axis=0))):
                        continue
                    intersects, gap = _sat(vertices, rock, other_vertices, other["source"])
                    nearest_gap = min(nearest_gap, gap)
                    if intersects or gap < required_gap:
                        valid = False
                        break
                if valid:
                    score = (float(np.linalg.norm(xy - target))
                             + float(packing["nearest_neighbor_score_weight"]) * nearest_gap)
                    candidates.append((score, position, vertices))
                    if len(candidates) >= int(packing["accepted_candidate_pool_size"]):
                        break
            if not candidates:
                raise ValueError("layered_mound_physical_no_candidate:" + rock["rock_id"])
            _, position, vertices = min(candidates, key=lambda item: item[0])
            placed.append({"source": rock, "position": position, "vertices": vertices})
    return placed


def build(config_path: str | Path, repository_root: str | Path,
          output_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("layered_mound_physical_output_exists")
    config, visual, visual_path = load_config(config_path, repository)
    rocks, prototypes = _prepare_rocks(visual)
    placed = _pack(rocks, config)
    by_id = {item["source"]["rock_id"]: item for item in placed}

    intersections = []
    gaps = []
    for first, second in itertools.combinations(placed, 2):
        intersects, gap = _sat(first["vertices"], first["source"],
                               second["vertices"], second["source"])
        gaps.append(gap)
        if intersects or gap < float(config["packing"]["minimum_sat_gap_m"]):
            intersections.append([first["source"]["rock_id"],
                                  second["source"]["rock_id"], gap])

    shifts = [float(np.linalg.norm(item["position"][:2] - item["source"]["target_xy"]))
              for item in placed]
    visual_shifts = [float(np.linalg.norm(
        item["position"][:2] - item["source"]["original_position"][:2])) for item in placed]
    outer = np.asarray(visual["mound_envelope"]["outer_xy_bounds_m"], dtype=float)
    outer_margin = min(min(
        float(item["vertices"][:, 0].min() - outer[0, 0]),
        float(outer[1, 0] - item["vertices"][:, 0].max()),
        float(item["vertices"][:, 1].min() - outer[0, 1]),
        float(outer[1, 1] - item["vertices"][:, 1].max())) for item in placed)

    with np.load(visual["source_bank"]["path"], allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    mound_mask = _mound_mask(source_q, visual)
    mound_indices = np.flatnonzero(mound_mask)
    mound_q = source_q[mound_mask]
    carve_rocks = []
    for original in rocks:
        item = by_id[original["rock_id"]]
        updated = dict(original)
        updated["position_world_m"] = item["position"].tolist()
        updated["vertices"] = item["vertices"]
        carve_rocks.append(updated)
    removed, carve = _carve(mound_q, carve_rocks, prototypes)
    retained = ~removed

    activation = np.full(len(mound_q), -1, dtype=np.int8)
    activation_counts = {}
    for index, stage in enumerate(visual["construction_stages"]):
        selected = ((activation < 0)
                    & (mound_q[:, 2] <= float(stage["soil_height_m"]) + 1e-9))
        activation[selected] = index
        activation_counts[stage["id"]] = int(np.count_nonzero(retained & selected))
    if np.any(activation < 0):
        raise ValueError("layered_mound_physical_unassigned_particle")

    layer_counts = Counter(rock["layer_id"] for rock in rocks)
    checks = {
        "approved_visual_config_hash_bound": sha256(visual_path) ==
            config["visual_config"]["sha256"],
        "rock_count_and_layers_preserved": len(placed) == 65 and
            [layer_counts[name] for name in LAYERS] == [9, 11, 13, 14, 18],
        "rock_identity_preserved": set(by_id) == {rock["rock_id"] for rock in rocks},
        "rock_rock_sat_gap_passed": not intersections and min(gaps) >=
            float(config["packing"]["minimum_sat_gap_m"]) - 1e-12,
        "rocks_within_mound_outer_xy": outer_margin >= -1e-12,
        "maximum_target_shift_bounded": max(shifts) <=
            float(config["packing"]["maximum_target_xy_shift_m"]),
        "mean_target_shift_bounded": float(np.mean(shifts)) <=
            float(config["packing"]["maximum_mean_target_xy_shift_m"]),
        "particle_partition_closed": int(retained.sum()) + int(removed.sum()) == len(mound_q),
        "carve_removed_particles": bool(removed.any()),
        "retained_particle_center_inside_rock_zero":
            carve["retained_particle_center_inside_rock_count"] == 0,
        "all_six_activation_stages_nonempty": all(value > 0 for value in activation_counts.values()),
        "physics_steps_zero": True,
    }
    if not all(checks.values()):
        raise ValueError("layered_mound_physical_checks_failed:" + ",".join(
            name for name, passed in checks.items() if not passed))

    public_rocks = []
    baseline_records = {item["id"]: item for item in build_rock_records(visual)}
    for rock in rocks:
        item = by_id[rock["rock_id"]]
        baseline = baseline_records[rock["rock_id"]]
        public_rocks.append({
            "rock_id": rock["rock_id"], "layer_id": rock["layer_id"],
            "prototype_id": rock["prototype_id"],
            "equivalent_diameter_m": rock["equivalent_diameter_m"],
            "position_world_m": item["position"].tolist(),
            "orientation_quaternion_wxyz": rock["orientation_quaternion_wxyz"],
            "color_rgb": baseline["color_rgb"],
            "volume_m3": rock["volume_m3"], "mass_kg": rock["mass_kg"],
            "inertia_about_com_body_kg_m2": rock["inertia_about_com_body_kg_m2"],
            "target_xy_m": rock["target_xy"].tolist(),
            "xy_shift_from_target_m": float(np.linalg.norm(
                item["position"][:2] - rock["target_xy"])),
        })
    manifest = {
        "schema": "excavation_layered_mound_physical_manifest_v1",
        "status": "nonintersecting_geometry_admitted_for_time_zero",
        "visual_config_sha256": sha256(visual_path),
        "rock_count": len(public_rocks), "rocks": public_rocks,
        "activation_stage_ids": {stage["id"]: index for index, stage in enumerate(
            visual["construction_stages"])},
        "checks": checks, "physics_steps_executed": 0,
        "stable_scene_claimed": False,
    }
    metrics = {
        "schema": "excavation_layered_mound_physical_metrics_v1",
        "source_particle_count": len(source_q),
        "mound_particle_count_before_carve": len(mound_q),
        "retained_particle_count": int(retained.sum()),
        "removed_particle_count": int(removed.sum()),
        "activation_particle_count": activation_counts,
        "rock_count_by_layer": dict(layer_counts),
        "minimum_rock_rock_sat_gap_m": min(gaps),
        "maximum_xy_shift_from_scaled_visual_target_m": max(shifts),
        "mean_xy_shift_from_scaled_visual_target_m": float(np.mean(shifts)),
        "maximum_xy_shift_from_literal_visual_position_m": max(visual_shifts),
        "mean_xy_shift_from_literal_visual_position_m": float(np.mean(visual_shifts)),
        "minimum_mound_outer_xy_margin_m": outer_margin,
        **carve, "checks": checks,
    }
    arrays = {
        "q": mound_q[retained],
        "material_id": np.zeros(int(retained.sum()), dtype=np.uint8),
        "activation_stage": activation[retained],
        "original_bank_index": mound_indices[retained].astype(np.int64),
        "removed_original_bank_index": mound_indices[removed].astype(np.int64),
    }
    payloads = {
        "layered-mound-bank.npz": _npz_bytes(arrays),
        "scene-manifest.json": _json_bytes(manifest),
        "metrics.json": _json_bytes(metrics),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "nonintersecting_geometry_preflight_passed",
        "config_sha256": sha256(config_path),
        "visual_config_sha256": sha256(visual_path),
        "outputs": {name: hashlib.sha256(payload).hexdigest()
                    for name, payload in payloads.items()},
        "checks": checks, "physics_steps_executed": 0,
        "mpm_or_gpu_run_executed": False, "stable_scene_claimed": False,
        "next_gate": "actual_65_rock_time_zero_and_single_press_smoke",
    }
    payloads["report.json"] = _json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.repository_root, args.output_dir),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
