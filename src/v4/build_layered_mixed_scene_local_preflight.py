#!/usr/bin/env python3
"""Compile a reduced three-layer mixed rock/MPM geometry preflight.

This module intentionally performs no physics.  It derives each rock height
from its transformed mesh and requested embed ratio, carves only MPM particle
centres inside closed convex rocks, and labels when each retained particle is
activated by the staged construction sequence.
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

from build_fullscale_graded_gravel_slope import _inside_convex, sat_axis_gap, sha256
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz


CONFIG_SCHEMA = "excavation_layered_mixed_scene_local_preflight_config_v1"
REPORT_SCHEMA = "excavation_layered_mixed_scene_local_preflight_report_v1"
SIZE_CLASSES = {
    "small_0p16_0p24": (0.16, 0.24),
    "medium_0p24_0p32": (0.24, 0.32),
    "large_0p32_0p40": (0.32, 0.40),
}
ACTIVATION_NAMES = ["foundation", "lower_fill", "middle_fill", "top_fill"]


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return stream.getvalue()


def _surface(config: dict, surface_name: str, xy: np.ndarray) -> np.ndarray:
    patch = config["local_patch"]
    item = patch["surfaces"][surface_name]
    center = np.asarray(patch["surface_center_xy_m"], dtype=float)
    period = np.asarray(patch["surface_wave_period_xy_m"], dtype=float)
    offset = np.asarray(xy, dtype=float) - center
    wave = float(patch["surface_wave_amplitude_m"]) * (
        np.sin(2.0 * math.pi * offset[..., 0] / period[0])
        * np.cos(2.0 * math.pi * offset[..., 1] / period[1]))
    return (float(item["base_height_m"])
            + offset @ np.asarray(item["slope_xy"], dtype=float) + wave)


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "reduced_three_layer_geometry_preflight_only"):
        raise ValueError("layered_local_config_schema_or_status")
    bank = config.get("frozen_bank", {})
    expected_bank = {
        "path": "/home/qunwang/codex-work/compact-bank-2026-09-11/manual-20260911-090834/initial-bank.npz",
        "sha256": "bc3fb89dc46f2787f1c65b8699fd5eb79d497a7bf02998c4da5b9f744622e89b",
        "q_array_sha256": "5c4cd8f0a12885f0d78bb4157dc1706d7e363c2c7564c1415093b89cfba6dba8",
        "q_count": 390400, "q_dtype": "float32", "particle_spacing_m": 0.025,
        "particle_density_kg_m3": 1670.0, "material_id": 0,
    }
    if bank != expected_bank:
        raise ValueError("layered_local_bank_contract_drift")
    if not Path(bank["path"]).is_file() or sha256(bank["path"]) != bank["sha256"]:
        raise ValueError("layered_local_bank_binding_mismatch")
    if config.get("asset_root") != "assets/v4/mixed-material-r0":
        raise ValueError("layered_local_asset_root_drift")
    expected_assets = {
        "angular_tetra": "6934783ff5fc95e68cfddfb30f5630ff19be7633c6679d5f1d8f2a3ab0d2afb7",
        "angular_bipyramid": "e9e432a263512a21232a6b590a51ca923a7b2432c8458312b5fbd76a3d2d55cd",
        "angular_octa": "6279098235a6dd22ef97ffd2c2d2d2ff028580e26c3110040cb08264faea1fd8",
        "angular_block": "963182f35d733b763f9d829f2a5cae71ba03c8123a889a75bc19997cf6142b43",
    }
    if config.get("assets") != expected_assets:
        raise ValueError("layered_local_asset_contract_drift")
    for name, digest in expected_assets.items():
        if sha256(repository / config["asset_root"] / f"{name}.obj") != digest:
            raise ValueError("layered_local_asset_binding_mismatch:" + name)
    if float(config.get("density_kg_m3", -1)) != 2600.0:
        raise ValueError("layered_local_density_drift")
    patch = config.get("local_patch", {})
    bounds = np.asarray(patch.get("xy_bounds_m"), dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0])):
        raise ValueError("layered_local_patch_bounds")
    surface_names = list(patch.get("surfaces", {}))
    if surface_names != ACTIVATION_NAMES:
        raise ValueError("layered_local_surface_order")
    sample_xy = np.asarray([
        bounds[0], bounds[1], [bounds[0, 0], bounds[1, 1]],
        [bounds[1, 0], bounds[0, 1]], np.mean(bounds, axis=0)])
    heights = np.stack([_surface(config, name, sample_xy)
                        for name in surface_names], axis=1)
    if not np.all(np.diff(heights, axis=1) > 0.10):
        raise ValueError("layered_local_surface_separation")
    layers = config.get("layers")
    if ([item.get("layer_id") for item in layers or []] !=
            ["lower_skeleton", "middle_small_dominant", "top_exposed_small"]
            or [item.get("fill_surface") for item in layers] !=
            ["lower_fill", "middle_fill", "top_fill"]):
        raise ValueError("layered_local_layer_contract")
    seen = set()
    for layer in layers:
        allowed = np.asarray(layer.get("embed_ratio_range"), dtype=float)
        if allowed.shape != (2,) or not 0.0 < allowed[0] <= allowed[1] < 1.0:
            raise ValueError("layered_local_embed_range")
        for rock in layer.get("rocks", []):
            rock_id = rock.get("rock_id")
            if not isinstance(rock_id, str) or not rock_id or rock_id in seen:
                raise ValueError("layered_local_rock_id")
            seen.add(rock_id)
            if rock.get("prototype_id") not in expected_assets:
                raise ValueError("layered_local_prototype")
            size_class = rock.get("size_class")
            diameter = float(rock.get("equivalent_diameter_m", math.nan))
            if size_class not in SIZE_CLASSES or not (
                    SIZE_CLASSES[size_class][0] <= diameter <= SIZE_CLASSES[size_class][1]):
                raise ValueError("layered_local_size_class")
            xy = np.asarray(rock.get("xy_m"), dtype=float)
            quaternion = np.asarray(rock.get("orientation_quaternion_wxyz"), dtype=float)
            ratio = float(rock.get("embed_ratio", math.nan))
            if (xy.shape != (2,) or quaternion.shape != (4,)
                    or not np.isfinite(xy).all() or not np.isfinite(quaternion).all()
                    or not math.isclose(float(np.linalg.norm(quaternion)), 1.0,
                                        abs_tol=1e-9)
                    or not allowed[0] <= ratio <= allowed[1]
                    or np.any(xy < bounds[0]) or np.any(xy > bounds[1])):
                raise ValueError("layered_local_rock_pose_or_embed")
    construction = config.get("construction", {})
    if (construction.get("carve_policy") !=
            "remove_particle_center_inside_closed_convex_rock_only"
            or construction.get("embed_measure") !=
            "true_transformed_mesh_vertical_span_at_stage_fill_surface"
            or float(construction.get("minimum_rock_sat_gap_m", -1)) < 0.0
            or float(construction.get("embed_ratio_tolerance", -1)) <= 0.0):
        raise ValueError("layered_local_construction_contract")
    if config.get("scope") != {
        "geometry_only": True, "reduced_local_patch": True,
        "staged_particle_activation_labels": True, "physics_step": False,
        "mpm_or_gpu_run": False, "wall_free_hold_executed": False,
        "stable_scene_claimed": False, "formal_scene_qualified": False,
        "training": False,
    }:
        raise ValueError("layered_local_scope_expanded")
    return config


def _load_bank(config: dict) -> np.ndarray:
    bank = config["frozen_bank"]
    with np.load(bank["path"], allow_pickle=False) as archive:
        q = archive["q"]
    if (q.shape != (bank["q_count"], 3) or str(q.dtype) != bank["q_dtype"]
            or hashlib.sha256(q.tobytes(order="C")).hexdigest() !=
            bank["q_array_sha256"]):
        raise ValueError("layered_local_bank_q_contract")
    return q


def _derive_rocks(config: dict) -> list[dict]:
    prototypes = normalized_prototypes(sorted(config["assets"]))
    rocks = []
    for layer_index, layer in enumerate(config["layers"]):
        for source in layer["rocks"]:
            prototype = prototypes[source["prototype_id"]]
            diameter = float(source["equivalent_diameter_m"])
            rotation = quaternion_matrix_wxyz(source["orientation_quaternion_wxyz"])
            local_world_axes = prototype["vertices"] * diameter @ rotation.T
            local_min_z = float(local_world_axes[:, 2].min())
            local_max_z = float(local_world_axes[:, 2].max())
            vertical_span = local_max_z - local_min_z
            xy = np.asarray(source["xy_m"], dtype=float)
            fill_height = float(_surface(config, layer["fill_surface"], xy))
            requested_ratio = float(source["embed_ratio"])
            center_z = fill_height - local_min_z - requested_ratio * vertical_span
            position = np.asarray([xy[0], xy[1], center_z])
            vertices = local_world_axes + position
            achieved_ratio = (fill_height - float(vertices[:, 2].min())) / vertical_span
            volume = float(prototype["unit_volume_m3"]) * diameter ** 3
            density = float(config["density_kg_m3"])
            rocks.append({
                "rock_id": source["rock_id"], "layer_id": layer["layer_id"],
                "layer_index": layer_index, "fill_surface": layer["fill_surface"],
                "prototype_id": source["prototype_id"],
                "size_class": source["size_class"], "equivalent_diameter_m": diameter,
                "density_kg_m3": density, "volume_m3": volume,
                "mass_kg": density * volume,
                "inertia_about_com_body_kg_m2":
                    (density * prototype["unit_inertia_m5"] * diameter ** 5).tolist(),
                "position_world_m": position.tolist(),
                "orientation_quaternion_wxyz": source["orientation_quaternion_wxyz"],
                "fill_surface_height_m": fill_height,
                "true_vertical_span_m": vertical_span,
                "requested_embed_ratio": requested_ratio,
                "achieved_embed_ratio": achieved_ratio,
                "vertices": vertices, "faces": prototype["faces"],
                "rotation": rotation,
            })
    return rocks


def _carve(q: np.ndarray, rocks: list[dict], prototypes: dict) -> tuple[np.ndarray, dict]:
    removed = np.zeros(len(q), dtype=bool)
    removed_by_id = {}
    for rock in rocks:
        vertices = rock["vertices"]
        candidates = np.flatnonzero(
            (~removed) & np.all(q >= vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(q <= vertices.max(axis=0) + 1e-10, axis=1))
        count = 0
        if len(candidates):
            local = ((q[candidates].astype(float)
                      - np.asarray(rock["position_world_m"])) @ rock["rotation"]
                     / rock["equivalent_diameter_m"])
            prototype = prototypes[rock["prototype_id"]]
            inside = _inside_convex(local, prototype["vertices"], prototype["faces"])
            selected = candidates[inside]
            removed[selected] = True
            count = int(len(selected))
        removed_by_id[rock["rock_id"]] = count
    retained_inside = 0
    for rock in rocks:
        vertices = rock["vertices"]
        candidates = np.flatnonzero(
            (~removed) & np.all(q >= vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(q <= vertices.max(axis=0) + 1e-10, axis=1))
        if len(candidates):
            local = ((q[candidates].astype(float)
                      - np.asarray(rock["position_world_m"])) @ rock["rotation"]
                     / rock["equivalent_diameter_m"])
            prototype = prototypes[rock["prototype_id"]]
            retained_inside += int(np.count_nonzero(_inside_convex(
                local, prototype["vertices"], prototype["faces"])))
    return removed, {"removed_particle_count_by_rock_id": removed_by_id,
                     "retained_particle_center_inside_rock_count": retained_inside}


def build(config_path: str | Path, repository_root: str | Path,
          output_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("layered_local_output_dir_exists")
    config = load_and_validate_config(config_path, repository)
    q_all = _load_bank(config)
    bounds = np.asarray(config["local_patch"]["xy_bounds_m"], dtype=float)
    xy_mask = np.all(q_all[:, :2] >= bounds[0], axis=1) & np.all(
        q_all[:, :2] <= bounds[1], axis=1)
    top_height = _surface(config, "top_fill", q_all[:, :2])
    local_indices = np.flatnonzero(xy_mask & (q_all[:, 2] <= top_height + 1e-9))
    q = q_all[local_indices]
    rocks = _derive_rocks(config)
    prototypes = normalized_prototypes(sorted(config["assets"]))
    pair_gaps = []
    intersecting_pairs = []
    for first, second in itertools.combinations(rocks, 2):
        intersects, gap = sat_axis_gap(first["vertices"], first["faces"],
                                       second["vertices"], second["faces"])
        pair_gaps.append(gap)
        if intersects:
            intersecting_pairs.append([first["rock_id"], second["rock_id"]])
    removed, carve = _carve(q, rocks, prototypes)
    retained = ~removed
    surfaces = np.stack([_surface(config, name, q[:, :2])
                         for name in ACTIVATION_NAMES], axis=1)
    activation = np.full(len(q), -1, dtype=np.int8)
    for index in range(len(ACTIVATION_NAMES)):
        activation[(activation < 0) & (q[:, 2] <= surfaces[:, index] + 1e-9)] = index
    if np.any(activation < 0):
        raise ValueError("layered_local_unassigned_particle")
    tolerance = float(config["construction"]["embed_ratio_tolerance"])
    embed_errors = [abs(rock["achieved_embed_ratio"] - rock["requested_embed_ratio"])
                    for rock in rocks]
    layer_counts = Counter(rock["layer_id"] for rock in rocks)
    size_counts = Counter(rock["size_class"] for rock in rocks)
    activation_counts = {name: int(np.count_nonzero(
        retained & (activation == index))) for index, name in enumerate(ACTIVATION_NAMES)}
    patch_margin = min(
        min(float(rock["vertices"][:, 0].min() - bounds[0, 0]),
            float(bounds[1, 0] - rock["vertices"][:, 0].max()),
            float(rock["vertices"][:, 1].min() - bounds[0, 1]),
            float(bounds[1, 1] - rock["vertices"][:, 1].max()))
        for rock in rocks)
    checks = {
        "rock_count_is_22": len(rocks) == 22,
        "all_three_rock_layers_present": set(layer_counts) == {
            "lower_skeleton", "middle_small_dominant", "top_exposed_small"},
        "small_rocks_are_count_majority": size_counts["small_0p16_0p24"] >
            size_counts["medium_0p24_0p32"] + size_counts["large_0p32_0p40"],
        "medium_large_only_in_lower_skeleton": all(
            rock["layer_id"] == "lower_skeleton" for rock in rocks
            if rock["size_class"] != "small_0p16_0p24"),
        "true_mesh_embed_ratios_exact": max(embed_errors) <= tolerance,
        "rock_rock_sat_no_intersection": not intersecting_pairs,
        "rocks_within_local_patch_xy": patch_margin >= -1e-12,
        "all_activation_stages_nonempty": all(value > 0 for value in activation_counts.values()),
        "particle_partition_closed": int(retained.sum()) + int(removed.sum()) == len(q),
        "carve_removed_particles": bool(removed.any()),
        "retained_particle_center_inside_rock_zero":
            carve["retained_particle_center_inside_rock_count"] == 0,
        "no_physics_step": True,
    }
    if not all(checks.values()):
        raise ValueError("layered_local_checks_failed:" + ",".join(
            name for name, passed in checks.items() if not passed)
            + (":" + json.dumps(intersecting_pairs) if intersecting_pairs else ""))
    public_rocks = []
    for rock in rocks:
        public_rocks.append({key: value for key, value in rock.items()
                             if key not in {"vertices", "faces", "rotation"}})
    manifest = {
        "schema": "excavation_layered_mixed_scene_local_manifest_v1",
        "status": "reduced_three_layer_geometry_candidate",
        "construction_sequence": config["construction"]["sequence"],
        "rock_count": len(rocks), "rocks": public_rocks,
        "activation_stage_ids": {name: index for index, name in enumerate(ACTIVATION_NAMES)},
        "checks": checks, "physics_step_executed": False,
        "stable_scene_claimed": False,
    }
    particle_volume = float(config["frozen_bank"]["particle_spacing_m"]) ** 3
    metrics = {
        "schema": "excavation_layered_mixed_scene_local_metrics_v1",
        "local_source_particle_count": len(q),
        "retained_particle_count": int(retained.sum()),
        "removed_particle_count": int(removed.sum()),
        "activation_particle_count": activation_counts,
        "rock_count_by_layer": dict(sorted(layer_counts.items())),
        "rock_count_by_size_class": dict(sorted(size_counts.items())),
        "minimum_rock_rock_sat_gap_m": min(pair_gaps),
        "minimum_rock_patch_xy_margin_m": patch_margin,
        "maximum_embed_ratio_error": max(embed_errors),
        "retained_mpm_equivalent_volume_m3": int(retained.sum()) * particle_volume,
        "rock_analytic_volume_m3": sum(rock["volume_m3"] for rock in rocks),
        "geometry_only": True,
        **carve,
        "checks": checks,
    }
    arrays = {
        "q": q[retained],
        "material_id": np.full(int(retained.sum()),
                               config["frozen_bank"]["material_id"], dtype=np.uint8),
        "activation_stage": activation[retained],
        "original_bank_index": local_indices[retained].astype(np.int64),
        "removed_original_bank_index": local_indices[removed].astype(np.int64),
    }
    payloads = {
        "layered-local-bank.npz": _npz_bytes(arrays),
        "scene-manifest.json": _json_bytes(manifest),
        "metrics.json": _json_bytes(metrics),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "reduced_three_layer_geometry_preflight_passed",
        "config_sha256": sha256(config_path),
        "outputs": {name: hashlib.sha256(payload).hexdigest()
                    for name, payload in payloads.items()},
        "checks": checks, "physics_step_executed": False,
        "mpm_or_gpu_run_executed": False, "wall_free_hold_executed": False,
        "stable_scene_claimed": False, "formal_scene_qualified": False,
        "next_gate": "reduced_mixed_dynamics_per_layer_settle_then_wall_free_nominal_hold",
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
