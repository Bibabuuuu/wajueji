#!/usr/bin/env python3
"""Audit the user-approved five-layer mound preview against real convex meshes.

This is a zero-physics admission gate.  It deliberately refuses to publish a
physical manifest while any transformed rock meshes intersect.  Visual texture
and per-instance colour are excluded from collision semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from build_fullscale_graded_gravel_slope import _inside_convex, sat_axis_gap, sha256
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz
from preview_layered_mound_existing_viewer import _load_config, _mound_mask, build_rock_records


REPORT_SCHEMA = "excavation_layered_mound_preview_geometry_audit_v1"
EXPECTED_LAYERS = ("lower", "lower_middle", "middle", "upper", "top")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _physical_rocks(config: dict) -> tuple[list[dict], dict]:
    records = build_rock_records(config)
    prototypes = normalized_prototypes(sorted({item["prototype"] for item in records}))
    rocks = []
    for item in records:
        quaternion_xyzw = item["quaternion_xyzw"]
        quaternion_wxyz = [quaternion_xyzw[3], *quaternion_xyzw[:3]]
        rotation = quaternion_matrix_wxyz(quaternion_wxyz)
        prototype = prototypes[item["prototype"]]
        diameter = float(item["diameter_m"])
        local_vertices = prototype["vertices"] * diameter @ rotation.T
        position = np.asarray(item["position_m"], dtype=float)
        vertices = local_vertices + position
        density = 2600.0
        volume = float(prototype["unit_volume_m3"]) * diameter ** 3
        rocks.append({
            "rock_id": item["id"],
            "layer_id": item["layer"],
            "prototype_id": item["prototype"],
            "equivalent_diameter_m": diameter,
            "position_world_m": position.tolist(),
            "orientation_quaternion_wxyz": quaternion_wxyz,
            "volume_m3": volume,
            "mass_kg": density * volume,
            "inertia_about_com_body_kg_m2": (
                density * prototype["unit_inertia_m5"] * diameter ** 5).tolist(),
            "vertices": vertices,
            "faces": prototype["faces"],
            "rotation": rotation,
        })
    return rocks, prototypes


def _carve(q: np.ndarray, rocks: list[dict], prototypes: dict) -> tuple[np.ndarray, dict]:
    removed = np.zeros(len(q), dtype=bool)
    removed_by_rock = {}
    for rock in rocks:
        vertices = rock["vertices"]
        candidates = np.flatnonzero(
            np.all(q >= vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(q <= vertices.max(axis=0) + 1e-10, axis=1))
        selected = np.empty(0, dtype=np.int64)
        if len(candidates):
            local = ((q[candidates].astype(float)
                      - np.asarray(rock["position_world_m"])) @ rock["rotation"]
                     / rock["equivalent_diameter_m"])
            prototype = prototypes[rock["prototype_id"]]
            selected = candidates[_inside_convex(
                local, prototype["vertices"], prototype["faces"])]
            removed[selected] = True
        removed_by_rock[rock["rock_id"]] = int(len(selected))
    retained_inside = 0
    for rock in rocks:
        vertices = rock["vertices"]
        candidates = np.flatnonzero(
            (~removed)
            & np.all(q >= vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(q <= vertices.max(axis=0) + 1e-10, axis=1))
        if len(candidates):
            local = ((q[candidates].astype(float)
                      - np.asarray(rock["position_world_m"])) @ rock["rotation"]
                     / rock["equivalent_diameter_m"])
            prototype = prototypes[rock["prototype_id"]]
            retained_inside += int(np.count_nonzero(_inside_convex(
                local, prototype["vertices"], prototype["faces"])))
    return removed, {
        "removed_particle_count_by_rock_id": removed_by_rock,
        "retained_particle_center_inside_rock_count": retained_inside,
    }


def audit(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("layered_mound_geometry_audit_output_exists")
    config = _load_config(config_path)
    review = config.get("visual_review", {})
    if review != {
            "status": "user_approved", "date": "2026-09-21",
            "scope": "geometry_distribution_and_rendering_only",
            "physics_stability_included": False}:
        raise ValueError("layered_mound_visual_review_contract")
    rocks, prototypes = _physical_rocks(config)
    layer_counts = Counter(item["layer_id"] for item in rocks)

    intersections = []
    pair_gaps = []
    pair_counts = Counter()
    for first, second in itertools.combinations(rocks, 2):
        intersects, gap = sat_axis_gap(
            first["vertices"], first["faces"],
            second["vertices"], second["faces"])
        pair_gaps.append(gap)
        if intersects:
            key = "__".join(sorted((first["layer_id"], second["layer_id"])))
            pair_counts[key] += 1
            intersections.append({
                "first_rock_id": first["rock_id"],
                "second_rock_id": second["rock_id"],
                "first_layer_id": first["layer_id"],
                "second_layer_id": second["layer_id"],
                "sat_gap_m": gap,
                "penetration_lower_bound_m": -gap,
            })

    with np.load(config["source_bank"]["path"], allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    mound_q = source_q[_mound_mask(source_q, config)]
    removed, carve = _carve(mound_q, rocks, prototypes)
    retained = ~removed

    stage_counts = {}
    assigned = np.full(len(mound_q), -1, dtype=np.int8)
    stages = config["construction_stages"]
    for index, stage in enumerate(stages):
        selected = (assigned < 0) & (mound_q[:, 2] <= float(stage["soil_height_m"]) + 1e-9)
        assigned[selected] = index
        stage_counts[stage["id"]] = int(np.count_nonzero(retained & (assigned == index)))
    if np.any(assigned < 0):
        raise ValueError("layered_mound_unassigned_particle")

    outer = np.asarray(config["mound_envelope"]["outer_xy_bounds_m"], dtype=float)
    outer_margin = min(min(
        float(rock["vertices"][:, 0].min() - outer[0, 0]),
        float(outer[1, 0] - rock["vertices"][:, 0].max()),
        float(rock["vertices"][:, 1].min() - outer[0, 1]),
        float(outer[1, 1] - rock["vertices"][:, 1].max())) for rock in rocks)
    checks = {
        "approved_visual_baseline_bound": True,
        "rock_count_is_65": len(rocks) == 65,
        "five_layers_present": tuple(layer_counts) == EXPECTED_LAYERS,
        "counts_match_approved_baseline": [layer_counts[name] for name in EXPECTED_LAYERS]
            == [9, 11, 13, 14, 18],
        "rocks_within_mound_outer_xy": outer_margin >= -1e-12,
        "rock_rock_sat_no_intersection": not intersections,
        "particle_partition_closed": int(retained.sum()) + int(removed.sum()) == len(mound_q),
        "carve_removed_particles": bool(removed.any()),
        "retained_particle_center_inside_rock_zero":
            carve["retained_particle_center_inside_rock_count"] == 0,
        "all_particle_activation_stages_nonempty": all(value > 0 for value in stage_counts.values()),
        "physics_steps_zero": True,
    }
    admitted = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("approved_visual_geometry_admitted_for_time_zero"
                   if admitted else "approved_visual_geometry_rejected_for_physics"),
        "config_sha256": sha256(config_path),
        "source_bank_sha256": sha256(config["source_bank"]["path"]),
        "rock_count": len(rocks),
        "layer_counts": dict(layer_counts),
        "source_particle_count": len(source_q),
        "mound_particle_count_before_carve": len(mound_q),
        "retained_particle_count": int(retained.sum()),
        "removed_particle_count": int(removed.sum()),
        "activation_particle_count": stage_counts,
        "sat_pair_count": len(pair_gaps),
        "sat_intersection_count": len(intersections),
        "minimum_sat_gap_m": min(pair_gaps),
        "maximum_penetration_lower_bound_m": max(
            (item["penetration_lower_bound_m"] for item in intersections), default=0.0),
        "intersection_count_by_layer_pair": dict(sorted(pair_counts.items())),
        "intersections": intersections,
        "minimum_mound_outer_xy_margin_m": outer_margin,
        **carve,
        "checks": checks,
        "physics_steps_executed": 0,
        "physical_manifest_published": admitted,
        "physics_input_authorized": admitted,
        "next_gate": ("time_zero" if admitted else
                      "derive_nonintersecting_layout_preserving_approved_visual_intent"),
    }
    payload = _json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "audit.json").write_bytes(payload)
        (temporary / "audit.sha256").write_text(
            hashlib.sha256(payload).hexdigest() + "  audit.json\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config, args.output_dir), sort_keys=True,
                     allow_nan=False))


if __name__ == "__main__":
    main()
