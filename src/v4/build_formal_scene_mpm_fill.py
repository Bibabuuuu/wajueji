#!/usr/bin/env python3
"""Carve the frozen MPM bank around a validated D-063 rigid-rock manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from build_formal_scene_model_adapter import (
    load_and_validate_config as load_adapter_config,
    validate_scene_manifest,
)
from build_fullscale_graded_gravel_slope import (
    _deterministic_npz_bytes,
    _inside_convex,
    sha256,
)
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz


CONFIG_SCHEMA = "excavation_formal_scene_mpm_fill_config_v1"
REPORT_SCHEMA = "excavation_formal_scene_mpm_fill_report_v1"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> tuple[dict, Path]:
    repository = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "visual_stable_scene_carve_only"):
        raise ValueError("mpm_fill_config_schema_or_status")
    binding = config.get("frozen_adapter_config")
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}):
        raise ValueError("mpm_fill_adapter_binding")
    adapter_path = repository / binding["path"]
    if not adapter_path.is_file() or sha256(adapter_path) != binding["sha256"]:
        raise ValueError("mpm_fill_adapter_binding_mismatch")
    load_adapter_config(adapter_path, repository)
    bank = config.get("frozen_bank")
    if bank != {
        "path": "/home/qunwang/codex-work/compact-bank-2026-09-11/manual-20260911-090834/initial-bank.npz",
        "sha256": "bc3fb89dc46f2787f1c65b8699fd5eb79d497a7bf02998c4da5b9f744622e89b",
        "q_array_sha256": "5c4cd8f0a12885f0d78bb4157dc1706d7e363c2c7564c1415093b89cfba6dba8",
        "q_count": 390400,
        "q_dtype": "float32",
        "particle_spacing_m": 0.025,
        "particle_density_kg_m3": 1670.0,
        "material_id": 0,
    }:
        raise ValueError("mpm_fill_bank_contract_drift")
    bank_path = Path(bank["path"])
    if not bank_path.is_file() or sha256(bank_path) != bank["sha256"]:
        raise ValueError("mpm_fill_bank_binding_mismatch")
    if config.get("carve_policy") != (
            "remove_particle_center_inside_closed_convex_rock_only"):
        raise ValueError("mpm_fill_carve_policy_drift")
    if config.get("reference_only") != {
        "nominal_total_equivalent_volume_m3": 6.1,
        "advisory_total_equivalent_volume_range_m3": [5.2, 7.0],
        "volume_or_fraction_error_is_not_a_standalone_failure": True,
    }:
        raise ValueError("mpm_fill_reference_policy_drift")
    if config.get("scope") != {
        "deterministic_particle_partition": True,
        "retained_particle_rock_center_overlap_check": True,
        "physics_step": False,
        "mpm_or_gpu_run": False,
        "settling": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
        "training": False,
    }:
        raise ValueError("mpm_fill_scope_expanded")
    return config, adapter_path


def _load_bank(config: dict) -> np.ndarray:
    bank = config["frozen_bank"]
    with np.load(bank["path"], allow_pickle=False) as archive:
        if "q" not in archive.files:
            raise ValueError("mpm_fill_bank_q_missing")
        q = archive["q"]
    if q.shape != (bank["q_count"], 3) or str(q.dtype) != bank["q_dtype"]:
        raise ValueError("mpm_fill_bank_q_shape_or_dtype")
    if hashlib.sha256(q.tobytes(order="C")).hexdigest() != bank["q_array_sha256"]:
        raise ValueError("mpm_fill_bank_q_hash_mismatch")
    return q


def _carve(q: np.ndarray, rocks: list[dict]) -> tuple[np.ndarray, dict]:
    prototypes = normalized_prototypes(sorted({rock["prototype_id"] for rock in rocks}))
    removed = np.zeros(len(q), dtype=bool)
    removed_by_rock = {}
    transforms = []
    for rock in rocks:
        prototype = prototypes[rock["prototype_id"]]
        rotation = quaternion_matrix_wxyz(
            np.asarray(rock["orientation_quaternion_wxyz"], dtype=float))
        diameter = float(rock["equivalent_diameter_m"])
        position = np.asarray(rock["position_world_m"], dtype=float)
        vertices = prototype["vertices"] * diameter @ rotation.T + position
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        candidates = np.flatnonzero(
            (~removed) & np.all(q >= low - 1e-10, axis=1)
            & np.all(q <= high + 1e-10, axis=1))
        inside_indices = np.empty(0, dtype=np.int64)
        if len(candidates):
            local = ((q[candidates].astype(float) - position)
                     @ rotation / diameter)
            inside = _inside_convex(
                local, prototype["vertices"], prototype["faces"])
            inside_indices = candidates[inside]
            removed[inside_indices] = True
        removed_by_rock[rock["source_rock_id"]] = int(len(inside_indices))
        transforms.append((rock, prototype, rotation, position, low, high))
    retained = ~removed
    overlap_count = 0
    for rock, prototype, rotation, position, low, high in transforms:
        candidates = np.flatnonzero(
            retained & np.all(q >= low - 1e-10, axis=1)
            & np.all(q <= high + 1e-10, axis=1))
        if len(candidates):
            local = ((q[candidates].astype(float) - position)
                     @ rotation / float(rock["equivalent_diameter_m"]))
            overlap_count += int(np.count_nonzero(_inside_convex(
                local, prototype["vertices"], prototype["faces"])))
    return removed, {
        "removed_particle_count_by_source_rock_id": removed_by_rock,
        "retained_particle_center_inside_rock_count": overlap_count,
    }


def build(config_path: str | Path, scene_manifest_path: str | Path,
          repository_root: str | Path, output_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    scene_path = Path(scene_manifest_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("mpm_fill_output_dir_exists")
    config, adapter_path = load_and_validate_config(config_path, repository)
    adapter, _ = load_adapter_config(adapter_path, repository)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    rocks = validate_scene_manifest(adapter, scene, repository)
    q = _load_bank(config)
    removed, carve_diagnostics = _carve(q, rocks)
    retained = ~removed
    bank = config["frozen_bank"]
    particle_volume = float(bank["particle_spacing_m"]) ** 3
    particle_mass = float(bank["particle_density_kg_m3"]) * particle_volume
    rock_volume = sum(float(rock["volume_m3"]) for rock in rocks)
    retained_volume = int(retained.sum()) * particle_volume
    combined_volume = rock_volume + retained_volume
    reference = config["reference_only"]
    advisory_low, advisory_high = reference[
        "advisory_total_equivalent_volume_range_m3"]
    checks = {
        "particle_partition": int(retained.sum()) + int(removed.sum()) == len(q),
        "retained_particles_nonempty": bool(retained.any()),
        "removed_particles_nonempty": bool(removed.any()),
        "retained_particle_center_inside_rock_zero": carve_diagnostics[
            "retained_particle_center_inside_rock_count"] == 0,
        "source_rock_count_preserved": len(rocks) == scene["rock_count"],
        "no_physics_step": True,
    }
    if not all(checks.values()):
        raise ValueError("mpm_fill_checks_failed:" + ",".join(
            name for name, passed in checks.items() if not passed))
    arrays = {
        "q": q[retained],
        "material_id": np.full(int(retained.sum()), bank["material_id"], dtype=np.uint8),
        "original_index": np.flatnonzero(retained).astype(np.int64),
        "removed_index": np.flatnonzero(removed).astype(np.int64),
    }
    npz_bytes = _deterministic_npz_bytes(arrays)
    metrics = {
        "schema": "excavation_formal_scene_mpm_fill_metrics_v1",
        "scene_manifest_sha256": sha256(scene_path),
        "rock_count": len(rocks),
        "original_particle_count": len(q),
        "retained_particle_count": int(retained.sum()),
        "removed_particle_count": int(removed.sum()),
        "particle_equivalent_volume_m3": particle_volume,
        "particle_mass_kg": particle_mass,
        "retained_mpm_equivalent_volume_m3": retained_volume,
        "retained_mpm_mass_kg": int(retained.sum()) * particle_mass,
        "rock_analytic_volume_m3": rock_volume,
        "combined_equivalent_solid_volume_m3": combined_volume,
        "nominal_reference_volume_m3": reference[
            "nominal_total_equivalent_volume_m3"],
        "relative_nominal_reference_difference": abs(
            combined_volume - reference["nominal_total_equivalent_volume_m3"]
        ) / reference["nominal_total_equivalent_volume_m3"],
        "within_advisory_total_equivalent_volume_range": (
            advisory_low <= combined_volume <= advisory_high),
        **carve_diagnostics,
        "checks": checks,
        "volume_or_fraction_error_is_not_a_standalone_failure": True,
    }
    metrics_bytes = _json_bytes(metrics)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "formal_scene_mpm_fill_geometry_candidate",
        "config_sha256": sha256(config_path),
        "adapter_config_sha256": sha256(adapter_path),
        "scene_manifest_sha256": sha256(scene_path),
        "outputs": {
            "layered_bank_npz_sha256": hashlib.sha256(npz_bytes).hexdigest(),
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        },
        "checks": checks,
        "physics_step_executed": False,
        "mpm_or_gpu_run_executed": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    }
    report_bytes = _json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (temporary / "layered-bank.npz").write_bytes(npz_bytes)
        (temporary / "metrics.json").write_bytes(metrics_bytes)
        (temporary / "report.json").write_bytes(report_bytes)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = build(
        args.config, args.scene_manifest, args.repository_root, args.output_dir)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
