#!/usr/bin/env python3
"""Build an atomic actual-N model and MPM-fill bundle from one stable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from build_formal_scene_model_adapter import build_model
from build_formal_scene_mpm_fill import build as build_mpm_fill


CONFIG_SCHEMA = "excavation_formal_scene_geometry_preparation_config_v1"
REPORT_SCHEMA = "excavation_formal_scene_geometry_bundle_report_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> tuple[dict, Path, Path]:
    repository = Path(repository_root).resolve()
    config = json.loads(Path(config_path).resolve().read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "validated_manifest_to_atomic_geometry_bundle_only"):
        raise ValueError("geometry_preparation_config_schema_or_status")
    resolved = []
    for name in ("frozen_model_adapter_config", "frozen_mpm_fill_config"):
        binding = config.get(name)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("geometry_preparation_binding:" + name)
        path = repository / binding["path"]
        if not path.is_file() or sha256(path) != binding["sha256"]:
            raise ValueError("geometry_preparation_binding_mismatch:" + name)
        resolved.append(path)
    if config.get("output_layout") != {
            "model": "actual-n-scene.xml",
            "model_manifest": "actual-n-model-manifest.json",
            "mpm_fill_directory": "mpm-fill",
            "bundle_report": "geometry-bundle-report.json",
    }:
        raise ValueError("geometry_preparation_output_layout_drift")
    if config.get("scope") != {
            "atomic_geometry_bundle": True,
            "physics_step": False,
            "mpm_or_gpu_run": False,
            "settling": False,
            "formal_scene_qualified": False,
            "physics_qualified": False,
            "training": False,
    }:
        raise ValueError("geometry_preparation_scope_expanded")
    return config, resolved[0], resolved[1]


def prepare(config_path: str | Path, scene_manifest_path: str | Path,
            repository_root: str | Path, output_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    scene_path = Path(scene_manifest_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("geometry_preparation_output_dir_exists")
    config, adapter_config, fill_config = load_and_validate_config(
        config_path, repository)
    layout = config["output_layout"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        model_path = temporary / layout["model"]
        model_manifest = build_model(
            adapter_config, scene_path, repository, model_path)
        model_manifest_path = temporary / layout["model_manifest"]
        model_manifest_path.write_bytes(_json_bytes(model_manifest))
        fill_dir = temporary / layout["mpm_fill_directory"]
        fill_report = build_mpm_fill(
            fill_config, scene_path, repository, fill_dir)
        fill_metrics_path = fill_dir / "metrics.json"
        fill_metrics = json.loads(fill_metrics_path.read_text(encoding="utf-8"))
        scene_hash = sha256(scene_path)
        if (model_manifest["source_scene_manifest_sha256"] != scene_hash
                or fill_report["scene_manifest_sha256"] != scene_hash
                or fill_metrics["scene_manifest_sha256"] != scene_hash
                or model_manifest["rock_count"] != fill_metrics["rock_count"]):
            raise ValueError("geometry_preparation_cross_output_mismatch")
        report = {
            "schema": REPORT_SCHEMA,
            "status": "formal_scene_geometry_bundle_candidate",
            "config_sha256": sha256(config_path),
            "scene_manifest_sha256": scene_hash,
            "rock_count": model_manifest["rock_count"],
            "expected_nq": model_manifest["expected_nq"],
            "expected_nv": model_manifest["expected_nv"],
            "outputs": {
                layout["model"]: sha256(model_path),
                layout["model_manifest"]: sha256(model_manifest_path),
                f"{layout['mpm_fill_directory']}/layered-bank.npz": sha256(
                    fill_dir / "layered-bank.npz"),
                f"{layout['mpm_fill_directory']}/metrics.json": sha256(
                    fill_metrics_path),
                f"{layout['mpm_fill_directory']}/report.json": sha256(
                    fill_dir / "report.json"),
            },
            "checks": {
                "single_scene_manifest_bound": True,
                "model_and_fill_rock_count_match": True,
                "mpm_particle_partition": fill_metrics["checks"][
                    "particle_partition"],
                "retained_particle_center_inside_rock_zero": fill_metrics[
                    "checks"]["retained_particle_center_inside_rock_zero"],
            },
            "time_zero_executed": False,
            "physics_step_executed": False,
            "mpm_or_gpu_run_executed": False,
            "formal_scene_qualified": False,
            "physics_qualified": False,
        }
        if not all(report["checks"].values()):
            raise ValueError("geometry_preparation_checks_failed")
        (temporary / layout["bundle_report"]).write_bytes(_json_bytes(report))
        os.rename(temporary, output)
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
    report = prepare(
        args.config, args.scene_manifest, args.repository_root, args.output_dir)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
