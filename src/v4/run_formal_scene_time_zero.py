#!/usr/bin/env python3
"""Validate an atomic actual-N geometry bundle and emit its time-zero report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

from build_mixed_material_capacity_preflight import render_capacity_bridge
from run_mixed_material_capacity_preflight import _compile, _time_zero


CONFIG_SCHEMA = "excavation_formal_scene_time_zero_config_v1"
BUNDLE_SCHEMA = "excavation_formal_scene_geometry_bundle_report_v1"
MODEL_MANIFEST_SCHEMA = "excavation_formal_scene_model_adapter_manifest_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("formal_time_zero_json_object_required")
    return value


def _binding(path: str | Path) -> dict:
    target = Path(path).resolve()
    if not target.is_file():
        raise ValueError("formal_time_zero_missing_artifact:" + str(target))
    return {"path": str(target), "sha256": sha256(target)}


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    config = _load_json(path)
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "actual_n_model_load_and_time_zero_protocol_only"):
        raise ValueError("formal_time_zero_config_schema_or_status")
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, dict) or set(frozen) != {
            "capacity_protocol_config", "geometry_preparation_config"}:
        raise ValueError("formal_time_zero_frozen_input_set")
    repository = Path(repository_root).resolve()
    for name, binding in frozen.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError("formal_time_zero_binding_shape:" + name)
        target = repository / binding["path"]
        if not target.is_file() or sha256(target) != binding["sha256"]:
            raise ValueError("formal_time_zero_binding_mismatch:" + name)
    if config.get("bundle_layout") != {
            "model": "actual-n-scene.xml",
            "model_manifest": "actual-n-model-manifest.json",
            "bundle_report": "geometry-bundle-report.json"}:
        raise ValueError("formal_time_zero_bundle_layout_drift")
    if config.get("report_contract") != {
            "schema": "excavation_fullscale_mixed_scene_time_zero_report_v1",
            "passing_status": "fullscale_time_zero_passed",
            "failure_status": "fullscale_time_zero_failed",
            "checks": [
                "model_loaded", "ready_protocol_passed", "state_time_zero",
                "actual_n_matches", "finite_state",
                "zero_external_wrenches", "zero_forbidden_contact_samples"]}:
        raise ValueError("formal_time_zero_report_contract_drift")
    if config.get("scope") != {
            "geometry_bundle_validation": True,
            "bridge_compile": True,
            "time_zero_protocol": True,
            "physics_step": False,
            "mpm_or_gpu_run": False,
            "settling": False,
            "fullscale_scene_qualification": False,
            "p1a_single_scoop_authorization": False,
            "training": False}:
        raise ValueError("formal_time_zero_scope_expanded")
    return config


def validate_bundle(config: dict, scene_manifest_path: str | Path,
                    bundle_directory: str | Path) -> dict:
    scene_path = Path(scene_manifest_path).resolve()
    directory = Path(bundle_directory).resolve()
    if not scene_path.is_file() or not directory.is_dir():
        raise ValueError("formal_time_zero_input_missing")
    layout = config["bundle_layout"]
    report_path = directory / layout["bundle_report"]
    model_path = directory / layout["model"]
    manifest_path = directory / layout["model_manifest"]
    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    scene = _load_json(scene_path)
    expected_outputs = {
        "actual-n-scene.xml", "actual-n-model-manifest.json",
        "mpm-fill/layered-bank.npz", "mpm-fill/metrics.json",
        "mpm-fill/report.json",
    }
    outputs = report.get("outputs")
    if (report.get("schema") != BUNDLE_SCHEMA or report.get("status") !=
            "formal_scene_geometry_bundle_candidate" or
            not isinstance(outputs, dict) or set(outputs) != expected_outputs or
            any(not (directory / relative).is_file() or
                    sha256(directory / relative) != digest
                    for relative, digest in outputs.items()) or
            not isinstance(report.get("checks"), dict) or
            not report["checks"] or not all(report["checks"].values()) or
            report.get("time_zero_executed") is not False or
            report.get("physics_step_executed") is not False or
            report.get("formal_scene_qualified") is not False or
            report.get("physics_qualified") is not False):
        raise ValueError("formal_time_zero_geometry_bundle_invalid")
    scene_hash = sha256(scene_path)
    if (manifest.get("schema") != MODEL_MANIFEST_SCHEMA or
            manifest.get("status") != "formal_scene_actual_n_model_generated" or
            manifest.get("source_scene_manifest_sha256") != scene_hash or
            manifest.get("candidate_model_sha256") != sha256(model_path) or
            report.get("scene_manifest_sha256") != scene_hash or
            report.get("rock_count") != manifest.get("rock_count") or
            report.get("rock_count") != scene.get("rock_count") or
            report.get("expected_nq") != manifest.get("expected_nq") or
            report.get("expected_nv") != manifest.get("expected_nv") or
            manifest.get("time_zero_executed") is not False or
            manifest.get("physics_step_executed") is not False):
        raise ValueError("formal_time_zero_bundle_cross_binding")
    return {
        "scene": scene,
        "scene_binding": _binding(scene_path),
        "bundle_report": report,
        "bundle_binding": _binding(report_path),
        "model_manifest": manifest,
        "model_binding": _binding(model_path),
        "model_path": model_path,
    }


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def build_report(config: dict, bundle: dict, ready: dict, state: dict,
                 bridge: dict) -> dict:
    manifest = bundle["model_manifest"]
    count = manifest["rock_count"]
    expected_ids = [f"r1-rock-{index:03d}" for index in range(count)]
    rocks = state.get("rocks", [])
    applied_rocks = state.get("applied_rocks_mpm", [])
    zeros = [0, 0, 0]
    zero_external = (
        state.get("applied_bucket_mpm_force_world_n") == zeros and
        state.get("applied_bucket_mpm_torque_world_nm") == zeros and
        isinstance(applied_rocks, list) and
        [item.get("rock_id") for item in applied_rocks] == expected_ids and
        all(item.get("force_world_n") == zeros and
            item.get("torque_world_nm") == zeros for item in applied_rocks))
    checks = {
        "model_loaded": isinstance(ready, dict) and isinstance(state, dict),
        "ready_protocol_passed": (
            ready.get("schema") == "excavation_mixed_coupled_bridge_capacity_v1" and
            ready.get("rock_count") == count and
            ready.get("maximum_external_wrench_count") == count + 1 and
            ready.get("allowed_external_wrench_bodies") ==
            ["bucket", *expected_ids]),
        "state_time_zero": state.get("time_s") == 0,
        "actual_n_matches": (
            len(rocks) == count and
            [item.get("rock_id") for item in rocks] == expected_ids and
            len(state.get("mechanism_qpos", [])) == 19 and
            len(state.get("mechanism_qvel", [])) == 18 and
            all(len(item.get("qpos", [])) == 7 and
                len(item.get("qvel", [])) == 6 for item in rocks)),
        "finite_state": _finite_tree(state),
        "zero_external_wrenches": zero_external,
        "zero_forbidden_contact_samples": (
            state.get("forbidden_contact_count") == 0),
    }
    passed = all(checks.values())
    contract = config["report_contract"]
    return {
        "schema": contract["schema"],
        "status": (contract["passing_status"] if passed else
                   contract["failure_status"]),
        "inputs": {
            "scene_manifest": bundle["scene_binding"],
            "geometry_bundle_report": bundle["bundle_binding"],
        },
        "model": {
            "binding": bundle["model_binding"],
            "rock_count": count,
            "expected_nq": manifest["expected_nq"],
            "expected_nv": manifest["expected_nv"],
        },
        "rock_count": count,
        "expected_nq": manifest["expected_nq"],
        "expected_nv": manifest["expected_nv"],
        "ready": ready,
        "state": state,
        "bridge": bridge,
        "checks": checks,
        "execution": {
            "mj_step_calls": 0,
            "mpm_or_gpu_steps": 0,
            "physics_qualified": False,
            "fullscale_scene_qualified": False,
            "p1a_single_scoop_authorized": False,
        },
    }


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    config = load_config(config_path, repository)
    bundle = validate_bundle(
        config, args.scene_manifest, args.geometry_bundle_directory)
    capacity_config = repository / config["frozen_inputs"][
        "capacity_protocol_config"]["path"]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="formal-scene-time-zero-") as raw:
        temporary = Path(raw)
        source = temporary / "capacity-bridge.cc"
        source_text, bridge_manifest = render_capacity_bridge(
            capacity_config, repository)
        source.write_text(source_text, encoding="utf-8")
        binary = temporary / "capacity-bridge"
        command, compile_wall = _compile(source, binary, args)
        ready, state, timings = _time_zero(binary, bundle["model_path"])
    bridge = {
        "source_sha256": bridge_manifest["generated_source_sha256"],
        "compile_command": command,
        "compile_wall_seconds": compile_wall,
        "time_zero_timings": timings,
        "environment": {"python": platform.python_version()},
        "wall_seconds": time.perf_counter() - started,
    }
    report = build_report(config, bundle, ready, state, bridge)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("formal_time_zero_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--geometry-bundle-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "checks": report["checks"]},
        sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
