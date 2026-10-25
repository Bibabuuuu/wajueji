#!/usr/bin/env python3
"""Exercise the old 887-rock count only as a D-063 time-zero capacity bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from build_mixed_material_capacity_preflight import (
    build_capacity_model,
    load_and_validate_config as load_capacity_config,
    render_capacity_bridge,
    sha256,
)
from run_mixed_material_capacity_preflight import _compile, _time_zero


CONFIG_SCHEMA = "excavation_mixed_material_capacity_upper_bound_config_v1"
REPORT_SCHEMA = "excavation_mixed_material_capacity_upper_bound_report_v1"


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> tuple[dict, Path]:
    repository = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "d063_advisory_capacity_upper_bound_only"):
        raise ValueError("upper_bound_config_schema_or_status")
    binding = config.get("frozen_capacity_config")
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or not isinstance(binding["sha256"], str)):
        raise ValueError("upper_bound_capacity_binding")
    base_path = repository / binding["path"]
    if not base_path.is_file() or sha256(base_path) != binding["sha256"]:
        raise ValueError("upper_bound_capacity_binding_mismatch")
    base = load_capacity_config(base_path, repository)
    count = config.get("reference_rock_count")
    if (isinstance(count, bool) or not isinstance(count, int) or count != 887
            or count > int(base["capacity"]["maximum_formal_rock_count"])):
        raise ValueError("upper_bound_reference_count_drift")
    if config.get("reference_role") != (
            "old_exact_recipe_count_retained_only_as_capacity_upper_bound_candidate"):
        raise ValueError("upper_bound_reference_role_drift")
    if config.get("scope") != {
        "synthetic_geometry_only": True,
        "bridge_compile": True,
        "time_zero_protocol": True,
        "one_clear_step": False,
        "mpm_or_gpu_run": False,
        "settling": False,
        "formal_scene_count_claim": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
        "training": False,
    }:
        raise ValueError("upper_bound_scope_expanded")
    return config, base_path


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config, base_config_path = load_and_validate_config(config_path, repository)
    rock_count = int(config["reference_rock_count"])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mixed-capacity-upper-bound-") as temp:
        work = Path(temp)
        source_text, bridge_manifest = render_capacity_bridge(
            base_config_path, repository)
        source_path = work / "capacity-bridge.cc"
        source_path.write_text(source_text, encoding="utf-8")
        binary = work / "capacity-bridge"
        compile_command, compile_wall = _compile(source_path, binary, args)
        first = work / "capacity-887-first.xml"
        second = work / "capacity-887-second.xml"
        first_manifest = build_capacity_model(
            base_config_path, repository, rock_count, first)
        second_manifest = build_capacity_model(
            base_config_path, repository, rock_count, second)
        ready, state, timings = _time_zero(binary, first)
        expected_ids = [f"r1-rock-{index:03d}" for index in range(rock_count)]
        rocks = state.get("rocks", [])
        applied = state.get("applied_rocks_mpm", [])
        checks = {
            "model_generation_deterministic": (
                first.read_bytes() == second.read_bytes()
                and first_manifest == second_manifest),
            "ready_rock_count": ready.get("rock_count") == rock_count,
            "ready_wrench_capacity": ready.get(
                "maximum_external_wrench_count") == rock_count + 1,
            "ready_stable_id_order": ready.get(
                "allowed_external_wrench_bodies") == ["bucket", *expected_ids],
            "state_time_zero": state.get("time_s") == 0,
            "state_stable_id_order": [item.get("rock_id") for item in rocks]
                == expected_ids,
            "state_applied_wrenches_zero": (
                [item.get("rock_id") for item in applied] == expected_ids
                and all(item.get("force_world_n") == [0, 0, 0]
                        and item.get("torque_world_nm") == [0, 0, 0]
                        for item in applied)),
            "expected_dimensions": (
                first_manifest.get("expected_nq") == 19 + 7 * rock_count
                and first_manifest.get("expected_nv") == 18 + 6 * rock_count),
            "no_physics_step": state.get("time_s") == 0,
        }
        passed = all(checks.values())
        report = {
            "schema": REPORT_SCHEMA,
            "status": "capacity_upper_bound_time_zero_passed" if passed
                else "capacity_upper_bound_time_zero_failed",
            "capacity_upper_bound_qualified": passed,
            "formal_scene_count_claim": False,
            "formal_scene_qualified": False,
            "physics_qualified": False,
            "config_sha256": sha256(config_path),
            "base_capacity_config_sha256": sha256(base_config_path),
            "reference_rock_count": rock_count,
            "reference_role": config["reference_role"],
            "model_sha256": sha256(first),
            "model_bytes": first.stat().st_size,
            "expected_nq": 19 + 7 * rock_count,
            "expected_nv": 18 + 6 * rock_count,
            "ready_message_bytes": timings["ready_message_bytes"],
            "state_message_bytes": timings["state_message_bytes"],
            "ready_wall_seconds": timings["ready_wall_seconds"],
            "init_default_wall_seconds": timings["init_default_wall_seconds"],
            "bridge_compile_wall_seconds": compile_wall,
            "bridge_source_sha256": bridge_manifest["generated_source_sha256"],
            "compile_command": compile_command,
            "checks": checks,
            "wall_seconds": time.perf_counter() - started,
            "scope_limit": (
                "synthetic 887-rock upper-bound model and time-zero protocol only; "
                "zero mj_step, zero MPM/GPU, no formal count, scene, settling, or "
                "physics qualification"),
        }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "reference_rock_count": report["reference_rock_count"],
        "capacity_upper_bound_qualified": report[
            "capacity_upper_bound_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
