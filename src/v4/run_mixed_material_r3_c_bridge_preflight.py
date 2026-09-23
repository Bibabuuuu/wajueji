#!/usr/bin/env python3
"""Qualify R3-C two-rock model/bridge generation without advancing physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np

from build_mixed_coupled_bridge_r3_c import write_bridge
from build_mixed_multirock_candidate_r3_c import (
    build_two_rock_candidate,
    sha256,
    validate_config,
)


REPORT_SCHEMA = "excavation_mixed_material_r3_c_bridge_preflight_report_v1"


def _compile(cxx: str, source: Path, output: Path, includes: list[str],
             library_dir: Path) -> list[str]:
    command = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra"]
    command += [item for include in includes for item in ("-I", include)]
    command += [
        str(source), f"-L{library_dir}", f"-Wl,-rpath,{library_dir}",
        "-l:libmujoco.so.2.3.8", "-o", str(output),
    ]
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False)
    if completed.returncode or completed.stdout.strip() or completed.stderr.strip():
        raise RuntimeError(
            "r3_c_bridge_compile_failed:"
            f"{completed.returncode}:{completed.stdout}:{completed.stderr}")
    return command


def _time_zero_protocol(binary: Path, model: Path) -> tuple[dict, dict]:
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        ready_line = process.stdout.readline().rstrip("\n")
        if not ready_line.startswith("@READY "):
            raise RuntimeError("r3_c_ready_missing:" + ready_line)
        ready = json.loads(ready_line[7:])
        process.stdin.write("init_default\n")
        process.stdin.flush()
        state_line = process.stdout.readline().rstrip("\n")
        if not state_line.startswith("@STATE "):
            raise RuntimeError("r3_c_state_missing:" + state_line)
        state = json.loads(state_line[7:])
        process.stdin.write("close\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode or stdout.strip() or stderr.strip():
            raise RuntimeError(
                f"r3_c_bridge_close_failed:{process.returncode}:{stdout}:{stderr}")
        return ready, state
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    inventory = Path(args.inventory).resolve()
    output = Path(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mixed-material-r3-c-preflight-") as temporary:
        work = Path(temporary)
        model_first = work / "model-first.xml"
        model_second = work / "model-second.xml"
        model_manifest_first = build_two_rock_candidate(
            config_path, repository, inventory, model_first)
        model_manifest_second = build_two_rock_candidate(
            config_path, repository, inventory, model_second)
        source_first = work / "bridge-first.cc"
        source_second = work / "bridge-second.cc"
        source_manifest_first = write_bridge(
            config_path, repository, source_first)
        source_manifest_second = write_bridge(
            config_path, repository, source_second)
        binary = work / "bridge"
        compile_command = _compile(
            args.cxx, source_first, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        ready, state = _time_zero_protocol(binary, model_first)
        model_hash_first = sha256(model_first)
        model_hash_second = sha256(model_second)
        source_hash_first = sha256(source_first)
        source_hash_second = sha256(source_second)
    expected_ids = ["r1-rock-000", "r1-rock-001"]
    rocks = state.get("rocks", [])
    finite_vectors = (
        list(state.get("mechanism_qpos", [])) +
        list(state.get("mechanism_qvel", [])) +
        list(state.get("bucket_origin", [])) +
        [number for rock in rocks for field in (
            "qpos", "qvel", "position", "linear_velocity_world_m_s",
            "angular_velocity_world_rad_s") for number in rock.get(field, [])])
    applied = state.get("applied_rocks_mpm", [])
    checks = {
        "model_generation_deterministic": (
            model_hash_first == model_hash_second and
            model_manifest_first == model_manifest_second),
        "bridge_generation_deterministic": (
            source_hash_first == source_hash_second and
            source_manifest_first == source_manifest_second),
        "model_contract_exactly_two_rocks": (
            model_manifest_first["expected_candidate_contract"] == {
                "nq": 33, "nv": 30, "nu": 12, "nplugin": 0,
                "rock_count": 2}),
        "ready_schema": ready.get("schema") == config["bridge"]["ready_schema"],
        "ready_exactly_two_rocks": ready.get("rock_count") == 2,
        "ready_three_named_wrenches": (
            ready.get("maximum_external_wrench_count") == 3 and
            ready.get("allowed_external_wrench_bodies") ==
            config["bridge"]["allowed_external_wrench_bodies"]),
        "state_schema": state.get("schema") == config["bridge"]["state_schema"],
        "time_zero_no_step": state.get("time_s") == 0,
        "stable_rock_id_order": (
            [rock.get("rock_id") for rock in rocks] == expected_ids),
        "per_rock_world_velocity_present": all(
            len(rock.get("linear_velocity_world_m_s", [])) == 3 and
            len(rock.get("angular_velocity_world_rad_s", [])) == 3
            for rock in rocks),
        "per_rock_applied_wrench_zero_at_init": (
            [item.get("rock_id") for item in applied] == expected_ids and
            all(not np.any(item.get("force_world_n", [])) and
                not np.any(item.get("torque_world_nm", []))
                for item in applied)),
        "bucket_applied_wrench_zero_at_init": (
            not np.any(state.get("applied_bucket_mpm_force_world_n", [])) and
            not np.any(state.get("applied_bucket_mpm_torque_world_nm", []))),
        "interval_reference_equals_bucket_origin_at_init": np.array_equal(
            np.asarray(state.get("bucket_interval_reference_point_world_m", [])),
            np.asarray(state.get("bucket_origin", []))),
        "all_protocol_state_finite": bool(
            finite_vectors and np.isfinite(np.asarray(finite_vectors)).all()),
        "frozen_inputs_unmodified": (
            not model_manifest_first["frozen_r1_r2_r3_a_r3_b_modified"] and
            not source_manifest_first["frozen_r1_r2_r3_a_r3_b_modified"]),
        "physics_and_training_not_claimed": (
            not config["first_gate"]["physics_qualified"] and
            not config["first_gate"]["training_qualified"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "r3_c_two_rock_bridge_preflight_passed" if all(checks.values())
        else "r3_c_two_rock_bridge_preflight_failed",
        "bridge_preflight_qualified": bool(all(checks.values())),
        "physics_qualified": False,
        "training_qualified": False,
        "checks": {key: bool(value) for key, value in checks.items()},
        "ready": ready,
        "time_zero_state": state,
        "inputs": {
            "config_sha256": sha256(config_path),
            "inventory_sha256": sha256(inventory),
            "model_sha256": model_hash_first,
            "model_manifest": model_manifest_first,
            "bridge_source_sha256": source_hash_first,
            "bridge_source_manifest": source_manifest_first,
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "python": platform.python_version(),
            "wall_seconds": time.perf_counter() - started,
        },
        "scope_limit": (
            "two-rock model and three-body external-wrench bridge generation, "
            "compile, and time-zero protocol only; zero mj_step, zero MPM/GPU "
            "steps, no physics qualification, no complete in-process reset, "
            "no official envelope, and no training"),
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
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
        "bridge_preflight_qualified": report["bridge_preflight_qualified"],
        "physics_qualified": report["physics_qualified"],
        "checks": report["checks"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
