#!/usr/bin/env python3
"""Compile and exercise 64/256/512-rock time-zero capacity models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

from build_mixed_material_capacity_preflight import (
    build_capacity_model,
    load_and_validate_config,
    render_capacity_bridge,
    sha256,
)


REPORT_SCHEMA = "excavation_mixed_material_capacity_preflight_report_v1"


def _compile(source: Path, output: Path, args) -> tuple[list[str], float]:
    command = [args.cxx, "-O2", "-std=c++17", "-Wall", "-Wextra"]
    command += [item for include in (
        args.mechanism_header_dir, args.controller_header_dir,
        args.mujoco_include) for item in ("-I", include)]
    command += [
        str(source), f"-L{args.mujoco_library_dir}",
        f"-Wl,-rpath,{args.mujoco_library_dir}",
        "-l:libmujoco.so.2.3.8", "-o", str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode or completed.stdout.strip() or completed.stderr.strip():
        raise RuntimeError(
            "capacity_bridge_compile_failed:"
            f"{completed.returncode}:{completed.stdout}:{completed.stderr}")
    return command, elapsed


def _time_zero(binary: Path, model: Path) -> tuple[dict, dict, dict]:
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    try:
        ready_line = process.stdout.readline().rstrip("\n")
        ready_elapsed = time.perf_counter() - started
        if not ready_line.startswith("@READY "):
            raise RuntimeError("capacity_ready_missing:" + ready_line)
        ready = json.loads(ready_line[7:])
        init_started = time.perf_counter()
        process.stdin.write("init_default\n")
        process.stdin.flush()
        state_line = process.stdout.readline().rstrip("\n")
        init_elapsed = time.perf_counter() - init_started
        if not state_line.startswith("@STATE "):
            raise RuntimeError("capacity_state_missing:" + state_line)
        state = json.loads(state_line[7:])
        process.stdin.write("close\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode or stdout.strip() or stderr.strip():
            raise RuntimeError(
                "capacity_close_failed:"
                f"{process.returncode}:{stdout}:{stderr}")
        return ready, state, {
            "ready_wall_seconds": ready_elapsed,
            "init_default_wall_seconds": init_elapsed,
            "ready_message_bytes": len(ready_line.encode("utf-8")),
            "state_message_bytes": len(state_line.encode("utf-8")),
        }
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
    output = Path(args.output).resolve()
    config = load_and_validate_config(config_path, repository)
    started = time.perf_counter()
    tiers = []
    with tempfile.TemporaryDirectory(prefix="mixed-capacity-preflight-") as temp:
        work = Path(temp)
        bridge_source = work / "capacity-bridge.cc"
        bridge_text, bridge_manifest = render_capacity_bridge(
            config_path, repository)
        bridge_source.write_text(bridge_text, encoding="utf-8")
        binary = work / "capacity-bridge"
        compile_command, compile_wall = _compile(bridge_source, binary, args)
        for rock_count in config["capacity"]["preflight_rock_counts"]:
            model_first = work / f"capacity-{rock_count}-first.xml"
            model_second = work / f"capacity-{rock_count}-second.xml"
            build_started = time.perf_counter()
            first_manifest = build_capacity_model(
                config_path, repository, rock_count, model_first)
            build_wall = time.perf_counter() - build_started
            second_manifest = build_capacity_model(
                config_path, repository, rock_count, model_second)
            deterministic = (
                model_first.read_bytes() == model_second.read_bytes()
                and first_manifest == second_manifest)
            ready, state, timings = _time_zero(binary, model_first)
            expected_ids = [f"r1-rock-{index:03d}"
                            for index in range(rock_count)]
            rocks = state.get("rocks", [])
            applied = state.get("applied_rocks_mpm", [])
            checks = {
                "model_generation_deterministic": deterministic,
                "ready_schema": ready.get("schema") ==
                    config["protocol"]["ready_schema"],
                "state_schema": state.get("schema") ==
                    config["protocol"]["state_schema"],
                "ready_rock_count": ready.get("rock_count") == rock_count,
                "ready_wrench_capacity": ready.get(
                    "maximum_external_wrench_count") == rock_count + 1,
                "ready_stable_id_order": ready.get(
                    "allowed_external_wrench_bodies") ==
                    ["bucket", *expected_ids],
                "state_time_zero": state.get("time_s") == 0,
                "state_stable_id_order": [item.get("rock_id") for item in rocks]
                    == expected_ids,
                "state_per_rock_applied_zero": (
                    [item.get("rock_id") for item in applied] == expected_ids
                    and all(item.get("force_world_n") == [0, 0, 0]
                            and item.get("torque_world_nm") == [0, 0, 0]
                            for item in applied)),
                "model_dimensions": (
                    len(state.get("mechanism_qpos", [])) == 19
                    and len(state.get("mechanism_qvel", [])) == 18
                    and all(len(item.get("qpos", [])) == 7
                            and len(item.get("qvel", [])) == 6
                            for item in rocks)),
                "no_physics_step": state.get("time_s") == 0,
            }
            tiers.append({
                "rock_count": rock_count,
                "model_sha256": sha256(model_first),
                "model_bytes": model_first.stat().st_size,
                "expected_nq": 19 + 7 * rock_count,
                "expected_nv": 18 + 6 * rock_count,
                "build_wall_seconds": build_wall,
                **timings,
                "checks": {key: bool(value) for key, value in checks.items()},
                "qualified": bool(all(checks.values())),
            })
    all_passed = all(tier["qualified"] for tier in tiers)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "capacity_time_zero_preflight_passed" if all_passed
        else "capacity_time_zero_preflight_failed",
        "capacity_preflight_qualified": all_passed,
        "physics_qualified": False,
        "formal_scene_qualified": False,
        "config_sha256": sha256(config_path),
        "bridge": {
            "source_sha256": bridge_manifest["generated_source_sha256"],
            "manifest": bridge_manifest,
            "compile_command": compile_command,
            "compile_wall_seconds": compile_wall,
        },
        "tiers": tiers,
        "environment": {"python": platform.python_version()},
        "wall_seconds": time.perf_counter() - started,
        "scope_limit": (
            "synthetic 64/256/512-rock model generation, bridge compile, READY "
            "and time-zero STATE only; zero mj_step, zero MPM/GPU, no settling, "
            "no formal scene or physics qualification"),
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
        "capacity_preflight_qualified": report[
            "capacity_preflight_qualified"],
        "tiers": [{"rock_count": tier["rock_count"],
                   "qualified": tier["qualified"]}
                  for tier in report["tiers"]],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
