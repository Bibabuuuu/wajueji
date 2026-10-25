"""Compile the derived R2 bridge and initialize all six first-admission models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from build_mixed_rigid_bridge_r2 import write_bridge
from mixed_material_r2_mechanism import _validate_config, sha256, write_json


def _compile(cxx: str, source: Path, output: Path, include_dirs,
             library_dir: Path) -> list[str]:
    command = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra"]
    command += ["-I" + str(Path(value).resolve()) for value in include_dirs]
    command += [
        str(source.resolve()), "-L" + str(library_dir.resolve()),
        "-Wl,-rpath," + str(library_dir.resolve()),
        "-l:libmujoco.so.2.3.8", "-o", str(output.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("r2_bridge_compile_failed:" + completed.stderr)
    return command


def _finite(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _line_hash(value: str) -> str:
    return hashlib.sha256((value + "\n").encode("utf-8")).hexdigest()


def _run_init(binary: Path, model: Path) -> tuple[dict, dict, str, str]:
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        process.kill()
        raise RuntimeError("r2_bridge_ready_failed:" + ready_line)
    process.stdin.write("init_default\n")
    process.stdin.flush()
    state_line = process.stdout.readline().rstrip("\n")
    if not state_line.startswith("@STATE "):
        process.kill()
        raise RuntimeError("r2_bridge_init_failed:" + state_line)
    process.stdin.write("close\n")
    process.stdin.flush()
    stdout_tail, stderr = process.communicate(timeout=10)
    if process.returncode or stdout_tail.strip() or stderr.strip():
        raise RuntimeError(
            f"r2_bridge_close_failed:{process.returncode}:{stdout_tail}:{stderr}")
    return (
        json.loads(ready_line[7:]), json.loads(state_line[7:]),
        _line_hash(ready_line), _line_hash(state_line),
    )


def build_bridge_init(config_path: str | Path, model_root: str | Path,
                      output_root: str | Path, repository_root: str | Path,
                      cxx: str, mujoco_include: str | Path,
                      mujoco_library_dir: str | Path,
                      mechanism_header_dir: str | Path,
                      controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    model_root = Path(model_root).resolve()
    output = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    model_report_path = model_root / "report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    if (model_report["status"] != "r2_six_scene_load_only_passed" or
            model_report["scene_count"] != 6 or model_report["physics_qualified"]):
        raise ValueError("invalid_r2_model_load_report")
    indexed = {record["scene_id"]: record for record in model_report["scenes"]}
    output.mkdir(parents=True, exist_ok=True)
    generated_source = output / "generated-bridge-r2.cc"
    source_manifest = write_bridge(config_path, repository, generated_source)
    source_manifest_path = output / "generated-bridge-r2-manifest.json"
    write_json(source_manifest_path, source_manifest)

    with tempfile.TemporaryDirectory(prefix="mixed-material-r2-bridge-init-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, generated_source, binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        compile_command_record = [
            "<temporary>/mixed-rigid-bridge-r2"
            if value == str(binary.resolve()) else value
            for value in compile_command
        ]
        scene_records = []
        for scene_id in config["first_admission"]["scene_ids"]:
            model_record = indexed.get(scene_id)
            if model_record is None:
                raise ValueError("missing_first_admission_model:" + scene_id)
            model_path = model_root / scene_id / "model.xml"
            if sha256(model_path) != model_record["model_sha256"]:
                raise ValueError("r2_model_sha256_mismatch:" + scene_id)
            ready, state, ready_hash, state_hash = _run_init(binary, model_path)
            rocks = state["rocks"]
            checks = {
                "ready_schema_v2": (
                    ready["schema"] == config["bridge_candidate"]["ready_schema"]),
                "state_schema_v2": (
                    state["schema"] == config["bridge_candidate"]["state_schema"]),
                "actual_count_twenty": ready["rock_count"] == state["rock_count"] == 20,
                "maximum_capacity_twenty": (
                    ready["maximum_rock_capacity"] ==
                    state["maximum_rock_capacity"] == 20),
                "no_runtime_add_remove": ready["runtime_add_remove_allowed"] is False,
                "no_placeholder_rocks": ready["placeholder_rocks_allowed"] is False,
                "time_remained_zero": state["time_s"] == 0,
                "mechanism_state_dimensions": (
                    len(state["mechanism_qpos"]) == 19 and
                    len(state["mechanism_qvel"]) == 18),
                "twenty_exact_rock_records": len(rocks) == 20,
                "stable_contiguous_rock_ids": [
                    rock["rock_id"] for rock in rocks
                ] == [f"r2-rock-{index:03d}" for index in range(20)],
                "per_rock_state_dimensions": all(
                    len(rock["qpos"]) == 7 and len(rock["qvel"]) == 6 and
                    len(rock["position"]) == 3 for rock in rocks),
                "legacy_first_alias_exact": (
                    state["rock_qpos"] == rocks[0]["qpos"] and
                    state["rock_qvel"] == rocks[0]["qvel"] and
                    state["rock_position"] == rocks[0]["position"]),
                "finite_state": _finite(state),
                "zero_initial_contact_counters": all(
                    value == 0 for value in state["contact_substeps"].values()),
                "zero_initial_forbidden_count": state["forbidden_contact_count"] == 0,
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(
                    "r2_bridge_init_contract_failed:" + scene_id + ":" + ",".join(failed))
            scene_report = {
                "schema": "excavation_mixed_material_r2_bridge_init_scene_report_v1",
                "status": "r2_bridge_init_passed",
                "scene_id": scene_id,
                "rock_count": 20,
                "maximum_rock_capacity": 20,
                "ready_line_sha256": ready_hash,
                "state_line_sha256": state_hash,
                "model_sha256": sha256(model_path),
                "checks": checks,
                "time_advanced": False,
                "mj_step_called": False,
                "physics_qualified": False,
                "scope_limit": (
                    "bridge start, mj_forward-based initialization and state emission only; "
                    "no integration step or multi-rock dynamics qualification"
                ),
            }
            scene_report_path = output / scene_id / "bridge-init-report.json"
            scene_report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(scene_report_path, scene_report)
            scene_records.append({
                "scene_id": scene_id,
                "rock_count": 20,
                "model_sha256": scene_report["model_sha256"],
                "ready_line_sha256": ready_hash,
                "state_line_sha256": state_hash,
                "report_sha256": sha256(scene_report_path),
            })

    checks = {
        "all_six_models_initialized": len(scene_records) == 6,
        "all_models_actual_twenty": all(
            record["rock_count"] == 20 for record in scene_records),
        "one_generated_bridge_source": source_manifest["maximum_rock_count"] == 20,
        "actual_count_from_loaded_model": source_manifest["actual_count_from_loaded_model"],
        "no_placeholder_contract": not source_manifest["placeholder_rocks_allowed"],
        "no_runtime_add_remove_contract": not source_manifest["runtime_add_remove_allowed"],
        "no_time_step": True,
    }
    report = {
        "schema": "excavation_mixed_material_r2_bridge_init_report_v1",
        "status": "r2_six_scene_bridge_init_passed",
        "scene_count": len(scene_records),
        "rock_count_per_scene": 20,
        "total_initialized_rock_records": 120,
        "maximum_rock_capacity": 20,
        "bridge_source_count": 1,
        "time_advanced": False,
        "mj_forward_initialization_called": True,
        "mj_step_called": False,
        "physics_qualified": False,
        "checks": checks,
        "scenes": scene_records,
        "inputs": {
            "config_sha256": sha256(config_path),
            "model_load_report_sha256": sha256(model_report_path),
            "base_r1_bridge_sha256": source_manifest["base_r1_source_sha256"],
        },
        "implementation": {
            "generator_sha256": sha256(
                repository / "src/v4/build_mixed_rigid_bridge_r2.py"),
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_bridge_init.py"),
            "generated_source_sha256": sha256(generated_source),
            "generated_source_manifest_sha256": sha256(source_manifest_path),
            "compile_command": compile_command_record,
        },
        "scope_limit": (
            "one R2 bridge source initialized six 20-rock models and emitted finite state "
            "at time zero; no mj_step, settling, contact evolution, excavation or fill-rate "
            "qualification"
        ),
    }
    write_json(output / "bridge-init-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = build_bridge_init(
        args.config, args.model_root, args.output_root, args.repository_root,
        args.cxx, args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir)
    print(json.dumps({
        "status": report["status"],
        "scene_count": report["scene_count"],
        "total_initialized_rock_records": report["total_initialized_rock_records"],
        "mj_step_called": report["mj_step_called"],
        "physics_qualified": report["physics_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
