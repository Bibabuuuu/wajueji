"""Run the first bounded zero-command settling smoke for six R2 models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from build_mixed_rigid_bridge_r2 import write_bridge
from mixed_material_r2_mechanism import _validate_config, sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite


def _norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _validate_state_shape(state: dict, rock_count: int) -> None:
    if (state["rock_count"] != rock_count or
            state["maximum_rock_capacity"] != 20 or
            len(state["mechanism_qpos"]) != 19 or
            len(state["mechanism_qvel"]) != 18 or
            len(state["rocks"]) != rock_count or not _finite(state)):
        raise RuntimeError("r2_settling_state_shape_or_finite_contract")
    expected_ids = [f"r2-rock-{index:03d}" for index in range(rock_count)]
    if [rock["rock_id"] for rock in state["rocks"]] != expected_ids:
        raise RuntimeError("r2_settling_rock_identity_contract")
    if not all(len(rock["qpos"]) == 7 and len(rock["qvel"]) == 6
               and len(rock["position"]) == 3 for rock in state["rocks"]):
        raise RuntimeError("r2_settling_per_rock_state_width")


def _run_scene(binary: Path, model: Path, settling: dict) -> dict:
    dt = float(settling["output_dt_s"])
    duration = float(settling["duration_s"])
    step_count = round(duration / dt)
    if step_count <= 0 or not math.isclose(step_count * dt, duration,
                                            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("settling_duration_not_integral")
    substeps = int(settling["internal_substeps_per_output"])
    command = [float(value) for value in settling["command"]]
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        process.kill()
        raise RuntimeError("r2_settling_ready_failed:" + ready_line)
    ready = json.loads(ready_line[7:])
    rock_count = int(ready["rock_count"])
    if (rock_count != 20 or ready["maximum_rock_capacity"] != 20 or
            ready["runtime_add_remove_allowed"] or
            ready["placeholder_rocks_allowed"]):
        process.kill()
        raise RuntimeError("r2_settling_ready_contract")
    process.stdin.write("init_default\n")
    process.stdin.flush()
    initial_line = process.stdout.readline().rstrip("\n")
    if not initial_line.startswith("@STATE "):
        process.kill()
        raise RuntimeError("r2_settling_init_failed:" + initial_line)
    initial = json.loads(initial_line[7:])
    _validate_state_shape(initial, rock_count)
    initial_positions = [rock["position"] for rock in initial["rocks"]]
    total_contacts = {key: 0 for key in initial["contact_substeps"]}
    maximum_closure = 0.0
    maximum_linear_speed = 0.0
    maximum_angular_speed = 0.0
    maximum_quaternion_norm_error = 0.0
    maximum_rock_displacement = 0.0
    maximum_horizontal_displacement = 0.0
    maximum_vertical_displacement = 0.0
    maximum_downward_displacement = 0.0
    maximum_upward_displacement = 0.0
    state = initial
    for output_step in range(step_count):
        fields = [
            "step", f"{float(state['time_s']):.17g}", f"{dt:.17g}", str(substeps),
            *(f"{value:.17g}" for value in command),
            str(int(settling["external_wrench_count"])),
        ]
        process.stdin.write(" ".join(fields) + "\n")
        process.stdin.flush()
        line = process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            stderr = process.stderr.read()
            process.kill()
            raise RuntimeError(
                f"r2_settling_step_failed:{output_step}:{line}:{stderr}")
        state = json.loads(line[7:])
        _validate_state_shape(state, rock_count)
        expected_time = (output_step + 1) * dt
        if not math.isclose(float(state["time_s"]), expected_time,
                            rel_tol=0.0, abs_tol=1e-9):
            process.kill()
            raise RuntimeError("r2_settling_time_contract")
        if int(state["forbidden_contact_count"]) != 0:
            process.kill()
            raise RuntimeError("r2_settling_forbidden_contact")
        maximum_closure = max(maximum_closure, float(state["closure_m"]))
        for key, value in state["contact_substeps"].items():
            total_contacts[key] += int(value)
        for index, rock in enumerate(state["rocks"]):
            maximum_linear_speed = max(
                maximum_linear_speed, _norm(rock["qvel"][:3]))
            maximum_angular_speed = max(
                maximum_angular_speed, _norm(rock["qvel"][3:]))
            maximum_quaternion_norm_error = max(
                maximum_quaternion_norm_error,
                abs(_norm(rock["qpos"][3:7]) - 1.0))
            maximum_rock_displacement = max(
                maximum_rock_displacement,
                _norm([
                    float(rock["position"][axis]) - initial_positions[index][axis]
                    for axis in range(3)
                ]),
            )
            horizontal = _norm([
                float(rock["position"][axis]) - initial_positions[index][axis]
                for axis in range(2)
            ])
            vertical = float(rock["position"][2]) - initial_positions[index][2]
            maximum_horizontal_displacement = max(
                maximum_horizontal_displacement, horizontal)
            maximum_vertical_displacement = max(
                maximum_vertical_displacement, abs(vertical))
            maximum_downward_displacement = max(
                maximum_downward_displacement, -vertical)
            maximum_upward_displacement = max(
                maximum_upward_displacement, vertical)
    process.stdin.write("close\n")
    process.stdin.flush()
    stdout_tail, stderr = process.communicate(timeout=10)
    if process.returncode or stdout_tail.strip() or stderr.strip():
        raise RuntimeError(
            f"r2_settling_close_failed:{process.returncode}:{stdout_tail}:{stderr}")
    checks = {
        "completed_all_output_steps": step_count == 120,
        "final_time_one_second": math.isclose(
            float(state["time_s"]), 1.0, rel_tol=0.0, abs_tol=1e-9),
        "twenty_identities_preserved": len(state["rocks"]) == 20,
        "finite_state_throughout": True,
        "no_forbidden_contacts": int(state["forbidden_contact_count"]) == 0,
        "closure_within_one_mm": (
            maximum_closure <= float(settling["maximum_closure_m"])),
        "ground_contact_observed": total_contacts["rock_ground"] > 0,
        "quaternions_normalized": maximum_quaternion_norm_error <= 1e-9,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("r2_settling_acceptance_failed:" + ",".join(failed))
    return {
        "output_step_count": step_count,
        "internal_substep_count": step_count * substeps,
        "final_time_s": float(state["time_s"]),
        "total_contact_substeps": total_contacts,
        "maximum_closure_m": maximum_closure,
        "maximum_linear_speed_m_s": maximum_linear_speed,
        "maximum_angular_speed_rad_s": maximum_angular_speed,
        "maximum_quaternion_norm_error": maximum_quaternion_norm_error,
        "maximum_rock_displacement_m": maximum_rock_displacement,
        "maximum_horizontal_displacement_m": maximum_horizontal_displacement,
        "maximum_vertical_displacement_m": maximum_vertical_displacement,
        "maximum_downward_displacement_m": maximum_downward_displacement,
        "maximum_upward_displacement_m": maximum_upward_displacement,
        "final_maximum_linear_speed_m_s": max(
            _norm(rock["qvel"][:3]) for rock in state["rocks"]),
        "final_maximum_angular_speed_rad_s": max(
            _norm(rock["qvel"][3:]) for rock in state["rocks"]),
        "checks": checks,
    }


def run_settling(config_path: str | Path, model_root: str | Path,
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
    load_report_path = model_root / "report.json"
    init_report_path = model_root / "bridge-init-report.json"
    load_report = json.loads(load_report_path.read_text(encoding="utf-8"))
    init_report = json.loads(init_report_path.read_text(encoding="utf-8"))
    if (load_report["status"] != "r2_six_scene_load_only_passed" or
            init_report["status"] != "r2_six_scene_bridge_init_passed"):
        raise ValueError("r2_settling_prerequisite_report_status")
    load_index = {record["scene_id"]: record for record in load_report["scenes"]}
    output.mkdir(parents=True, exist_ok=True)
    generated_source = output / "generated-bridge-r2.cc"
    source_manifest = write_bridge(config_path, repository, generated_source)
    write_json(output / "generated-bridge-r2-manifest.json", source_manifest)
    with tempfile.TemporaryDirectory(prefix="mixed-material-r2-settling-") as temporary:
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
            model_path = model_root / scene_id / "model.xml"
            if sha256(model_path) != load_index[scene_id]["model_sha256"]:
                raise ValueError("r2_settling_model_sha256_mismatch:" + scene_id)
            metrics = _run_scene(binary, model_path, config["settling_probe"])
            scene_report = {
                "schema": "excavation_mixed_material_r2_settling_scene_report_v1",
                "status": "r2_settling_smoke_passed",
                "scene_id": scene_id,
                "rock_count": 20,
                "model_sha256": sha256(model_path),
                **metrics,
                "physics_qualified": False,
                "scope_limit": (
                    "one-second zero-command pure-rigid settling smoke only; not "
                    "excavation, retention, dump, MPM coupling or fill-rate qualification"
                ),
            }
            report_path = output / scene_id / "settling-report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(report_path, scene_report)
            scene_records.append({
                "scene_id": scene_id,
                "rock_count": 20,
                "model_sha256": scene_report["model_sha256"],
                "report_sha256": sha256(report_path),
                "total_contact_substeps": metrics["total_contact_substeps"],
                "maximum_closure_m": metrics["maximum_closure_m"],
                "maximum_rock_displacement_m": metrics["maximum_rock_displacement_m"],
                "maximum_horizontal_displacement_m": metrics[
                    "maximum_horizontal_displacement_m"],
                "maximum_downward_displacement_m": metrics[
                    "maximum_downward_displacement_m"],
                "final_maximum_linear_speed_m_s": metrics[
                    "final_maximum_linear_speed_m_s"],
            })
    checks = {
        "all_six_scenes_completed": len(scene_records) == 6,
        "all_120_identities_preserved": all(
            record["rock_count"] == 20 for record in scene_records),
        "ground_contact_observed_in_every_scene": all(
            record["total_contact_substeps"]["rock_ground"] > 0
            for record in scene_records),
        "closure_within_one_mm_in_every_scene": all(
            record["maximum_closure_m"] <= 0.001 for record in scene_records),
    }
    if not all(checks.values()):
        raise RuntimeError("r2_settling_root_acceptance_failed")
    report = {
        "schema": "excavation_mixed_material_r2_settling_report_v1",
        "status": "r2_six_scene_settling_smoke_passed",
        "scene_count": len(scene_records),
        "rock_count_per_scene": 20,
        "total_preserved_rock_identities": 120,
        "duration_per_scene_s": config["settling_probe"]["duration_s"],
        "total_internal_substeps": sum(2400 for _ in scene_records),
        "checks": checks,
        "scenes": scene_records,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "load_report_sha256": sha256(load_report_path),
            "bridge_init_report_sha256": sha256(init_report_path),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_settling.py"),
            "generated_source_sha256": sha256(generated_source),
            "compile_command": compile_command_record,
        },
        "scope_limit": (
            "bounded one-second zero-command numerical/contact smoke; does not qualify "
            "multi-rock excavation, carrying, dump, MPM coupling or bucket fill rate"
        ),
    }
    write_json(output / "settling-report.json", report)
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
    report = run_settling(
        args.config, args.model_root, args.output_root, args.repository_root,
        args.cxx, args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir)
    print(json.dumps({
        "status": report["status"],
        "scene_count": report["scene_count"],
        "total_internal_substeps": report["total_internal_substeps"],
        "physics_qualified": report["physics_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
