"""Pre-settle R2 rocks, bake stable poses, and verify rebaked initial models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from build_mixed_rigid_bridge_r2 import write_bridge
from build_scene_reconstruction import _boundary_clearance, _surface_height
from mixed_material_r0 import load_obj, quaternion_matrix_wxyz
from mixed_material_r2_mechanism import (
    STABLE_INITIAL_STATE_SCHEMA,
    _validate_config as validate_r2_config,
    build_candidate,
    sha256,
    write_json,
)
from run_mixed_material_r2_bridge_init import _compile, _finite
from run_mixed_material_r2_settling import _norm, _validate_state_shape


CONFIG_SCHEMA = "excavation_mixed_material_r2_stabilization_config_v1"


def _validate_config(config: dict) -> None:
    if config.get("schema") != CONFIG_SCHEMA or not _finite(config):
        raise ValueError("invalid_r2_stabilization_config")
    settling = config["pre_settling"]
    verification = config["rebaked_verification"]
    if (float(settling["maximum_duration_s"]) != 6.0 or
            float(settling["continuous_stability_window_s"]) != 0.5 or
            float(settling["output_dt_s"]) != 1.0 / 120.0 or
            int(settling["internal_substeps_per_output"]) != 20 or
            float(settling["maximum_linear_speed_m_s"]) != 0.002 or
            float(settling["maximum_angular_speed_rad_s"]) != 0.03 or
            settling["command"] != [0.0] * 4 or
            int(settling["external_wrench_count"]) != 0):
        raise ValueError("invalid_pre_settling_contract")
    if (float(verification["duration_s"]) != 1.0 or
            float(verification["output_dt_s"]) != 1.0 / 120.0 or
            int(verification["internal_substeps_per_output"]) != 20 or
            float(verification["maximum_rock_displacement_m"]) != 0.005 or
            verification["command"] != [0.0] * 4 or
            int(verification["external_wrench_count"]) != 0):
        raise ValueError("invalid_rebaked_verification_contract")
    scope = config["scope"]
    if (scope["source_scene_manifests_modified"] or
            scope["source_assembly_plans_modified"] or
            scope["rock_count_or_size_modified"] or
            scope["controlled_excavation"] or
            scope["physics_qualification"] or scope["training"]):
        raise ValueError("r2_stabilization_scope_expanded")
    if (config["spatial_review"][
            "require_original_boundary_safety_retention"] is not True):
        raise ValueError("original_boundary_safety_must_be_retained")


def _start(binary: Path, model: Path) -> tuple[subprocess.Popen, dict, dict]:
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        process.kill()
        raise RuntimeError("r2_stabilization_ready_failed:" + ready_line)
    ready = json.loads(ready_line[7:])
    process.stdin.write("init_default\n")
    process.stdin.flush()
    state_line = process.stdout.readline().rstrip("\n")
    if not state_line.startswith("@STATE "):
        process.kill()
        raise RuntimeError("r2_stabilization_init_failed:" + state_line)
    state = json.loads(state_line[7:])
    _validate_state_shape(state, int(ready["rock_count"]))
    return process, ready, state


def _step(process: subprocess.Popen, state: dict, contract: dict) -> dict:
    fields = [
        "step", f"{float(state['time_s']):.17g}",
        f"{float(contract['output_dt_s']):.17g}",
        str(int(contract["internal_substeps_per_output"])),
        *(f"{float(value):.17g}" for value in contract["command"]),
        str(int(contract["external_wrench_count"])),
    ]
    process.stdin.write(" ".join(fields) + "\n")
    process.stdin.flush()
    line = process.stdout.readline().rstrip("\n")
    if not line.startswith("@STATE "):
        stderr = process.stderr.read()
        process.kill()
        raise RuntimeError("r2_stabilization_step_failed:" + line + ":" + stderr)
    result = json.loads(line[7:])
    _validate_state_shape(result, len(state["rocks"]))
    return result


def _close(process: subprocess.Popen) -> None:
    process.stdin.write("close\n")
    process.stdin.flush()
    stdout_tail, stderr = process.communicate(timeout=10)
    if process.returncode or stdout_tail.strip() or stderr.strip():
        raise RuntimeError(
            f"r2_stabilization_close_failed:{process.returncode}:"
            f"{stdout_tail}:{stderr}")


def _speed_maxima(state: dict) -> tuple[float, float]:
    return (
        max(_norm(rock["qvel"][:3]) for rock in state["rocks"]),
        max(_norm(rock["qvel"][3:]) for rock in state["rocks"]),
    )


def _pre_settle(binary: Path, model: Path, contract: dict) -> tuple[dict, dict]:
    process, ready, state = _start(binary, model)
    if ready["rock_count"] != 20:
        process.kill()
        raise RuntimeError("r2_stabilization_requires_twenty_rocks")
    initial_positions = [rock["position"] for rock in state["rocks"]]
    dt = float(contract["output_dt_s"])
    maximum_steps = round(float(contract["maximum_duration_s"]) / dt)
    window_steps = round(float(contract["continuous_stability_window_s"]) / dt)
    stable_steps = 0
    total_contacts = {key: 0 for key in state["contact_substeps"]}
    maximum_closure = 0.0
    maximum_horizontal_shift = 0.0
    maximum_downward_shift = 0.0
    for step_index in range(maximum_steps):
        state = _step(process, state, contract)
        linear, angular = _speed_maxima(state)
        maximum_closure = max(maximum_closure, float(state["closure_m"]))
        for key, value in state["contact_substeps"].items():
            total_contacts[key] += int(value)
        for index, rock in enumerate(state["rocks"]):
            delta = [float(rock["position"][axis]) - initial_positions[index][axis]
                     for axis in range(3)]
            maximum_horizontal_shift = max(
                maximum_horizontal_shift, _norm(delta[:2]))
            maximum_downward_shift = max(maximum_downward_shift, -delta[2])
        stable = (
            linear <= float(contract["maximum_linear_speed_m_s"]) and
            angular <= float(contract["maximum_angular_speed_rad_s"]) and
            float(state["closure_m"]) <= float(contract["maximum_closure_m"]) and
            int(state["forbidden_contact_count"]) ==
            int(contract["required_forbidden_contact_count"]))
        stable_steps = stable_steps + 1 if stable else 0
        if stable_steps >= window_steps:
            break
    _close(process)
    final_linear, final_angular = _speed_maxima(state)
    passed = stable_steps >= window_steps
    metrics = {
        "status": "pre_settling_stable" if passed else "pre_settling_timeout",
        "elapsed_s": float(state["time_s"]),
        "output_step_count": step_index + 1,
        "internal_substep_count": (
            (step_index + 1) * int(contract["internal_substeps_per_output"])),
        "continuous_stable_steps": stable_steps,
        "required_continuous_stable_steps": window_steps,
        "final_maximum_linear_speed_m_s": final_linear,
        "final_maximum_angular_speed_rad_s": final_angular,
        "maximum_closure_m": maximum_closure,
        "maximum_horizontal_shift_from_source_m": maximum_horizontal_shift,
        "maximum_downward_shift_from_source_m": maximum_downward_shift,
        "total_contact_substeps": total_contacts,
        "checks": {
            "continuous_stability_window_reached": passed,
            "no_forbidden_contacts": int(state["forbidden_contact_count"]) == 0,
            "closure_within_limit": maximum_closure <=
            float(contract["maximum_closure_m"]),
            "ground_contact_observed": total_contacts["rock_ground"] > 0,
        },
    }
    return state, metrics


def _stable_state(scene_id: str, plan: dict, state: dict,
                  pre_settle_metrics: dict, source_model: Path) -> dict:
    records = []
    for slot, (source, rock) in enumerate(zip(plan["rocks"], state["rocks"])):
        quaternion = np.asarray(rock["qpos"][3:7], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0:
            quaternion = -quaternion
        records.append({
            "slot": slot,
            "source_rock_id": source["rock_id"],
            "body_name": f"r2-rock-{slot:03d}",
            "position_world_m": [float(value) for value in rock["qpos"][:3]],
            "orientation_quaternion_wxyz": quaternion.tolist(),
            "initial_qvel": [0.0] * 6,
            "equivalent_diameter_m": float(source["equivalent_diameter_m"]),
            "mass_kg": float(source["mass_kg"]),
        })
    return {
        "schema": STABLE_INITIAL_STATE_SCHEMA,
        "status": "stable_initial_state_candidate",
        "scene_id": scene_id,
        "scene_class": plan["scene_class"],
        "rock_count": len(records),
        "source_assembly_plan_sha256": sha256(source_model.parent / "assembly-plan.json")
        if (source_model.parent / "assembly-plan.json").is_file() else None,
        "source_model_sha256": sha256(source_model),
        "pre_settling_elapsed_s": pre_settle_metrics["elapsed_s"],
        "velocity_policy": "zero_all_freejoint_velocities_after_pose_bake",
        "rocks": records,
        "scope_limit": (
            "candidate initial poses baked from gravity-only rigid settling; rocks "
            "remain dynamic and require independent rebaked stability and spatial review"),
    }


def _spatial_review(plan: dict, stable: dict, scene_manifest: dict,
                    repository: Path, config: dict) -> dict:
    polygon = np.asarray(scene_manifest["region_polygon_xy_m"], dtype=float)[:-1]
    surface = scene_manifest["surface_model"]
    source_by_id = {rock["rock_id"]: rock for rock in plan["rocks"]}
    minimum_boundary = math.inf
    minimum_surface_clearance = math.inf
    maximum_horizontal_shift = 0.0
    for record in stable["rocks"]:
        source = source_by_id[record["source_rock_id"]]
        vertices, _ = load_obj(repository / source["asset_path"])
        rotation = quaternion_matrix_wxyz(record["orientation_quaternion_wxyz"])
        world = (vertices * float(source["equivalent_diameter_m"]) @ rotation.T
                 + np.asarray(record["position_world_m"], dtype=float))
        minimum_boundary = min(minimum_boundary, _boundary_clearance(world[:, :2], polygon))
        minimum_surface_clearance = min(
            minimum_surface_clearance,
            min(float(vertex[2]) - _surface_height(
                surface, float(vertex[0]), float(vertex[1])) for vertex in world))
        source_position = np.asarray(source["position_world_m"], dtype=float)
        maximum_horizontal_shift = max(
            maximum_horizontal_shift,
            float(np.linalg.norm(
                np.asarray(record["position_world_m"][:2]) - source_position[:2])))
    review = config["spatial_review"]
    source_boundary = 0.08
    checks = {
        "all_vertices_inside_source_region": minimum_boundary >= 0.0,
        "original_boundary_safety_retained": minimum_boundary >= source_boundary,
        "surface_penetration_within_limit": minimum_surface_clearance >= -float(
            review["maximum_surface_penetration_m"]),
        "center_horizontal_shift_within_limit": maximum_horizontal_shift <= float(
            review["maximum_center_horizontal_shift_m"]),
        "rock_count_unchanged": len(stable["rocks"]) == len(plan["rocks"]),
        "rock_sizes_and_masses_unchanged": all(
            math.isclose(record["equivalent_diameter_m"],
                         float(source_by_id[record["source_rock_id"]]["equivalent_diameter_m"]),
                         rel_tol=0.0, abs_tol=0.0) and
            math.isclose(record["mass_kg"],
                         float(source_by_id[record["source_rock_id"]]["mass_kg"]),
                         rel_tol=0.0, abs_tol=0.0)
            for record in stable["rocks"]),
    }
    return {
        "minimum_boundary_clearance_m": minimum_boundary,
        "source_boundary_safety_requirement_m": source_boundary,
        "minimum_vertex_analytic_surface_clearance_m": minimum_surface_clearance,
        "maximum_center_horizontal_shift_m": maximum_horizontal_shift,
        "checks": checks,
    }


def _verify_rebaked(binary: Path, model: Path, contract: dict) -> dict:
    process, ready, state = _start(binary, model)
    initial_positions = [rock["position"] for rock in state["rocks"]]
    step_count = round(float(contract["duration_s"]) /
                       float(contract["output_dt_s"]))
    maximum_displacement = 0.0
    maximum_closure = 0.0
    total_ground = 0
    forbidden = 0
    for _ in range(step_count):
        state = _step(process, state, contract)
        maximum_closure = max(maximum_closure, float(state["closure_m"]))
        total_ground += int(state["contact_substeps"]["rock_ground"])
        forbidden += int(state["forbidden_contact_count"])
        for index, rock in enumerate(state["rocks"]):
            maximum_displacement = max(
                maximum_displacement,
                _norm([float(rock["position"][axis]) - initial_positions[index][axis]
                       for axis in range(3)]))
    _close(process)
    final_linear, final_angular = _speed_maxima(state)
    checks = {
        "twenty_rocks_preserved": ready["rock_count"] == len(state["rocks"]) == 20,
        "one_second_completed": math.isclose(
            float(state["time_s"]), float(contract["duration_s"]),
            rel_tol=0.0, abs_tol=1e-9),
        "displacement_within_limit": maximum_displacement <= float(
            contract["maximum_rock_displacement_m"]),
        "final_linear_speed_within_limit": final_linear <= float(
            contract["maximum_final_linear_speed_m_s"]),
        "final_angular_speed_within_limit": final_angular <= float(
            contract["maximum_final_angular_speed_rad_s"]),
        "closure_within_limit": maximum_closure <= float(contract["maximum_closure_m"]),
        "no_forbidden_contacts": forbidden == int(
            contract["required_forbidden_contact_count"]),
        "ground_contact_observed": total_ground > 0,
    }
    return {
        "status": "rebaked_stability_passed" if all(checks.values())
        else "rebaked_stability_failed",
        "maximum_rock_displacement_m": maximum_displacement,
        "final_maximum_linear_speed_m_s": final_linear,
        "final_maximum_angular_speed_rad_s": final_angular,
        "maximum_closure_m": maximum_closure,
        "rock_ground_contact_substeps": total_ground,
        "forbidden_contact_count": forbidden,
        "checks": checks,
    }


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    r2_config_path = repository / config["source_r2_config"]["path"]
    if sha256(r2_config_path) != config["source_r2_config"]["sha256"]:
        raise ValueError("source_r2_config_sha256_mismatch")
    r2_config = json.loads(r2_config_path.read_text(encoding="utf-8"))
    validate_r2_config(r2_config)
    baseline_root = repository / config["baseline_model_root"]["path"]
    required_reports = (
        ("report.json", "load_report_sha256"),
        ("bridge-init-report.json", "bridge_init_report_sha256"),
        ("settling-report.json", "settling_report_sha256"),
    )
    for file_name, hash_key in required_reports:
        if sha256(baseline_root / file_name) != config["baseline_model_root"][hash_key]:
            raise ValueError("baseline_report_sha256_mismatch:" + file_name)
    load_report = json.loads((baseline_root / "report.json").read_text(encoding="utf-8"))
    load_index = {record["scene_id"]: record for record in load_report["scenes"]}
    preflight_root = repository / r2_config["assembly_preflight"]["root"]
    reconstruction_root = repository / r2_config["surface_support"][
        "scene_reconstruction_root"]
    output.mkdir(parents=True, exist_ok=True)
    generated_source = output / "generated-bridge-r2.cc"
    source_manifest = write_bridge(r2_config_path, repository, generated_source)
    write_json(output / "generated-bridge-r2-manifest.json", source_manifest)

    with tempfile.TemporaryDirectory(prefix="mixed-material-r2-stabilize-") as temporary:
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
        for scene_id in r2_config["first_admission"]["scene_ids"]:
            scene_output = output / scene_id
            scene_output.mkdir(parents=True, exist_ok=True)
            baseline_model = baseline_root / scene_id / "model.xml"
            if sha256(baseline_model) != load_index[scene_id]["model_sha256"]:
                raise ValueError("baseline_model_sha256_mismatch:" + scene_id)
            plan_path = preflight_root / scene_id / "assembly-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            final_state, pre_metrics = _pre_settle(
                binary, baseline_model, config["pre_settling"])
            stable = _stable_state(
                scene_id, plan, final_state, pre_metrics, baseline_model)
            stable["source_assembly_plan_sha256"] = sha256(plan_path)
            stable_path = scene_output / "stable-initial-state.json"
            write_json(stable_path, stable)
            scene_manifest_path = reconstruction_root / scene_id / "scene-manifest.json"
            scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
            spatial = _spatial_review(
                plan, stable, scene_manifest, repository, config)
            model_path = scene_output / "model.xml"
            model_manifest = build_candidate(
                r2_config_path, plan_path, model_path, repository, stable_path)
            model_manifest_path = scene_output / "model-manifest.json"
            write_json(model_manifest_path, model_manifest)
            verification = _verify_rebaked(
                binary, model_path, config["rebaked_verification"])
            scene_report = {
                "schema": "excavation_mixed_material_r2_stabilization_scene_report_v1",
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "pre_settling": pre_metrics,
                "spatial_review": spatial,
                "rebaked_verification": verification,
                "stable_initial_state_sha256": sha256(stable_path),
                "model_sha256": sha256(model_path),
                "model_manifest_sha256": sha256(model_manifest_path),
                "heightfield_asset_sha256": model_manifest[
                    "surface_support"]["asset_sha256"],
                "physics_qualified": False,
            }
            required_spatial = (
                spatial["checks"]["all_vertices_inside_source_region"] and
                spatial["checks"]["original_boundary_safety_retained"] and
                spatial["checks"]["surface_penetration_within_limit"] and
                spatial["checks"]["center_horizontal_shift_within_limit"] and
                spatial["checks"]["rock_count_unchanged"] and
                spatial["checks"]["rock_sizes_and_masses_unchanged"])
            passed = (all(pre_metrics["checks"].values()) and required_spatial and
                      all(verification["checks"].values()))
            scene_report["status"] = (
                "stable_initial_state_verified" if passed
                else "stable_initial_state_needs_review")
            scene_report_path = scene_output / "report.json"
            write_json(scene_report_path, scene_report)
            scene_records.append({
                "scene_id": scene_id,
                "status": scene_report["status"],
                "pre_settling_elapsed_s": pre_metrics["elapsed_s"],
                "source_maximum_horizontal_shift_m": spatial[
                    "maximum_center_horizontal_shift_m"],
                "minimum_boundary_clearance_m": spatial[
                    "minimum_boundary_clearance_m"],
                "rebaked_maximum_displacement_m": verification[
                    "maximum_rock_displacement_m"],
                "stable_initial_state_sha256": scene_report[
                    "stable_initial_state_sha256"],
                "model_sha256": scene_report["model_sha256"],
                "report_sha256": sha256(scene_report_path),
            })
    all_verified = all(
        record["status"] == "stable_initial_state_verified"
        for record in scene_records)
    report = {
        "schema": "excavation_mixed_material_r2_stabilization_report_v1",
        "status": ("r2_six_scene_stable_initial_states_verified" if all_verified
                   else "r2_stable_initial_states_need_review"),
        "scene_count": len(scene_records),
        "rock_count_per_scene": 20,
        "source_scene_manifests_modified": False,
        "source_assembly_plans_modified": False,
        "rock_count_or_size_modified": False,
        "physics_qualified": False,
        "scenes": scene_records,
        "inputs": {
            "stabilization_config_sha256": sha256(config_path),
            "source_r2_config_sha256": sha256(r2_config_path),
            "baseline_load_report_sha256": sha256(baseline_root / "report.json"),
            "baseline_bridge_init_report_sha256": sha256(
                baseline_root / "bridge-init-report.json"),
            "baseline_settling_report_sha256": sha256(
                baseline_root / "settling-report.json"),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_stabilization.py"),
            "generated_bridge_source_sha256": sha256(generated_source),
            "compile_command": compile_command_record,
        },
        "scope_limit": (
            "gravity-only stable-pose candidates and one-second rebaked stability "
            "verification; no controlled excavation, carrying, dump, MPM or fill-rate qualification"),
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(
        args.config, args.output_root, args.repository_root, args.cxx,
        args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir)
    print(json.dumps({
        "status": report["status"],
        "scene_count": report["scene_count"],
        "physics_qualified": report["physics_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
