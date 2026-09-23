"""Replay the two-scene intake candidate and add a short lift/hold gate."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile

from mixed_material_r2_mechanism import sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite
from run_mixed_material_r2_controlled_intake import (
    COMMAND_LIMITS,
    _exact_keys,
    _norm,
    _number,
    _scene_observations,
    _summary,
    _validate_config as _validate_intake_config,
)


SCHEMA = "excavation_mixed_material_r2_intake_lift_config_v1"


def _validate_config(config: dict) -> None:
    _exact_keys(config, {
        "schema", "source_intake_config", "source_intake_report",
        "additional_phases", "acceptance", "scope",
    }, "lift_config")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_intake_lift_schema")
    _exact_keys(config["source_intake_config"], {"path", "sha256"},
                "source_intake_config")
    _exact_keys(config["source_intake_report"],
                {"path", "sha256", "required_status"},
                "source_intake_report")
    if (config["source_intake_report"]["required_status"] !=
            "two_scene_controlled_intake_candidate"):
        raise ValueError("invalid_intake_prerequisite_status")
    phases = config["additional_phases"]
    if (not isinstance(phases, list) or len(phases) != 2
            or [phase.get("name") for phase in phases]
            != ["short_lift", "hold_after_lift"]):
        raise ValueError("invalid_lift_phase_sequence")
    for index, phase in enumerate(phases):
        _exact_keys(phase, {"name", "duration_s", "command"},
                    f"additional_phases/{index}")
        _number(phase["duration_s"], "lift_phase_duration", positive=True)
        if not isinstance(phase["command"], list) or len(phase["command"]) != 4:
            raise ValueError("invalid_lift_command_width")
        for axis, value in enumerate(phase["command"]):
            number = _number(value, "lift_command")
            if abs(number) > COMMAND_LIMITS[axis] + 1e-12:
                raise ValueError("lift_command_above_bridge_limit")
    if phases[0]["command"] != [0.0, 0.02, 0.0, 0.0]:
        raise ValueError("short_lift_command_changed")
    if phases[1]["command"] != [0.0, 0.0, 0.0, 0.0]:
        raise ValueError("hold_after_lift_not_zero")
    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_final_retained_full_rock_count",
        "minimum_vertical_lift_from_intake_end_m",
        "minimum_final_continuous_full_inside_duration_s",
        "maximum_final_retained_rock_speed_m_s", "maximum_closure_m",
        "required_forbidden_contact_count",
        "force_and_torque_are_diagnostic_not_capacity_thresholds",
    }, "lift_acceptance")
    if (not isinstance(acceptance["minimum_final_retained_full_rock_count"], int)
            or acceptance["minimum_final_retained_full_rock_count"] < 1
            or acceptance["required_forbidden_contact_count"] != 0):
        raise ValueError("invalid_lift_acceptance_integer")
    for key in ("minimum_vertical_lift_from_intake_end_m",
                "minimum_final_continuous_full_inside_duration_s",
                "maximum_final_retained_rock_speed_m_s", "maximum_closure_m"):
        _number(acceptance[key], key, positive=True)
    if acceptance[
            "force_and_torque_are_diagnostic_not_capacity_thresholds"] is not True:
        raise ValueError("unjustified_force_capacity_threshold")
    scope = config["scope"]
    _exact_keys(scope, {
        "two_scene_short_lift_hold_diagnostic", "ten_second_hold_qualification",
        "all_six_scene_qualification", "dump", "mpm_coupling", "fill_rate",
        "training", "physics_qualified",
    }, "lift_scope")
    if (scope["two_scene_short_lift_hold_diagnostic"] is not True
            or any(scope[key] is not False for key in (
                "ten_second_hold_qualification", "all_six_scene_qualification",
                "dump", "mpm_coupling", "fill_rate", "training",
                "physics_qualified"))):
        raise ValueError("expanded_intake_lift_scope")


def _run_scene(binary: Path, model: Path, plan: dict, inventory: dict,
               proxy: dict, base_config: dict, lift_config: dict) -> dict:
    integration = base_config["integration"]
    base_acceptance = base_config["acceptance"]
    acceptance = lift_config["acceptance"]
    dt = float(integration["output_dt_s"])
    substeps = int(integration["internal_substeps_per_output"])
    phases = copy.deepcopy(base_config["phases"]) + copy.deepcopy(
        lift_config["additional_phases"])
    tolerance = float(base_acceptance["full_envelope_tolerance_m"])
    straddle = float(base_acceptance["mouth_straddle_depth_margin_m"])
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        raise RuntimeError("intake_lift_ready_failed:" + ready_line)
    process.stdin.write("init_default\n")
    process.stdin.flush()
    initial_line = process.stdout.readline().rstrip("\n")
    if not initial_line.startswith("@STATE "):
        raise RuntimeError("intake_lift_init_failed:" + initial_line)
    state = json.loads(initial_line[7:])
    if len(state["rocks"]) != 20 or not _finite(state):
        raise RuntimeError("intake_lift_initial_contract")
    rock_ids = [rock["rock_id"] for rock in state["rocks"]]
    consecutive_full = {rock_id: 0 for rock_id in rock_ids}
    maximum_consecutive_full = {rock_id: 0 for rock_id in rock_ids}
    contacts = {key: 0 for key in state["contact_substeps"]}
    maximum_force = 0.0
    maximum_torque = 0.0
    maximum_closure = 0.0
    maximum_forbidden = 0
    output_steps = 0
    runtime_error = None
    intake_end_positions = None
    intake_end_full_ids = []
    phase_end = {}
    final_observations = _scene_observations(
        state, plan, inventory, proxy, tolerance, straddle)
    try:
        for phase in phases:
            phase_steps = round(float(phase["duration_s"]) / dt)
            command = [float(value) for value in phase["command"]]
            for _ in range(phase_steps):
                fields = [
                    "step", f"{float(state['time_s']):.17g}", f"{dt:.17g}",
                    str(substeps), *(f"{value:.17g}" for value in command), "0",
                ]
                process.stdin.write(" ".join(fields) + "\n")
                process.stdin.flush()
                line = process.stdout.readline().rstrip("\n")
                if not line.startswith("@STATE "):
                    stderr = process.stderr.read().strip()
                    runtime_error = line + (":" + stderr if stderr else "")
                    raise RuntimeError("intake_lift_step_failed")
                state = json.loads(line[7:])
                if len(state["rocks"]) != 20 or not _finite(state):
                    runtime_error = "nonfinite_or_identity_count_changed"
                    raise RuntimeError("intake_lift_state_contract")
                final_observations = _scene_observations(
                    state, plan, inventory, proxy, tolerance, straddle)
                for item in final_observations:
                    rock_id = item["rock_id"]
                    consecutive_full[rock_id] = (
                        consecutive_full[rock_id] + 1
                        if item["all_vertices_inside_bucket_prism"] else 0)
                    maximum_consecutive_full[rock_id] = max(
                        maximum_consecutive_full[rock_id],
                        consecutive_full[rock_id])
                for key, value in state["contact_substeps"].items():
                    contacts[key] += int(value)
                maximum_force = max(
                    maximum_force,
                    _norm(state["last_bucket_rock_force_world_n"]))
                maximum_torque = max(
                    maximum_torque,
                    _norm(state[
                        "last_bucket_rock_torque_about_bucket_origin_world_nm"]))
                maximum_closure = max(maximum_closure, float(state["closure_m"]))
                maximum_forbidden = max(
                    maximum_forbidden, int(state["forbidden_contact_count"]))
                output_steps += 1
            phase_end[phase["name"]] = {
                "time_s": state["time_s"],
                "bucket_origin_world_m": state["bucket_origin"],
                "observation": _summary(final_observations),
                "cumulative_bucket_contact_substeps": contacts["rock_bucket"],
                "maximum_bucket_rock_force_n": maximum_force,
                "maximum_bucket_rock_torque_nm": maximum_torque,
            }
            if phase["name"] == "settle_after_intake":
                intake_end_positions = {
                    rock["rock_id"]: list(rock["position"])
                    for rock in state["rocks"]}
                intake_end_full_ids = [
                    item["rock_id"] for item in final_observations
                    if item["all_vertices_inside_bucket_prism"]]
    except RuntimeError:
        pass
    finally:
        if process.poll() is None:
            try:
                process.stdin.write("close\n")
                process.stdin.flush()
                stdout_tail, stderr_tail = process.communicate(timeout=5)
                if (process.returncode or stdout_tail.strip() or stderr_tail.strip()) and not runtime_error:
                    runtime_error = (
                        f"bridge_close_failed:{process.returncode}:"
                        f"{stdout_tail.strip()}:{stderr_tail.strip()}")
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                runtime_error = runtime_error or "bridge_close_failed"
    final_by_id = {item["rock_id"]: item for item in final_observations}
    final_state_by_id = {rock["rock_id"]: rock for rock in state["rocks"]}
    required_steps = math.ceil(
        float(acceptance["minimum_final_continuous_full_inside_duration_s"])
        / dt - 1e-12)
    retained_ids = [
        rock_id for rock_id in rock_ids
        if final_by_id[rock_id]["all_vertices_inside_bucket_prism"]
        and consecutive_full[rock_id] >= required_steps]
    lift_by_id = {
        rock_id: (final_state_by_id[rock_id]["position"][2]
                  - intake_end_positions[rock_id][2])
        for rock_id in rock_ids
    } if intake_end_positions is not None else {rock_id: 0.0 for rock_id in rock_ids}
    carried_from_intake_end_ids = [
        rock_id for rock_id in retained_ids
        if rock_id in intake_end_full_ids]
    lifted_carried_from_intake_end_ids = [
        rock_id for rock_id in carried_from_intake_end_ids
        if lift_by_id[rock_id] >= float(
            acceptance["minimum_vertical_lift_from_intake_end_m"])]
    maximum_retained_speed = max(
        (final_by_id[rock_id]["speed_m_s"] for rock_id in retained_ids),
        default=0.0)
    expected_steps = sum(round(float(phase["duration_s"]) / dt)
                         for phase in phases)
    checks = {
        "completed_full_schedule": runtime_error is None and output_steps == expected_steps,
        "twenty_rock_identities_preserved": (
            [rock["rock_id"] for rock in state["rocks"]] == rock_ids),
        "intake_end_had_full_rock": len(intake_end_full_ids) >= 1,
        "final_retained_full_rock_count": (
            len(retained_ids) >= int(
                acceptance["minimum_final_retained_full_rock_count"])),
        "same_intake_end_rock_retained_and_lifted": (
            len(lifted_carried_from_intake_end_ids) >= int(
                acceptance["minimum_final_retained_full_rock_count"])),
        "retained_rocks_final_speed_within_limit": (
            maximum_retained_speed <= float(
                acceptance["maximum_final_retained_rock_speed_m_s"])),
        "closure_within_one_mm": maximum_closure <= float(
            acceptance["maximum_closure_m"]),
        "no_forbidden_contacts": maximum_forbidden == 0,
        "finite_state": _finite(state),
    }
    return {
        "status": ("short_lift_hold_scene_candidate" if all(checks.values())
                   else "short_lift_hold_scene_needs_review"),
        "runtime_error": runtime_error,
        "checks": checks,
        "output_steps_completed": output_steps,
        "expected_output_steps": expected_steps,
        "simulated_duration_s": float(state["time_s"]),
        "contact_substeps": contacts,
        "maximum_closure_m": maximum_closure,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "maximum_bucket_rock_force_n": maximum_force,
        "maximum_bucket_rock_torque_nm": maximum_torque,
        "intake_end_full_rock_ids": intake_end_full_ids,
        "final_retained_full_rock_ids": retained_ids,
        "carried_from_intake_end_full_rock_ids": carried_from_intake_end_ids,
        "lifted_carried_from_intake_end_full_rock_ids": (
            lifted_carried_from_intake_end_ids),
        "final_retained_full_rock_count": len(retained_ids),
        "final_retained_mass_kg": sum(
            final_by_id[rock_id]["mass_kg"] for rock_id in retained_ids),
        "final_retained_maximum_speed_m_s": maximum_retained_speed,
        "vertical_lift_from_intake_end_m_by_rock": {
            rock_id: lift_by_id[rock_id] for rock_id in retained_ids},
        "final_observation": _summary(final_observations),
        "phase_end_observations": phase_end,
    }


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("intake_lift_output_already_exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    base_config_path = repository / config["source_intake_config"]["path"]
    if sha256(base_config_path) != config["source_intake_config"]["sha256"]:
        raise ValueError("intake_lift_source_config_sha256_mismatch")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    _validate_intake_config(base_config)
    source_report_path = repository / config["source_intake_report"]["path"]
    if sha256(source_report_path) != config["source_intake_report"]["sha256"]:
        raise ValueError("intake_lift_source_report_sha256_mismatch")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report["status"] != config["source_intake_report"]["required_status"]:
        raise ValueError("intake_lift_source_report_status")
    stable_root = repository / base_config["stable_model_root"]["path"]
    stable_report = json.loads((stable_root / "report.json").read_text(encoding="utf-8"))
    stable_index = {item["scene_id"]: item for item in stable_report["scenes"]}
    bridge_source = stable_root / "generated-bridge-r2.cc"
    geometry = base_config["bucket_geometry"]
    inventory_path = Path(geometry["inventory_path"])
    proxy_path = repository / geometry["proxy_manifest_path"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    r2_config = json.loads(
        (repository / base_config["source_r2_config"]["path"]).read_text(
            encoding="utf-8"))
    preflight_root = repository / r2_config["assembly_preflight"]["root"]
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="mixed-r2-intake-lift-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, bridge_source, binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        compile_record = [
            "<temporary>/mixed-rigid-bridge-r2"
            if value == str(binary.resolve()) else value
            for value in compile_command]
        scenes = []
        for scene_id in base_config["selected_scenes"]:
            model_path = stable_root / scene_id / "model.xml"
            if sha256(model_path) != stable_index[scene_id]["model_sha256"]:
                raise ValueError("intake_lift_model_sha256_mismatch:" + scene_id)
            plan_path = preflight_root / scene_id / "assembly-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            metrics = _run_scene(
                binary, model_path, plan, inventory, proxy, base_config, config)
            scene_report = {
                "schema": "excavation_mixed_material_r2_intake_lift_scene_report_v1",
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "status": metrics["status"],
                "metrics": metrics,
                "physics_qualified": False,
                "scope_limit": (
                    "replays the two-scene intake trajectory plus 1.5-second lift and "
                    "2-second hold; no ten-second hold, dump, fill-rate or MPM qualification"),
            }
            scene_dir = output / scene_id
            scene_dir.mkdir(parents=True)
            scene_path = scene_dir / "report.json"
            write_json(scene_path, scene_report)
            scenes.append({
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "status": metrics["status"],
                "intake_end_full_rock_count": len(
                    metrics["intake_end_full_rock_ids"]),
                "final_retained_full_rock_count": metrics[
                    "final_retained_full_rock_count"],
                "final_retained_mass_kg": metrics["final_retained_mass_kg"],
                "lifted_carried_from_intake_end_full_rock_count": len(
                    metrics[
                        "lifted_carried_from_intake_end_full_rock_ids"]),
                "maximum_bucket_rock_force_n": metrics[
                    "maximum_bucket_rock_force_n"],
                "maximum_bucket_rock_torque_nm": metrics[
                    "maximum_bucket_rock_torque_nm"],
                "runtime_error": metrics["runtime_error"],
                "report_sha256": sha256(scene_path),
            })
    all_candidate = all(
        scene["status"] == "short_lift_hold_scene_candidate" for scene in scenes)
    report = {
        "schema": "excavation_mixed_material_r2_intake_lift_report_v1",
        "status": ("two_scene_short_lift_hold_candidate" if all_candidate
                   else "short_lift_hold_needs_review"),
        "scene_count": len(scenes),
        "scenes": scenes,
        "ten_second_hold_qualified": False,
        "all_six_scene_qualification": False,
        "dump_qualified": False,
        "mpm_coupling_qualified": False,
        "fill_rate_qualified": False,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "source_intake_config_sha256": sha256(base_config_path),
            "source_intake_report_sha256": sha256(source_report_path),
            "bridge_source_sha256": sha256(bridge_source),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_intake_lift.py"),
            "compile_command": compile_record,
        },
        "scope_limit": (
            "two selected scenes, one fixed intake path, 1.5-second lift and 2-second "
            "hold only; no six-scene, ten-second hold, dump, fill-rate, MPM, hydraulic "
            "capacity or training qualification"),
    }
    write_json(output / "report.json", report)
    (output / "README.md").write_text(
        "# R2进斗后短举升保持报告\n\n"
        f"根状态：`{report['status']}`。本轮在两场进斗轨迹后追加1.5秒举升和"
        "2秒保持，用于区分真实随斗离地与铲斗仅罩住地面石块。仍不代表10秒"
        "保持、六场成功率、卸料、满斗率或MPM资格。\n",
        encoding="utf-8")
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
        "status": report["status"], "scene_count": report["scene_count"],
        "physics_qualified": report["physics_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
