"""Evaluate effective carried rocks and preliminary solid-volume fill rate."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile

from instrument_r2_bridge_contacts import instrument_per_rock_contacts
from mixed_material_r2_mechanism import sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite
from run_mixed_material_r2_controlled_intake import (
    _exact_keys,
    _norm,
    _number,
    _point_in_polygon_with_tolerance,
    _scene_observations,
    _summary,
    _validate_config as _validate_intake_config,
)
from run_mixed_material_r2_intake_lift import (
    _validate_config as _validate_lift_config,
)


SCHEMA = "excavation_mixed_material_r2_effective_load_config_v1"


def _validate_config(config: dict) -> None:
    _exact_keys(config, {
        "schema", "source_intake_config", "source_lift_config",
        "superseded_strict_lift_report", "bucket_capacity", "acceptance",
        "scope",
    }, "effective_load_config")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_effective_load_schema")
    for key in ("source_intake_config", "source_lift_config"):
        _exact_keys(config[key], {"path", "sha256"}, key)
    _exact_keys(config["superseded_strict_lift_report"],
                {"path", "sha256", "required_status"},
                "superseded_strict_lift_report")
    if (config["superseded_strict_lift_report"]["required_status"] !=
            "short_lift_hold_needs_review"):
        raise ValueError("invalid_superseded_lift_status")
    capacity = config["bucket_capacity"]
    _exact_keys(capacity, {
        "value_m3", "source_path", "source_sha256", "source_field",
        "conversion", "metric",
    }, "bucket_capacity")
    if (abs(_number(capacity["value_m3"], "capacity", positive=True)
            - 0.255164729950989) > 1e-15
            or capacity["source_field"] != "capacity_liters"
            or capacity["conversion"] != "capacity_liters / 1000"
            or capacity["metric"] !=
            "stable_carried_rock_solid_volume_divided_by_bucket_capacity"):
        raise ValueError("invalid_bucket_capacity_contract")
    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_carried_rock_count",
        "minimum_vertical_lift_from_lift_start_m",
        "minimum_final_continuous_off_ground_duration_s",
        "relaxed_bucket_neighborhood_margin_m",
        "maximum_relative_center_drift_during_final_window_m",
        "maximum_final_carried_rock_speed_m_s", "maximum_closure_m",
        "required_forbidden_contact_count",
        "strict_full_envelope_is_diagnostic_only",
        "direct_bucket_contact_is_diagnostic_only",
    }, "effective_load_acceptance")
    if (not isinstance(acceptance["minimum_carried_rock_count"], int)
            or isinstance(acceptance["minimum_carried_rock_count"], bool)
            or acceptance["minimum_carried_rock_count"] < 1
            or acceptance["required_forbidden_contact_count"] != 0):
        raise ValueError("invalid_effective_load_integer")
    for key in (
            "minimum_vertical_lift_from_lift_start_m",
            "minimum_final_continuous_off_ground_duration_s",
            "relaxed_bucket_neighborhood_margin_m",
            "maximum_relative_center_drift_during_final_window_m",
            "maximum_final_carried_rock_speed_m_s", "maximum_closure_m"):
        _number(acceptance[key], key, positive=True)
    if (acceptance["strict_full_envelope_is_diagnostic_only"] is not True
            or acceptance["direct_bucket_contact_is_diagnostic_only"] is not True):
        raise ValueError("strict_geometry_or_direct_contact_made_mandatory")
    scope = config["scope"]
    _exact_keys(scope, {
        "two_scene_effective_load_diagnostic", "solid_volume_fill_rate_only",
        "void_compensated_bulk_fill_rate", "all_six_scene_qualification",
        "ten_second_hold_qualification", "dump", "truck_cad", "mpm_coupling",
        "training", "physics_qualified",
    }, "effective_load_scope")
    if (scope["two_scene_effective_load_diagnostic"] is not True
            or scope["solid_volume_fill_rate_only"] is not True
            or any(scope[key] is not False for key in (
                "void_compensated_bulk_fill_rate", "all_six_scene_qualification",
                "ten_second_hold_qualification", "dump", "truck_cad",
                "mpm_coupling", "training", "physics_qualified"))):
        raise ValueError("expanded_effective_load_scope")


def _in_relaxed_neighborhood(observation: dict, inventory: dict,
                             margin: float) -> bool:
    center = observation["center_bucket_m"]
    bucket = inventory["bucket"]
    return (
        _point_in_polygon_with_tolerance(
            [center[0], center[2]], bucket["polygon_xz_m"], margin)
        and bucket["y_limits_m"][0] - margin <= center[1]
        <= bucket["y_limits_m"][1] + margin)


def _run_scene(binary: Path, model: Path, plan: dict, inventory: dict,
               proxy: dict, base_config: dict, lift_config: dict,
               config: dict) -> dict:
    integration = base_config["integration"]
    dt = float(integration["output_dt_s"])
    substeps = int(integration["internal_substeps_per_output"])
    phases = copy.deepcopy(base_config["phases"]) + copy.deepcopy(
        lift_config["additional_phases"])
    base_acceptance = base_config["acceptance"]
    acceptance = config["acceptance"]
    tolerance = float(base_acceptance["full_envelope_tolerance_m"])
    straddle = float(base_acceptance["mouth_straddle_depth_margin_m"])
    window_steps = math.ceil(
        float(acceptance["minimum_final_continuous_off_ground_duration_s"])
        / dt - 1e-12)
    hold_steps = round(float(phases[-1]["duration_s"]) / dt)
    if window_steps > hold_steps:
        raise ValueError("effective_load_window_exceeds_hold")

    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        raise RuntimeError("effective_load_ready_failed:" + ready_line)
    process.stdin.write("init_default\n")
    process.stdin.flush()
    initial_line = process.stdout.readline().rstrip("\n")
    if not initial_line.startswith("@STATE "):
        raise RuntimeError("effective_load_init_failed:" + initial_line)
    state = json.loads(initial_line[7:])
    if len(state["rocks"]) != 20 or not _finite(state):
        raise RuntimeError("effective_load_initial_contract")
    rock_ids = [rock["rock_id"] for rock in state["rocks"]]
    lift_start_positions = None
    final_window = {rock_id: [] for rock_id in rock_ids}
    phase_end = {}
    maximum_closure = 0.0
    maximum_forbidden = 0
    output_steps = 0
    runtime_error = None
    final_observations = _scene_observations(
        state, plan, inventory, proxy, tolerance, straddle)
    try:
        for phase in phases:
            phase_steps = round(float(phase["duration_s"]) / dt)
            if phase["name"] == "short_lift":
                lift_start_positions = {
                    rock["rock_id"]: list(rock["position"])
                    for rock in state["rocks"]}
            phase_ground = {rock_id: 0 for rock_id in rock_ids}
            phase_bucket = {rock_id: 0 for rock_id in rock_ids}
            for phase_step in range(phase_steps):
                command = [float(value) for value in phase["command"]]
                fields = [
                    "step", f"{float(state['time_s']):.17g}", f"{dt:.17g}",
                    str(substeps), *(f"{value:.17g}" for value in command), "0"]
                process.stdin.write(" ".join(fields) + "\n")
                process.stdin.flush()
                line = process.stdout.readline().rstrip("\n")
                if not line.startswith("@STATE "):
                    stderr = process.stderr.read().strip()
                    runtime_error = line + (":" + stderr if stderr else "")
                    raise RuntimeError("effective_load_step_failed")
                state = json.loads(line[7:])
                if ([rock["rock_id"] for rock in state["rocks"]] != rock_ids
                        or not _finite(state)):
                    runtime_error = "nonfinite_or_identity_changed"
                    raise RuntimeError("effective_load_state_contract")
                observations = _scene_observations(
                    state, plan, inventory, proxy, tolerance, straddle)
                ground_counts = state["rock_ground_contact_substeps_by_rock"]
                bucket_counts = state["rock_bucket_contact_substeps_by_rock"]
                if len(ground_counts) != 20 or len(bucket_counts) != 20:
                    runtime_error = "per_rock_contact_width_changed"
                    raise RuntimeError("effective_load_contact_contract")
                for index, observation in enumerate(observations):
                    rock_id = observation["rock_id"]
                    phase_ground[rock_id] += int(ground_counts[index])
                    phase_bucket[rock_id] += int(bucket_counts[index])
                    if (phase["name"] == "hold_after_lift"
                            and phase_step >= phase_steps - window_steps):
                        final_window[rock_id].append({
                            "ground_contact_substeps": int(ground_counts[index]),
                            "bucket_contact_substeps": int(bucket_counts[index]),
                            "center_bucket_m": observation["center_bucket_m"],
                            "in_relaxed_bucket_neighborhood":
                                _in_relaxed_neighborhood(
                                    observation, inventory,
                                    float(acceptance[
                                        "relaxed_bucket_neighborhood_margin_m"])),
                            "full_envelope_inside": observation[
                                "all_vertices_inside_bucket_prism"],
                        })
                final_observations = observations
                maximum_closure = max(maximum_closure, float(state["closure_m"]))
                maximum_forbidden = max(
                    maximum_forbidden, int(state["forbidden_contact_count"]))
                output_steps += 1
            phase_end[phase["name"]] = {
                "time_s": state["time_s"],
                "bucket_origin_world_m": state["bucket_origin"],
                "observation": _summary(final_observations),
                "ground_contact_substeps_by_rock": phase_ground,
                "bucket_contact_substeps_by_rock": phase_bucket,
            }
    except RuntimeError:
        pass
    finally:
        if process.poll() is None:
            try:
                process.stdin.write("close\n")
                process.stdin.flush()
                tail_out, tail_err = process.communicate(timeout=5)
                if (process.returncode or tail_out.strip() or tail_err.strip()) and not runtime_error:
                    runtime_error = (
                        f"bridge_close_failed:{process.returncode}:"
                        f"{tail_out.strip()}:{tail_err.strip()}")
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                runtime_error = runtime_error or "bridge_close_failed"

    if lift_start_positions is None:
        lift_start_positions = {
            rock_id: [0.0, 0.0, 0.0] for rock_id in rock_ids}
    final_by_id = {item["rock_id"]: item for item in final_observations}
    final_state_by_id = {rock["rock_id"]: rock for rock in state["rocks"]}
    plan_by_slot = {f"r2-rock-{index:03d}": rock
                    for index, rock in enumerate(plan["rocks"])}
    per_rock = []
    carried_ids = []
    for rock_id in rock_ids:
        samples = final_window[rock_id]
        reference = samples[0]["center_bucket_m"] if samples else [0.0] * 3
        maximum_relative_drift = max((
            _norm([sample["center_bucket_m"][axis] - reference[axis]
                   for axis in range(3)]) for sample in samples), default=math.inf)
        vertical_lift = (
            final_state_by_id[rock_id]["position"][2]
            - lift_start_positions[rock_id][2])
        checks = {
            "completed_final_window": len(samples) == window_steps,
            "lifted_from_source_pile": vertical_lift >= float(
                acceptance["minimum_vertical_lift_from_lift_start_m"]),
            "continuously_off_ground": bool(samples) and all(
                sample["ground_contact_substeps"] == 0 for sample in samples),
            "continuously_near_bucket": bool(samples) and all(
                sample["in_relaxed_bucket_neighborhood"] for sample in samples),
            "relative_motion_bounded": maximum_relative_drift <= float(
                acceptance[
                    "maximum_relative_center_drift_during_final_window_m"]),
            "final_speed_bounded": final_by_id[rock_id]["speed_m_s"] <= float(
                acceptance["maximum_final_carried_rock_speed_m_s"]),
        }
        carried = all(checks.values())
        if carried:
            carried_ids.append(rock_id)
        per_rock.append({
            "rock_id": rock_id,
            "source_rock_id": plan_by_slot[rock_id]["rock_id"],
            "carried": carried,
            "checks": checks,
            "vertical_lift_from_lift_start_m": vertical_lift,
            "maximum_relative_center_drift_during_final_window_m":
                maximum_relative_drift,
            "final_speed_m_s": final_by_id[rock_id]["speed_m_s"],
            "solid_volume_m3": plan_by_slot[rock_id]["volume_m3"],
            "mass_kg": plan_by_slot[rock_id]["mass_kg"],
            "final_full_envelope_inside_diagnostic": final_by_id[rock_id][
                "all_vertices_inside_bucket_prism"],
            "final_window_bucket_contact_substeps_diagnostic": sum(
                sample["bucket_contact_substeps"] for sample in samples),
        })
    carried_volume = sum(
        plan_by_slot[rock_id]["volume_m3"] for rock_id in carried_ids)
    carried_mass = sum(plan_by_slot[rock_id]["mass_kg"] for rock_id in carried_ids)
    expected_steps = sum(round(float(phase["duration_s"]) / dt)
                         for phase in phases)
    global_checks = {
        "completed_full_schedule": runtime_error is None and output_steps == expected_steps,
        "twenty_rock_identities_preserved":
            [rock["rock_id"] for rock in state["rocks"]] == rock_ids,
        "minimum_carried_rock_count": len(carried_ids) >= int(
            acceptance["minimum_carried_rock_count"]),
        "closure_within_limit": maximum_closure <= float(
            acceptance["maximum_closure_m"]),
        "no_forbidden_contacts": maximum_forbidden == 0,
        "finite_state": _finite(state),
    }
    return {
        "status": ("effective_carried_load_scene_candidate"
                   if all(global_checks.values())
                   else "effective_carried_load_scene_needs_review"),
        "runtime_error": runtime_error,
        "checks": global_checks,
        "output_steps_completed": output_steps,
        "expected_output_steps": expected_steps,
        "final_window_duration_s": window_steps * dt,
        "carried_rock_ids": carried_ids,
        "carried_rock_count": len(carried_ids),
        "carried_solid_volume_m3": carried_volume,
        "carried_mass_kg": carried_mass,
        "preliminary_solid_volume_fill_rate": (
            carried_volume / float(config["bucket_capacity"]["value_m3"])),
        "bucket_capacity_m3": config["bucket_capacity"]["value_m3"],
        "maximum_closure_m": maximum_closure,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "per_rock": per_rock,
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
        raise ValueError("effective_load_output_already_exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    intake_path = repository / config["source_intake_config"]["path"]
    lift_path = repository / config["source_lift_config"]["path"]
    strict_report_path = repository / config["superseded_strict_lift_report"]["path"]
    for path, expected, label in (
            (intake_path, config["source_intake_config"]["sha256"], "intake"),
            (lift_path, config["source_lift_config"]["sha256"], "lift"),
            (strict_report_path,
             config["superseded_strict_lift_report"]["sha256"], "strict_report")):
        if sha256(path) != expected:
            raise ValueError(f"effective_load_{label}_sha256_mismatch")
    base_config = json.loads(intake_path.read_text(encoding="utf-8"))
    lift_config = json.loads(lift_path.read_text(encoding="utf-8"))
    _validate_intake_config(base_config)
    _validate_lift_config(lift_config)
    strict_report = json.loads(strict_report_path.read_text(encoding="utf-8"))
    if strict_report["status"] != config[
            "superseded_strict_lift_report"]["required_status"]:
        raise ValueError("effective_load_strict_report_status")
    capacity_path = repository / config["bucket_capacity"]["source_path"]
    if sha256(capacity_path) != config["bucket_capacity"]["source_sha256"]:
        raise ValueError("effective_load_capacity_source_sha256_mismatch")
    capacity_source = json.loads(capacity_path.read_text(encoding="utf-8"))
    if abs(float(capacity_source["capacity_liters"]) / 1000
           - float(config["bucket_capacity"]["value_m3"])) > 1e-15:
        raise ValueError("effective_load_capacity_value_mismatch")

    stable_root = repository / base_config["stable_model_root"]["path"]
    stable_report = json.loads((stable_root / "report.json").read_text(encoding="utf-8"))
    stable_index = {item["scene_id"]: item for item in stable_report["scenes"]}
    frozen_bridge = stable_root / "generated-bridge-r2.cc"
    geometry = base_config["bucket_geometry"]
    inventory_path = Path(geometry["inventory_path"])
    proxy_path = repository / geometry["proxy_manifest_path"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    r2_config = json.loads((
        repository / base_config["source_r2_config"]["path"]
    ).read_text(encoding="utf-8"))
    preflight_root = repository / r2_config["assembly_preflight"]["root"]

    output.mkdir(parents=True)
    instrumented_source = output / "generated-bridge-r2-per-rock-contacts.cc"
    instrumented_source.write_text(instrument_per_rock_contacts(
        frozen_bridge.read_text(encoding="utf-8")), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mixed-r2-effective-load-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, instrumented_source, binary,
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
                raise ValueError("effective_load_model_sha256_mismatch:" + scene_id)
            plan_path = preflight_root / scene_id / "assembly-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            metrics = _run_scene(
                binary, model_path, plan, inventory, proxy,
                base_config, lift_config, config)
            scene_report = {
                "schema": "excavation_mixed_material_r2_effective_load_scene_report_v1",
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "status": metrics["status"],
                "metrics": metrics,
                "physics_qualified": False,
                "scope_limit": (
                    "two-scene short-lift diagnostic using carried rigid-rock solid "
                    "volume only; no void compensation, dump, truck, MPM or training"),
            }
            scene_dir = output / scene_id
            scene_dir.mkdir(parents=True)
            scene_path = scene_dir / "report.json"
            write_json(scene_path, scene_report)
            scenes.append({
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "status": metrics["status"],
                "carried_rock_count": metrics["carried_rock_count"],
                "carried_rock_ids": metrics["carried_rock_ids"],
                "carried_solid_volume_m3": metrics["carried_solid_volume_m3"],
                "carried_mass_kg": metrics["carried_mass_kg"],
                "preliminary_solid_volume_fill_rate": metrics[
                    "preliminary_solid_volume_fill_rate"],
                "report_sha256": sha256(scene_path),
            })
    all_candidate = all(
        scene["status"] == "effective_carried_load_scene_candidate"
        for scene in scenes)
    report = {
        "schema": "excavation_mixed_material_r2_effective_load_report_v1",
        "status": ("two_scene_effective_carried_load_candidate"
                   if all_candidate else "effective_carried_load_needs_review"),
        "scene_count": len(scenes),
        "scenes": scenes,
        "bucket_capacity_m3": config["bucket_capacity"]["value_m3"],
        "fill_rate_definition": config["bucket_capacity"]["metric"],
        "void_compensated_bulk_fill_rate_qualified": False,
        "all_six_scene_qualification": False,
        "ten_second_hold_qualified": False,
        "dump_qualified": False,
        "truck_cad_qualified": False,
        "mpm_coupling_qualified": False,
        "training_qualified": False,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "source_intake_config_sha256": sha256(intake_path),
            "source_lift_config_sha256": sha256(lift_path),
            "superseded_strict_lift_report_sha256": sha256(strict_report_path),
            "capacity_source_sha256": sha256(capacity_path),
            "frozen_bridge_sha256": sha256(frozen_bridge),
            "instrumented_bridge_sha256": sha256(instrumented_source),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_effective_load.py"),
            "instrumenter_sha256": sha256(
                repository / "src/v4/instrument_r2_bridge_contacts.py"),
            "compile_command": compile_record,
        },
        "scope_limit": (
            "preliminary carried rigid-rock solid-volume ratio in two scenes; not "
            "visual heap occupancy, void-compensated bulk fill, dump delivery, six-scene "
            "success rate, MPM, hydraulic capacity or training qualification"),
    }
    write_json(output / "report.json", report)
    (output / "README.md").write_text(
        "# R2有效携带与首版实体体积满斗率\n\n"
        f"根状态：`{report['status']}`。满斗率只按稳定携带石头实体体积之和"
        f"除以固定斗容`{report['bucket_capacity_m3']:.15f}m³`计算，不补偿石块"
        "间空隙，也不代表视觉堆高、卸料、车厢、MPM或训练资格。\n",
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
