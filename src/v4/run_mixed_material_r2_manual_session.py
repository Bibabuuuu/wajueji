"""Run one interactive R2 rock-loading session and record a BC candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from instrument_r2_bridge_contacts import instrument_per_rock_contacts
from mixed_material_r2_mechanism import sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite
from run_mixed_material_r2_controlled_intake import (
    _exact_keys,
    _number,
    _scene_observations,
)
from run_mixed_material_r2_effective_load import _in_relaxed_neighborhood


SCHEMA = "excavation_mixed_material_r2_manual_session_config_v1"
SESSION_SCHEMA = "excavation_mixed_material_r2_manual_session_report_v1"
COMMAND_LIMITS = [0.12, 0.0225, 0.0225, 0.0675]


def _validate_config(config: dict) -> None:
    _exact_keys(config, {
        "schema", "source_effective_load_config",
        "source_scripted_baseline_report", "selected_scene", "runtime",
        "manual_control", "markers", "viewer_dependency", "acceptance",
        "scope",
    }, "manual_session_config")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_manual_session_schema")
    _exact_keys(config["source_effective_load_config"], {"path", "sha256"},
                "source_effective_load_config")
    _exact_keys(config["source_scripted_baseline_report"],
                {"path", "sha256", "required_status"},
                "source_scripted_baseline_report")
    if config["source_scripted_baseline_report"]["required_status"] != (
            "single_scene_load_development_candidate"):
        raise ValueError("invalid_scripted_baseline_status")
    if config["selected_scene"] != "ordinary-small-rock-dominant-seed-26091912":
        raise ValueError("manual_session_must_remain_on_selected_ordinary_scene")

    runtime = config["runtime"]
    _exact_keys(runtime, {
        "output_dt_s", "internal_substeps_per_output",
        "maximum_simulated_duration_s", "maximum_wall_duration_s",
        "full_state_log_hz",
    }, "runtime")
    dt = _number(runtime["output_dt_s"], "output_dt_s", positive=True)
    if not math.isclose(dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("manual_session_dt_changed")
    if runtime["internal_substeps_per_output"] != 20:
        raise ValueError("manual_session_substeps_changed")
    for key in ("maximum_simulated_duration_s", "maximum_wall_duration_s"):
        _number(runtime[key], key, positive=True)
    if runtime["full_state_log_hz"] != 20:
        raise ValueError("manual_session_log_rate_changed")

    control = config["manual_control"]
    _exact_keys(control, {
        "axis_order", "positive_keys", "negative_keys", "maximum_command",
        "default_speed_scale", "speed_scales", "release_means_zero_hold",
        "focus_loss_pauses",
    }, "manual_control")
    if (control["axis_order"] != ["rotation", "boom", "arm", "bucket"]
            or control["positive_keys"] != ["J", "I", "U", "M"]
            or control["negative_keys"] != ["L", "K", "O", "N"]
            or control["maximum_command"] != COMMAND_LIMITS
            or control["default_speed_scale"] != 0.5
            or control["speed_scales"] != [0.25, 0.5, 1.0]
            or control["release_means_zero_hold"] is not True
            or control["focus_loss_pauses"] is not True):
        raise ValueError("manual_control_contract_changed")

    markers = config["markers"]
    _exact_keys(markers, {"F1", "F2", "F3", "F4", "required_for_scoring"},
                "markers")
    if (markers["F3"] != "lift_start"
            or markers["F4"] != "evaluation_hold_start"
            or markers["required_for_scoring"] !=
            ["lift_start", "evaluation_hold_start"]):
        raise ValueError("manual_marker_contract_changed")
    for label in ("glfw_header", "glfw_library"):
        _exact_keys(config["viewer_dependency"][label], {"path", "sha256"},
                    "viewer_dependency/" + label)

    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_carried_rock_count",
        "minimum_vertical_lift_from_lift_start_m",
        "minimum_evaluation_hold_duration_s",
        "relaxed_bucket_neighborhood_margin_m",
        "maximum_relative_center_drift_during_final_window_m",
        "maximum_final_carried_rock_speed_m_s", "maximum_closure_m",
        "required_forbidden_contact_count",
        "require_zero_command_during_final_window",
        "require_fill_rate_above_scripted_baseline",
        "scripted_baseline_fill_rate",
    }, "acceptance")
    if (acceptance["minimum_carried_rock_count"] != 3
            or acceptance["required_forbidden_contact_count"] != 0
            or acceptance["require_zero_command_during_final_window"] is not True
            or acceptance["require_fill_rate_above_scripted_baseline"] is not True):
        raise ValueError("manual_acceptance_contract_changed")
    for key in (
            "minimum_vertical_lift_from_lift_start_m",
            "minimum_evaluation_hold_duration_s",
            "relaxed_bucket_neighborhood_margin_m",
            "maximum_relative_center_drift_during_final_window_m",
            "maximum_final_carried_rock_speed_m_s", "maximum_closure_m",
            "scripted_baseline_fill_rate"):
        _number(acceptance[key], key, positive=True)

    scope = config["scope"]
    _exact_keys(scope, {
        "one_scene_manual_feasibility_and_demonstration_capture",
        "scripted_trajectory_optimization", "second_scene_confirmation",
        "all_six_scene_qualification", "dump", "truck_cad", "mpm_coupling",
        "training", "physics_qualified",
    }, "scope")
    if (scope["one_scene_manual_feasibility_and_demonstration_capture"] is not True
            or any(scope[key] is not False for key in scope
                   if key != "one_scene_manual_feasibility_and_demonstration_capture")):
        raise ValueError("manual_session_scope_expanded")


def _compile_viewer(cxx: str, source: Path, output: Path, mujoco_include: Path,
                    mujoco_library_dir: Path, glfw_header: Path,
                    glfw_library: Path) -> list[str]:
    command = [
        cxx, "-O2", "-std=c++17", "-Wall", "-Wextra", "-pthread",
        "-I" + str(mujoco_include.resolve()),
        "-I" + str(glfw_header.resolve().parent), str(source.resolve()),
        "-L" + str(mujoco_library_dir.resolve()),
        "-L" + str(glfw_library.resolve().parent),
        "-Wl,-rpath," + str(mujoco_library_dir.resolve()),
        "-Wl,-rpath," + str(glfw_library.resolve().parent),
        "-l:libmujoco.so.2.3.8", "-l:" + glfw_library.name,
        "-o", str(output.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True,
                               check=False)
    if completed.returncode:
        raise RuntimeError("manual_viewer_compile_failed:" + completed.stderr)
    return command


def _qpos(state: dict) -> list[float]:
    return list(state["mechanism_qpos"]) + [
        value for rock in state["rocks"] for value in rock["qpos"]]


def _viewer_state_line(state: dict, observations: list[dict], inventory: dict,
                       plan_by_slot: dict, margin: float,
                       bucket_capacity_m3: float) -> str:
    ground = state["rock_ground_contact_substeps_by_rock"]
    current_ids = [
        observation["rock_id"]
        for index, observation in enumerate(observations)
        if int(ground[index]) == 0
        and _in_relaxed_neighborhood(observation, inventory, margin)
    ]
    volume = sum(float(plan_by_slot[rock_id]["volume_m3"])
                 for rock_id in current_ids)
    values = _qpos(state)
    return " ".join([
        "state", f"{float(state['time_s']):.17g}",
        f"{float(state['closure_m']):.17g}",
        str(int(state["forbidden_contact_count"])), str(len(current_ids)),
        f"{100.0 * volume / bucket_capacity_m3:.17g}", str(len(values)),
        *(f"{float(value):.17g}" for value in values),
    ]) + "\n"


def _step_bridge(process: subprocess.Popen, state: dict, dt: float,
                 substeps: int, action: list[float]) -> dict:
    fields = [
        "step", f"{float(state['time_s']):.17g}", f"{dt:.17g}",
        str(substeps), *(f"{float(value):.17g}" for value in action), "0",
    ]
    process.stdin.write(" ".join(fields) + "\n")
    process.stdin.flush()
    line = process.stdout.readline().rstrip("\n")
    if not line.startswith("@STATE "):
        raise RuntimeError("manual_bridge_step_failed:" + line)
    result = json.loads(line[7:])
    if not _finite(result):
        raise RuntimeError("manual_bridge_nonfinite_state")
    return result


def _measurement(state: dict, observations: list[dict], action: list[float],
                 sequence: int) -> dict:
    ground = state["rock_ground_contact_substeps_by_rock"]
    bucket = state["rock_bucket_contact_substeps_by_rock"]
    return {
        "sequence": sequence,
        "time_s": state["time_s"],
        "action_command": action,
        "bucket_origin_world_m": state["bucket_origin"],
        "bucket_rotation_world_to_bucket": state["bucket_rotation"],
        "closure_m": state["closure_m"],
        "forbidden_contact_count": state["forbidden_contact_count"],
        "rocks": [{
            "rock_id": observation["rock_id"],
            "position_world_m": state["rocks"][index]["position"],
            "center_bucket_m": observation["center_bucket_m"],
            "speed_m_s": observation["speed_m_s"],
            "ground_contact_substeps": int(ground[index]),
            "bucket_contact_substeps": int(bucket[index]),
            "full_envelope_inside_diagnostic": observation[
                "all_vertices_inside_bucket_prism"],
        } for index, observation in enumerate(observations)],
    }


def _demo_sample(state: dict, action: list[float], sequence: int,
                 markers_seen: list[str]) -> dict:
    return {
        "schema": "excavation_mixed_material_r2_manual_bc_sample_v1",
        "sequence": sequence,
        "time_s": state["time_s"],
        "axis_order": ["rotation", "boom", "arm", "bucket"],
        "action_command": action,
        "mechanism_qpos": state["mechanism_qpos"],
        "mechanism_qvel": state["mechanism_qvel"],
        "bucket_origin_world_m": state["bucket_origin"],
        "bucket_rotation_world_to_bucket": state["bucket_rotation"],
        "rocks": [{
            "rock_id": rock["rock_id"], "qpos": rock["qpos"],
            "qvel": rock["qvel"],
        } for rock in state["rocks"]],
        "markers_seen": markers_seen,
        "training_eligible": False,
    }


def _score(samples: list[dict], lift_start_positions: dict | None,
           evaluation_time_s: float | None, plan: dict, inventory: dict,
           acceptance: dict, bucket_capacity_m3: float,
           maximum_closure_m: float, maximum_forbidden: int,
           runtime_error: str | None) -> dict:
    dt = 1.0 / 120.0
    window_steps = math.ceil(
        float(acceptance["minimum_evaluation_hold_duration_s"]) / dt - 1e-12)
    after_marker = ([sample for sample in samples
                     if evaluation_time_s is not None
                     and sample["time_s"] > evaluation_time_s + 1e-12])
    window = after_marker[-window_steps:]
    completed_window = len(window) == window_steps
    final = samples[-1] if samples else None
    plan_by_slot = {f"r2-rock-{index:03d}": rock
                    for index, rock in enumerate(plan["rocks"])}
    per_rock = []
    carried_ids = []
    for index in range(20):
        rock_id = f"r2-rock-{index:03d}"
        rock_samples = [sample["rocks"][index] for sample in window]
        reference = (rock_samples[0]["center_bucket_m"]
                     if rock_samples else [0.0, 0.0, 0.0])
        drift = max((math.sqrt(sum(
            (rock["center_bucket_m"][axis] - reference[axis]) ** 2
            for axis in range(3))) for rock in rock_samples), default=math.inf)
        final_rock = final["rocks"][index] if final else None
        lift = (final_rock["position_world_m"][2]
                - lift_start_positions[rock_id][2]
                if final_rock and lift_start_positions else -math.inf)
        checks = {
            "completed_final_window": completed_window,
            "lifted_from_source_pile": lift >= float(acceptance[
                "minimum_vertical_lift_from_lift_start_m"]),
            "continuously_off_ground": bool(rock_samples) and all(
                rock["ground_contact_substeps"] == 0 for rock in rock_samples),
            "continuously_near_bucket": bool(rock_samples) and all(
                _in_relaxed_neighborhood({"center_bucket_m":
                    rock["center_bucket_m"]}, inventory, float(acceptance[
                        "relaxed_bucket_neighborhood_margin_m"]))
                for rock in rock_samples),
            "relative_motion_bounded": drift <= float(acceptance[
                "maximum_relative_center_drift_during_final_window_m"]),
            "final_speed_bounded": bool(final_rock) and final_rock["speed_m_s"]
                <= float(acceptance["maximum_final_carried_rock_speed_m_s"]),
        }
        carried = all(checks.values())
        if carried:
            carried_ids.append(rock_id)
        per_rock.append({
            "rock_id": rock_id,
            "source_rock_id": plan_by_slot[rock_id]["rock_id"],
            "carried": carried,
            "checks": checks,
            "vertical_lift_from_lift_start_m": (
                lift if math.isfinite(lift) else None),
            "maximum_relative_center_drift_during_final_window_m": (
                drift if math.isfinite(drift) else None),
            "final_speed_m_s": final_rock["speed_m_s"] if final_rock else None,
            "solid_volume_m3": plan_by_slot[rock_id]["volume_m3"],
            "mass_kg": plan_by_slot[rock_id]["mass_kg"],
            "final_full_envelope_inside_diagnostic": (
                final_rock["full_envelope_inside_diagnostic"]
                if final_rock else False),
        })
    volume = sum(float(plan_by_slot[rock_id]["volume_m3"])
                 for rock_id in carried_ids)
    mass = sum(float(plan_by_slot[rock_id]["mass_kg"])
               for rock_id in carried_ids)
    fill_rate = volume / bucket_capacity_m3
    zero_command = completed_window and all(
        all(abs(float(value)) <= 1e-15 for value in sample["action_command"])
        for sample in window)
    checks = {
        "runtime_completed_without_error": runtime_error is None,
        "lift_start_marked": lift_start_positions is not None,
        "evaluation_hold_marked": evaluation_time_s is not None,
        "completed_evaluation_window": completed_window,
        "zero_command_during_final_window": zero_command,
        "minimum_carried_rock_count": len(carried_ids) >= int(
            acceptance["minimum_carried_rock_count"]),
        "fill_rate_above_scripted_baseline": fill_rate > float(
            acceptance["scripted_baseline_fill_rate"]),
        "closure_within_limit": maximum_closure_m <= float(
            acceptance["maximum_closure_m"]),
        "no_forbidden_contacts": maximum_forbidden == int(
            acceptance["required_forbidden_contact_count"]),
    }
    return {
        "checks": checks,
        "candidate": all(checks.values()),
        "evaluation_window_sample_count": len(window),
        "evaluation_window_duration_s": len(window) * dt,
        "carried_rock_ids": carried_ids,
        "carried_rock_count": len(carried_ids),
        "carried_solid_volume_m3": volume,
        "carried_mass_kg": mass,
        "preliminary_solid_volume_fill_rate": fill_rate,
        "bucket_capacity_m3": bucket_capacity_m3,
        "maximum_closure_m": maximum_closure_m,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "per_rock": per_rock,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path, *, smoke: bool = False) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("manual_session_output_already_exists")
    config = _read_json(config_path)
    _validate_config(config)

    effective_path = repository / config["source_effective_load_config"]["path"]
    baseline_path = repository / config["source_scripted_baseline_report"]["path"]
    for path, expected, label in (
            (effective_path, config["source_effective_load_config"]["sha256"],
             "effective_config"),
            (baseline_path,
             config["source_scripted_baseline_report"]["sha256"],
             "scripted_baseline")):
        if sha256(path) != expected:
            raise ValueError("manual_session_" + label + "_sha256_mismatch")
    effective = _read_json(effective_path)
    intake_path = repository / effective["source_intake_config"]["path"]
    if sha256(intake_path) != effective["source_intake_config"]["sha256"]:
        raise ValueError("manual_session_intake_config_sha256_mismatch")
    intake = _read_json(intake_path)
    baseline = _read_json(baseline_path)
    if baseline["status"] != config[
            "source_scripted_baseline_report"]["required_status"]:
        raise ValueError("manual_session_scripted_baseline_status")
    if not math.isclose(float(baseline[
            "candidate_preliminary_solid_volume_fill_rate"]), float(config[
                "acceptance"]["scripted_baseline_fill_rate"]),
            rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("manual_session_scripted_baseline_value")

    glfw_header = Path(config["viewer_dependency"]["glfw_header"]["path"])
    glfw_library = Path(config["viewer_dependency"]["glfw_library"]["path"])
    for path, expected, label in (
            (glfw_header, config["viewer_dependency"]["glfw_header"]["sha256"],
             "glfw_header"),
            (glfw_library,
             config["viewer_dependency"]["glfw_library"]["sha256"],
             "glfw_library")):
        if sha256(path) != expected:
            raise ValueError("manual_session_" + label + "_sha256_mismatch")

    stable_root = repository / intake["stable_model_root"]["path"]
    stable_report_path = stable_root / "report.json"
    if sha256(stable_report_path) != intake["stable_model_root"]["report_sha256"]:
        raise ValueError("manual_session_stable_report_sha256_mismatch")
    stable_report = _read_json(stable_report_path)
    stable_index = {item["scene_id"]: item for item in stable_report["scenes"]}
    scene_id = config["selected_scene"]
    model_path = stable_root / scene_id / "model.xml"
    if sha256(model_path) != stable_index[scene_id]["model_sha256"]:
        raise ValueError("manual_session_model_sha256_mismatch")
    frozen_bridge = stable_root / "generated-bridge-r2.cc"
    if sha256(frozen_bridge) != intake[
            "stable_model_root"]["generated_bridge_source_sha256"]:
        raise ValueError("manual_session_bridge_sha256_mismatch")

    geometry = intake["bucket_geometry"]
    inventory_path = Path(geometry["inventory_path"])
    proxy_path = repository / geometry["proxy_manifest_path"]
    for path, expected, label in (
            (inventory_path, geometry["inventory_sha256"], "inventory"),
            (proxy_path, geometry["proxy_manifest_sha256"], "proxy")):
        if sha256(path) != expected:
            raise ValueError("manual_session_" + label + "_sha256_mismatch")
    inventory = _read_json(inventory_path)
    proxy = _read_json(proxy_path)
    r2_config_path = repository / intake["source_r2_config"]["path"]
    if sha256(r2_config_path) != intake["source_r2_config"]["sha256"]:
        raise ValueError("manual_session_r2_config_sha256_mismatch")
    r2_config = _read_json(r2_config_path)
    plan_path = (repository / r2_config["assembly_preflight"]["root"] /
                 scene_id / "assembly-plan.json")
    plan = _read_json(plan_path)
    plan_by_slot = {f"r2-rock-{index:03d}": rock
                    for index, rock in enumerate(plan["rocks"])}
    bucket_capacity = float(effective["bucket_capacity"]["value_m3"])

    output.mkdir(parents=True)
    generated_bridge = output / "generated-bridge-r2-per-rock-contacts.cc"
    generated_bridge.write_text(instrument_per_rock_contacts(
        frozen_bridge.read_text(encoding="utf-8")), encoding="utf-8")
    viewer_source = repository / "src/v4/mixed_material_r2_manual_viewer.cc"
    dt = float(config["runtime"]["output_dt_s"])
    substeps = int(config["runtime"]["internal_substeps_per_output"])
    tolerance = float(intake["acceptance"]["full_envelope_tolerance_m"])
    straddle = float(intake["acceptance"]["mouth_straddle_depth_margin_m"])
    margin = float(config["acceptance"][
        "relaxed_bucket_neighborhood_margin_m"])
    runtime_error = None
    close_reason = None
    events = []
    markers = {}
    marker_order = []
    samples = []
    maximum_closure = 0.0
    maximum_forbidden = 0
    output_steps = 0
    demo_samples = 0
    bridge = None
    viewer = None
    bridge_compile = []
    viewer_compile = []
    viewer_stderr = ""
    bridge_stderr = ""
    started = time.monotonic()

    measurements_path = output / "measurements.ndjson"
    demo_path = output / "manual-demo.ndjson"
    events_path = output / "events.ndjson"
    with tempfile.TemporaryDirectory(prefix="mixed-r2-manual-session-") as temporary:
        temporary_path = Path(temporary)
        bridge_binary = temporary_path / "mixed-rigid-bridge-r2"
        viewer_binary = temporary_path / "mixed-r2-manual-viewer"
        bridge_compile = _compile(
            cxx, generated_bridge, bridge_binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        viewer_compile = _compile_viewer(
            cxx, viewer_source, viewer_binary, Path(mujoco_include),
            Path(mujoco_library_dir), glfw_header, glfw_library)
        bridge = subprocess.Popen(
            [str(bridge_binary), str(model_path)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        ready = bridge.stdout.readline().rstrip("\n")
        if not ready.startswith("@READY "):
            raise RuntimeError("manual_bridge_ready_failed:" + ready)
        bridge.stdin.write("init_default\n")
        bridge.stdin.flush()
        initial = bridge.stdout.readline().rstrip("\n")
        if not initial.startswith("@STATE "):
            raise RuntimeError("manual_bridge_init_failed:" + initial)
        state = json.loads(initial[7:])
        rock_ids = [rock["rock_id"] for rock in state["rocks"]]
        if (rock_ids != [f"r2-rock-{index:03d}" for index in range(20)]
                or not _finite(state)):
            raise RuntimeError("manual_bridge_initial_state_contract")
        viewer_command = [str(viewer_binary), str(model_path),
                          str(output / "viewer")]
        if smoke:
            viewer_command.append("--smoke")
        environment = os.environ.copy()
        environment.setdefault("DISPLAY", ":1")
        viewer = subprocess.Popen(
            viewer_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=environment)
        viewer_ready = viewer.stdout.readline().rstrip("\n")
        if viewer_ready != "@READY 159":
            raise RuntimeError("manual_viewer_ready_failed:" + viewer_ready)
        initial_observations = _scene_observations(
            state, plan, inventory, proxy, tolerance, straddle)
        viewer.stdin.write(_viewer_state_line(
            state, initial_observations, inventory, plan_by_slot, margin,
            bucket_capacity))
        viewer.stdin.flush()

        with (measurements_path.open("w", encoding="utf-8") as measurements,
              demo_path.open("w", encoding="utf-8") as demonstrations,
              events_path.open("w", encoding="utf-8") as event_file):
            try:
                while (time.monotonic() - started < float(config["runtime"]
                       ["maximum_wall_duration_s"])
                       and float(state["time_s"]) < float(config["runtime"]
                       ["maximum_simulated_duration_s"])):
                    line = viewer.stdout.readline()
                    if not line:
                        close_reason = "viewer_pipe_closed"
                        break
                    line = line.rstrip("\n")
                    if line.startswith("@EVENT "):
                        _, name, wall, sim_time = line.split()
                        record = {"kind": "event", "name": name,
                                  "viewer_wall_time_s": float(wall),
                                  "time_s": float(sim_time)}
                        events.append(record)
                        event_file.write(json.dumps(record, sort_keys=True) + "\n")
                        continue
                    if line.startswith("@MARK "):
                        _, name, wall, sim_time = line.split()
                        record = {"kind": "marker", "name": name,
                                  "viewer_wall_time_s": float(wall),
                                  "time_s": float(state["time_s"])}
                        events.append(record)
                        event_file.write(json.dumps(record, sort_keys=True) + "\n")
                        if name not in markers:
                            markers[name] = record
                            marker_order.append(name)
                            if name == "lift_start":
                                markers[name]["rock_positions_world_m"] = {
                                    rock["rock_id"]: list(rock["position"])
                                    for rock in state["rocks"]}
                        continue
                    if line.startswith("@CLOSE "):
                        close_reason = line.split()[1]
                        break
                    if not line.startswith("@ACTION "):
                        raise RuntimeError("manual_viewer_protocol:" + line)
                    fields = line.split()
                    if len(fields) != 8:
                        raise RuntimeError("manual_viewer_action_width")
                    paused = fields[2] == "1"
                    action = [float(value) for value in fields[4:8]]
                    if len(action) != 4 or any(
                            abs(action[index]) > COMMAND_LIMITS[index] + 1e-12
                            for index in range(4)):
                        raise RuntimeError("manual_viewer_action_limit")
                    if paused:
                        observations = _scene_observations(
                            state, plan, inventory, proxy, tolerance, straddle)
                        viewer.stdin.write(_viewer_state_line(
                            state, observations, inventory, plan_by_slot,
                            margin, bucket_capacity))
                        viewer.stdin.flush()
                        continue
                    for _ in range(2):
                        state = _step_bridge(bridge, state, dt, substeps, action)
                        if [rock["rock_id"] for rock in state["rocks"]] != rock_ids:
                            raise RuntimeError("manual_rock_identity_changed")
                        observations = _scene_observations(
                            state, plan, inventory, proxy, tolerance, straddle)
                        record = _measurement(
                            state, observations, action, output_steps)
                        samples.append(record)
                        measurements.write(json.dumps(
                            record, sort_keys=True, separators=(",", ":")) + "\n")
                        output_steps += 1
                        maximum_closure = max(
                            maximum_closure, float(state["closure_m"]))
                        maximum_forbidden = max(
                            maximum_forbidden,
                            int(state["forbidden_contact_count"]))
                        if output_steps % 6 == 0:
                            demo = _demo_sample(
                                state, action, demo_samples, list(marker_order))
                            demonstrations.write(json.dumps(
                                demo, sort_keys=True,
                                separators=(",", ":")) + "\n")
                            demo_samples += 1
                    viewer.stdin.write(_viewer_state_line(
                        state, observations, inventory, plan_by_slot, margin,
                        bucket_capacity))
                    viewer.stdin.flush()
            except BrokenPipeError:
                close_reason = close_reason or "viewer_pipe_closed"
            except (RuntimeError, ValueError) as error:
                runtime_error = str(error)
            finally:
                measurements.flush()
                demonstrations.flush()
                event_file.flush()

        if bridge.poll() is None:
            try:
                bridge.stdin.write("close\n")
                bridge.stdin.flush()
                tail, bridge_stderr = bridge.communicate(timeout=5)
                if bridge.returncode or tail.strip() or bridge_stderr.strip():
                    runtime_error = runtime_error or (
                        f"manual_bridge_close_failed:{bridge.returncode}:"
                        f"{tail.strip()}:{bridge_stderr.strip()}")
            except (BrokenPipeError, subprocess.TimeoutExpired):
                bridge.kill()
                runtime_error = runtime_error or "manual_bridge_close_failed"
        if viewer.poll() is None:
            try:
                viewer.stdin.close()
                viewer.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                viewer.kill()
                viewer.wait()
        if viewer.stderr:
            viewer_stderr = viewer.stderr.read().strip()
        if viewer.returncode:
            runtime_error = runtime_error or (
                f"manual_viewer_exit:{viewer.returncode}:{viewer_stderr}")

    lift_marker = markers.get("lift_start")
    evaluation_marker = markers.get("evaluation_hold_start")
    metrics = _score(
        samples,
        lift_marker.get("rock_positions_world_m") if lift_marker else None,
        evaluation_marker["time_s"] if evaluation_marker else None,
        plan, inventory, config["acceptance"], bucket_capacity,
        maximum_closure, maximum_forbidden, runtime_error)
    status = ("manual_session_smoke_completed" if smoke and runtime_error is None
              else "manual_demonstration_candidate" if metrics["candidate"]
              else "manual_session_recorded_needs_review")
    normalize = lambda command, binary: [
        "<temporary>/" + binary if value.endswith("/" + binary) else value
        for value in command]
    report = {
        "schema": SESSION_SCHEMA,
        "status": status,
        "scene_id": scene_id,
        "smoke_mode": smoke,
        "runtime_error": runtime_error,
        "close_reason": close_reason,
        "output_steps": output_steps,
        "recorded_duration_s": float(samples[-1]["time_s"]) if samples else 0.0,
        "demonstration_sample_count": demo_samples,
        "demonstration_sample_rate_hz": config["runtime"]["full_state_log_hz"],
        "demonstration_training_eligible": False,
        "markers": markers,
        "metrics": metrics,
        "scripted_baseline": {
            "carried_rock_count": baseline["candidate_carried_rock_count"],
            "preliminary_solid_volume_fill_rate": baseline[
                "candidate_preliminary_solid_volume_fill_rate"],
        },
        "manual_feasibility_demonstrated": metrics["candidate"],
        "training_qualified": False,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "effective_load_config_sha256": sha256(effective_path),
            "scripted_baseline_report_sha256": sha256(baseline_path),
            "stable_report_sha256": sha256(stable_report_path),
            "model_sha256": sha256(model_path),
            "assembly_plan_sha256": sha256(plan_path),
            "frozen_bridge_sha256": sha256(frozen_bridge),
            "instrumented_bridge_sha256": sha256(generated_bridge),
            "viewer_source_sha256": sha256(viewer_source),
        },
        "implementation": {
            "runner_sha256": sha256(repository /
                "src/v4/run_mixed_material_r2_manual_session.py"),
            "bridge_compile_command": normalize(
                bridge_compile, "mixed-rigid-bridge-r2"),
            "viewer_compile_command": normalize(
                viewer_compile, "mixed-r2-manual-viewer"),
            "viewer_stderr": viewer_stderr,
            "bridge_stderr": bridge_stderr,
        },
        "artifacts": {
            "measurements_ndjson": measurements_path.name,
            "manual_demo_ndjson": demo_path.name,
            "events_ndjson": events_path.name,
            "viewer_directory": "viewer",
        },
        "scope_limit": (
            "one-scene human-control feasibility and synchronized recording only; "
            "not a trained policy, physics qualification, second-scene confirmation, "
            "six-scene success rate, dump delivery, truck or MPM qualification"),
    }
    write_json(output / "report.json", report)
    (output / "README.md").write_text(
        "# R2 单场景手动装载会话\n\n"
        f"状态：`{status}`；记录 {demo_samples} 个 20 Hz 状态—动作样本。"
        "原始记录一律保持 `training_eligible=false`，必须先通过轨迹质量审查。\n\n"
        f"有效携带：{metrics['carried_rock_count']} 块；首版实体体积满斗率："
        f"{100.0 * metrics['preliminary_solid_volume_fill_rate']:.4f}%。\n",
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(
        args.config, args.output_root, args.repository_root, args.cxx,
        args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir,
        smoke=args.smoke)
    print(json.dumps({
        "status": report["status"],
        "output_steps": report["output_steps"],
        "demonstration_sample_count": report["demonstration_sample_count"],
        "manual_feasibility_demonstrated": report[
            "manual_feasibility_demonstrated"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
