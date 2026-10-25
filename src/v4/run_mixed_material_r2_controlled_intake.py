"""Run the first bounded two-scene controlled multi-rock intake diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from mixed_material_r0 import normalized_prototypes
from mixed_material_r2_mechanism import sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite


SCHEMA = "excavation_mixed_material_r2_controlled_intake_config_v1"
COMMAND_LIMITS = [0.12, 0.0225, 0.0225, 0.0675]


def _exact_keys(value: dict, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("expected_object:" + label)
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError("missing_field:" + label + ":" + sorted(missing)[0])
    if extra:
        raise ValueError("unexpected_field:" + label + ":" + sorted(extra)[0])


def _number(value, label: str, *, positive: bool = False) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value))):
        raise ValueError("invalid_number:" + label)
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError("nonpositive_number:" + label)
    return result


def _validate_config(config: dict) -> None:
    expected = {
        "schema", "source_r2_config", "stable_model_root", "bucket_geometry",
        "selected_scenes", "selection_policy", "integration", "phases",
        "acceptance", "scope",
    }
    _exact_keys(config, expected, "config")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_controlled_intake_schema")
    for label, value, keys in (
        ("source_r2_config", config["source_r2_config"], {"path", "sha256"}),
        ("stable_model_root", config["stable_model_root"],
         {"path", "report_sha256", "generated_bridge_source_sha256"}),
        ("bucket_geometry", config["bucket_geometry"],
         {"inventory_path", "inventory_sha256", "proxy_manifest_path",
          "proxy_manifest_sha256"}),
    ):
        _exact_keys(value, keys, label)
        for key in keys:
            if not isinstance(value[key], str) or not value[key]:
                raise ValueError("invalid_string:" + label + ":" + key)
    scenes = config["selected_scenes"]
    if (not isinstance(scenes, list) or len(scenes) != 2
            or len(set(scenes)) != 2
            or not any(scene.startswith("ordinary-small-rock-dominant-")
                       for scene in scenes)
            or not any(scene.startswith("sparse-large-rock-") for scene in scenes)):
        raise ValueError("selected_scenes_must_be_one_ordinary_one_sparse")
    if config["selection_policy"] != (
            "one_ordinary_and_one_sparse_scene_with_stable_rock_centers_in_"
            "bucket_width_approach_corridor"):
        raise ValueError("invalid_selection_policy")
    integration = config["integration"]
    _exact_keys(integration, {"output_dt_s", "internal_substeps_per_output",
                              "external_wrench_count"}, "integration")
    dt = _number(integration["output_dt_s"], "output_dt_s", positive=True)
    if dt > 1.0 / 120.0 + 1e-12:
        raise ValueError("output_dt_above_bridge_limit")
    if (not isinstance(integration["internal_substeps_per_output"], int)
            or integration["internal_substeps_per_output"] < 20
            or integration["internal_substeps_per_output"] > 160):
        raise ValueError("invalid_internal_substeps")
    if integration["external_wrench_count"] != 0:
        raise ValueError("external_wrench_not_zero")
    phases = config["phases"]
    if not isinstance(phases, list) or not phases:
        raise ValueError("missing_phases")
    names = []
    for index, phase in enumerate(phases):
        _exact_keys(phase, {"name", "duration_s", "command"},
                    f"phases/{index}")
        if not isinstance(phase["name"], str) or not phase["name"]:
            raise ValueError("invalid_phase_name")
        names.append(phase["name"])
        duration = _number(phase["duration_s"], "phase_duration", positive=True)
        steps = round(duration / dt)
        if not math.isclose(steps * dt, duration, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("phase_duration_not_integral")
        command = phase["command"]
        if not isinstance(command, list) or len(command) != 4:
            raise ValueError("invalid_phase_command_width")
        for axis, value in enumerate(command):
            number = _number(value, "phase_command")
            if abs(number) > COMMAND_LIMITS[axis] + 1e-12:
                raise ValueError("phase_command_above_bridge_limit")
    if len(names) != len(set(names)):
        raise ValueError("duplicate_phase_name")
    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_bucket_contact_substeps",
        "minimum_final_retained_full_rock_count",
        "minimum_final_continuous_full_inside_duration_s",
        "maximum_final_retained_rock_speed_m_s", "maximum_closure_m",
        "required_forbidden_contact_count", "full_envelope_tolerance_m",
        "mouth_straddle_depth_margin_m",
        "force_and_torque_are_diagnostic_not_capacity_thresholds",
    }, "acceptance")
    for key in ("minimum_bucket_contact_substeps",
                "minimum_final_retained_full_rock_count",
                "required_forbidden_contact_count"):
        if not isinstance(acceptance[key], int) or acceptance[key] < 0:
            raise ValueError("invalid_acceptance_integer:" + key)
    for key in ("minimum_final_continuous_full_inside_duration_s",
                "maximum_final_retained_rock_speed_m_s", "maximum_closure_m",
                "full_envelope_tolerance_m", "mouth_straddle_depth_margin_m"):
        _number(acceptance[key], key, positive=True)
    if (acceptance["required_forbidden_contact_count"] != 0
            or acceptance["force_and_torque_are_diagnostic_not_capacity_thresholds"]
            is not True):
        raise ValueError("unsafe_acceptance_scope")
    scope = config["scope"]
    _exact_keys(scope, {
        "controlled_two_scene_intake_diagnostic", "all_six_scene_qualification",
        "lift_hold", "dump", "mpm_coupling", "fill_rate", "training",
        "physics_qualified",
    }, "scope")
    if (scope["controlled_two_scene_intake_diagnostic"] is not True
            or any(scope[key] is not False for key in (
                "all_six_scene_qualification", "lift_hold", "dump",
                "mpm_coupling", "fill_rate", "training", "physics_qualified"))):
        raise ValueError("expanded_controlled_intake_scope")


def _point_on_segment(point, first, second, tolerance=1e-12) -> bool:
    cross = ((point[0] - first[0]) * (second[1] - first[1])
             - (point[1] - first[1]) * (second[0] - first[0]))
    if abs(cross) > tolerance:
        return False
    return (min(first[0], second[0]) - tolerance <= point[0]
            <= max(first[0], second[0]) + tolerance
            and min(first[1], second[1]) - tolerance <= point[1]
            <= max(first[1], second[1]) + tolerance)


def _point_in_polygon(point, polygon) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if ((current[1] > point[1]) != (previous[1] > point[1])):
            intersection_x = (previous[0]
                              + (point[1] - previous[1])
                              * (current[0] - previous[0])
                              / (current[1] - previous[1]))
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def _segment_distance(point, first, second) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-24:
        return math.hypot(point[0] - first[0], point[1] - first[1])
    projection = max(0.0, min(1.0,
        ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
        / denominator))
    return math.hypot(point[0] - (first[0] + projection * dx),
                      point[1] - (first[1] + projection * dy))


def _point_in_polygon_with_tolerance(point, polygon, tolerance: float) -> bool:
    return (_point_in_polygon(point, polygon)
            or min(_segment_distance(point, polygon[index - 1], polygon[index])
                   for index in range(len(polygon))) <= tolerance)


def _rotate_quaternion_wxyz(quaternion, vector) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _world_to_bucket(world, state: dict) -> list[float]:
    delta = [world[axis] - state["bucket_origin"][axis] for axis in range(3)]
    rotation = state["bucket_rotation"]
    return [sum(rotation[3 * row + axis] * delta[row] for row in range(3))
            for axis in range(3)]


def _norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _rock_observation(state: dict, rock_state: dict, rock_plan: dict,
                      prototypes: dict, bucket: dict, proxy: dict,
                      tolerance: float, straddle_margin: float) -> dict:
    prototype = prototypes[rock_plan["prototype_id"]]
    scale = float(rock_plan["equivalent_diameter_m"])
    position = rock_state["qpos"][:3]
    quaternion = rock_state["qpos"][3:7]
    vertices_world = []
    for source in prototype["vertices"]:
        rotated = _rotate_quaternion_wxyz(
            quaternion, [float(value) * scale for value in source])
        vertices_world.append(
            [position[axis] + rotated[axis] for axis in range(3)])
    vertices_bucket = [_world_to_bucket(vertex, state) for vertex in vertices_world]
    center_bucket = _world_to_bucket(rock_state["position"], state)
    polygon = bucket["polygon_xz_m"]
    y_min, y_max = bucket["y_limits_m"]
    center_inside = (
        _point_in_polygon_with_tolerance(
            [center_bucket[0], center_bucket[2]], polygon, tolerance)
        and y_min - tolerance <= center_bucket[1] <= y_max + tolerance)
    inside_vertices = [
        _point_in_polygon_with_tolerance([vertex[0], vertex[2]], polygon,
                                         tolerance)
        and y_min - tolerance <= vertex[1] <= y_max + tolerance
        for vertex in vertices_bucket
    ]
    center_depth = sum(
        (center_bucket[axis] - proxy["mouth_center_bucket_m"][axis])
        * proxy["mouth_inward_direction_bucket"][axis]
        for axis in range(3))
    vertex_depths = [sum(
        (vertex[axis] - proxy["mouth_center_bucket_m"][axis])
        * proxy["mouth_inward_direction_bucket"][axis]
        for axis in range(3)) for vertex in vertices_bucket]
    y_overlaps = (max(vertex[1] for vertex in vertices_bucket) >= y_min - tolerance
                  and min(vertex[1] for vertex in vertices_bucket)
                  <= y_max + tolerance)
    return {
        "center_inside_bucket_prism": center_inside,
        "all_vertices_inside_bucket_prism": all(inside_vertices),
        "inside_vertex_count": sum(inside_vertices),
        "vertex_count": len(inside_vertices),
        "center_mouth_depth_m": center_depth,
        "minimum_vertex_mouth_depth_m": min(vertex_depths),
        "maximum_vertex_mouth_depth_m": max(vertex_depths),
        "mouth_straddling_proxy": (
            y_overlaps
            and min(vertex_depths) <= straddle_margin
            and max(vertex_depths) >= -straddle_margin),
        "center_bucket_m": center_bucket,
        "speed_m_s": _norm(rock_state["qvel"][:3]),
    }


def _scene_observations(state: dict, plan: dict, inventory: dict, proxy: dict,
                        tolerance: float, straddle_margin: float) -> list[dict]:
    prototypes = normalized_prototypes(
        sorted({rock["prototype_id"] for rock in plan["rocks"]}))
    result = []
    for rock_state, rock_plan in zip(state["rocks"], plan["rocks"]):
        record = _rock_observation(
            state, rock_state, rock_plan, prototypes, inventory["bucket"], proxy,
            tolerance, straddle_margin)
        record.update({
            "rock_id": rock_state["rock_id"],
            "source_rock_id": rock_plan["rock_id"],
            "mass_kg": rock_plan["mass_kg"],
            "equivalent_diameter_m": rock_plan["equivalent_diameter_m"],
            "prototype_id": rock_plan["prototype_id"],
        })
        result.append(record)
    return result


def _summary(observations: list[dict]) -> dict:
    full = [item for item in observations
            if item["all_vertices_inside_bucket_prism"]]
    center = [item for item in observations
              if item["center_inside_bucket_prism"]]
    return {
        "center_inside_count": len(center),
        "full_envelope_inside_count": len(full),
        "full_envelope_inside_mass_kg": sum(item["mass_kg"] for item in full),
        "mouth_straddling_proxy_count": sum(
            item["mouth_straddling_proxy"] for item in observations),
        "full_envelope_inside_rock_ids": [item["rock_id"] for item in full],
    }


def _run_scene(binary: Path, model: Path, plan: dict, inventory: dict,
               proxy: dict, config: dict) -> dict:
    integration = config["integration"]
    acceptance = config["acceptance"]
    dt = float(integration["output_dt_s"])
    substeps = int(integration["internal_substeps_per_output"])
    tolerance = float(acceptance["full_envelope_tolerance_m"])
    straddle = float(acceptance["mouth_straddle_depth_margin_m"])
    process = subprocess.Popen(
        [str(binary), str(model)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ready_line = process.stdout.readline().rstrip("\n")
    if not ready_line.startswith("@READY "):
        raise RuntimeError("controlled_intake_ready_failed:" + ready_line)
    ready = json.loads(ready_line[7:])
    process.stdin.write("init_default\n")
    process.stdin.flush()
    initial_line = process.stdout.readline().rstrip("\n")
    if not initial_line.startswith("@STATE "):
        raise RuntimeError("controlled_intake_init_failed:" + initial_line)
    state = json.loads(initial_line[7:])
    if ready["rock_count"] != 20 or len(state["rocks"]) != 20 or not _finite(state):
        raise RuntimeError("controlled_intake_initial_contract")
    initial_positions = {
        rock["rock_id"]: list(rock["position"]) for rock in state["rocks"]}
    rock_ids = [rock["rock_id"] for rock in state["rocks"]]
    consecutive_full = {rock_id: 0 for rock_id in rock_ids}
    maximum_consecutive_full = {rock_id: 0 for rock_id in rock_ids}
    ever_full = {rock_id: False for rock_id in rock_ids}
    contacts = {key: 0 for key in state["contact_substeps"]}
    impulse = [0.0, 0.0, 0.0]
    angular_impulse = [0.0, 0.0, 0.0]
    maximum_force = 0.0
    maximum_torque = 0.0
    maximum_closure = 0.0
    maximum_forbidden = 0
    maximum_concurrent_full = 0
    maximum_concurrent_center = 0
    maximum_rock_displacement = 0.0
    phase_end = {}
    runtime_error = None
    output_steps_completed = 0
    final_observations = _scene_observations(
        state, plan, inventory, proxy, tolerance, straddle)
    try:
        for phase in config["phases"]:
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
                    runtime_error = (line + (":" + stderr if stderr else ""))
                    raise RuntimeError("controlled_intake_step_failed")
                state = json.loads(line[7:])
                if len(state["rocks"]) != 20 or not _finite(state):
                    runtime_error = "nonfinite_or_identity_count_changed"
                    raise RuntimeError("controlled_intake_state_contract")
                final_observations = _scene_observations(
                    state, plan, inventory, proxy, tolerance, straddle)
                maximum_concurrent_full = max(
                    maximum_concurrent_full,
                    sum(item["all_vertices_inside_bucket_prism"]
                        for item in final_observations))
                maximum_concurrent_center = max(
                    maximum_concurrent_center,
                    sum(item["center_inside_bucket_prism"]
                        for item in final_observations))
                for item in final_observations:
                    rock_id = item["rock_id"]
                    if item["all_vertices_inside_bucket_prism"]:
                        consecutive_full[rock_id] += 1
                        ever_full[rock_id] = True
                    else:
                        consecutive_full[rock_id] = 0
                    maximum_consecutive_full[rock_id] = max(
                        maximum_consecutive_full[rock_id],
                        consecutive_full[rock_id])
                for key, value in state["contact_substeps"].items():
                    contacts[key] += int(value)
                for axis in range(3):
                    impulse[axis] += float(
                        state["bucket_rock_impulse_world_ns"][axis])
                    angular_impulse[axis] += float(
                        state["bucket_rock_angular_impulse_world_nms"][axis])
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
                for rock in state["rocks"]:
                    maximum_rock_displacement = max(
                        maximum_rock_displacement,
                        _norm([rock["position"][axis]
                               - initial_positions[rock["rock_id"]][axis]
                               for axis in range(3)]))
                output_steps_completed += 1
            phase_end[phase["name"]] = {
                "time_s": state["time_s"],
                "bucket_origin_world_m": state["bucket_origin"],
                "observation": _summary(final_observations),
                "cumulative_contact_substeps": dict(contacts),
                "maximum_bucket_rock_force_n": maximum_force,
                "maximum_bucket_rock_torque_nm": maximum_torque,
            }
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
                if not runtime_error:
                    runtime_error = "bridge_close_failed"
    final_by_id = {item["rock_id"]: item for item in final_observations}
    required_steps = math.ceil(
        float(acceptance["minimum_final_continuous_full_inside_duration_s"])
        / dt - 1e-12)
    retained_ids = [
        rock_id for rock_id in rock_ids
        if final_by_id[rock_id]["all_vertices_inside_bucket_prism"]
        and consecutive_full[rock_id] >= required_steps
    ]
    dropped_ids = [
        rock_id for rock_id in rock_ids
        if ever_full[rock_id]
        and not final_by_id[rock_id]["all_vertices_inside_bucket_prism"]
    ]
    retained_mass = sum(
        final_by_id[rock_id]["mass_kg"] for rock_id in retained_ids)
    final_retained_maximum_speed = max(
        (final_by_id[rock_id]["speed_m_s"] for rock_id in retained_ids),
        default=0.0)
    expected_steps = sum(
        round(float(phase["duration_s"]) / dt) for phase in config["phases"])
    checks = {
        "completed_full_schedule": (
            runtime_error is None and output_steps_completed == expected_steps),
        "twenty_rock_identities_preserved": (
            [rock["rock_id"] for rock in state["rocks"]] == rock_ids),
        "bucket_contact_observed": (
            contacts["rock_bucket"] >=
            int(acceptance["minimum_bucket_contact_substeps"])),
        "final_retained_full_rock_count": (
            len(retained_ids) >= int(
                acceptance["minimum_final_retained_full_rock_count"])),
        "retained_rocks_final_speed_within_limit": (
            final_retained_maximum_speed <= float(
                acceptance["maximum_final_retained_rock_speed_m_s"])),
        "closure_within_one_mm": (
            maximum_closure <= float(acceptance["maximum_closure_m"])),
        "no_forbidden_contacts": (
            maximum_forbidden ==
            int(acceptance["required_forbidden_contact_count"])),
        "finite_state": _finite(state),
    }
    return {
        "status": ("controlled_intake_scene_candidate"
                   if all(checks.values()) else "controlled_intake_scene_needs_review"),
        "runtime_error": runtime_error,
        "output_steps_completed": output_steps_completed,
        "expected_output_steps": expected_steps,
        "simulated_duration_s": float(state["time_s"]),
        "checks": checks,
        "contact_substeps": contacts,
        "maximum_closure_m": maximum_closure,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "maximum_bucket_rock_force_n": maximum_force,
        "maximum_bucket_rock_torque_nm": maximum_torque,
        "total_bucket_rock_impulse_world_ns": impulse,
        "total_bucket_rock_angular_impulse_world_nms": angular_impulse,
        "maximum_concurrent_center_inside_count": maximum_concurrent_center,
        "maximum_concurrent_full_envelope_inside_count": maximum_concurrent_full,
        "final_observation": _summary(final_observations),
        "final_retained_full_rock_ids": retained_ids,
        "final_retained_full_rock_count": len(retained_ids),
        "final_retained_mass_kg": retained_mass,
        "final_retained_maximum_speed_m_s": final_retained_maximum_speed,
        "dropped_after_full_entry_rock_ids": dropped_ids,
        "maximum_rock_displacement_m": maximum_rock_displacement,
        "per_rock": [{
            **item,
            "maximum_continuous_full_inside_duration_s":
                maximum_consecutive_full[item["rock_id"]] * dt,
            "final_continuous_full_inside_duration_s":
                consecutive_full[item["rock_id"]] * dt,
            "ever_full_envelope_inside": ever_full[item["rock_id"]],
        } for item in final_observations],
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
        raise ValueError("controlled_intake_output_already_exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    r2_config_path = repository / config["source_r2_config"]["path"]
    if sha256(r2_config_path) != config["source_r2_config"]["sha256"]:
        raise ValueError("controlled_intake_r2_config_sha256_mismatch")
    r2_config = json.loads(r2_config_path.read_text(encoding="utf-8"))
    stable_root = repository / config["stable_model_root"]["path"]
    stable_report_path = stable_root / "report.json"
    if sha256(stable_report_path) != config["stable_model_root"]["report_sha256"]:
        raise ValueError("controlled_intake_stable_report_sha256_mismatch")
    stable_report = json.loads(stable_report_path.read_text(encoding="utf-8"))
    if stable_report["status"] != "r2_six_scene_stable_initial_states_verified":
        raise ValueError("controlled_intake_stable_report_not_verified")
    bridge_source = stable_root / "generated-bridge-r2.cc"
    if (sha256(bridge_source) !=
            config["stable_model_root"]["generated_bridge_source_sha256"]):
        raise ValueError("controlled_intake_bridge_source_sha256_mismatch")
    geometry = config["bucket_geometry"]
    inventory_path = Path(geometry["inventory_path"])
    if sha256(inventory_path) != geometry["inventory_sha256"]:
        raise ValueError("controlled_intake_inventory_sha256_mismatch")
    proxy_path = repository / geometry["proxy_manifest_path"]
    if sha256(proxy_path) != geometry["proxy_manifest_sha256"]:
        raise ValueError("controlled_intake_proxy_sha256_mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    preflight_root = repository / r2_config["assembly_preflight"]["root"]
    stable_index = {record["scene_id"]: record
                    for record in stable_report["scenes"]}
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="mixed-r2-controlled-intake-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, bridge_source, binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        compile_record = [
            "<temporary>/mixed-rigid-bridge-r2"
            if value == str(binary.resolve()) else value
            for value in compile_command
        ]
        scene_records = []
        for scene_id in config["selected_scenes"]:
            if scene_id not in stable_index:
                raise ValueError("controlled_intake_scene_not_stable:" + scene_id)
            model_path = stable_root / scene_id / "model.xml"
            if sha256(model_path) != stable_index[scene_id]["model_sha256"]:
                raise ValueError("controlled_intake_model_sha256_mismatch:" + scene_id)
            plan_path = preflight_root / scene_id / "assembly-plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            metrics = _run_scene(binary, model_path, plan, inventory, proxy, config)
            scene_report = {
                "schema": "excavation_mixed_material_r2_controlled_intake_scene_report_v1",
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "model_sha256": sha256(model_path),
                "assembly_plan_sha256": sha256(plan_path),
                "status": metrics["status"],
                "metrics": metrics,
                "physics_qualified": False,
                "scope_limit": (
                    "bounded low-speed approach, curl and one-second settle only; "
                    "no lift/hold, dump, fill-rate, MPM or all-six-scene qualification"),
            }
            scene_dir = output / scene_id
            scene_dir.mkdir(parents=True)
            scene_report_path = scene_dir / "report.json"
            write_json(scene_report_path, scene_report)
            scene_records.append({
                "scene_id": scene_id,
                "scene_class": plan["scene_class"],
                "status": metrics["status"],
                "bucket_contact_substeps": metrics["contact_substeps"]["rock_bucket"],
                "rock_rock_contact_substeps": metrics["contact_substeps"]["rock_rock"],
                "final_retained_full_rock_count": metrics[
                    "final_retained_full_rock_count"],
                "final_retained_mass_kg": metrics["final_retained_mass_kg"],
                "dropped_after_full_entry_count": len(
                    metrics["dropped_after_full_entry_rock_ids"]),
                "final_mouth_straddling_proxy_count": metrics[
                    "final_observation"]["mouth_straddling_proxy_count"],
                "maximum_bucket_rock_force_n": metrics[
                    "maximum_bucket_rock_force_n"],
                "maximum_bucket_rock_torque_nm": metrics[
                    "maximum_bucket_rock_torque_nm"],
                "runtime_error": metrics["runtime_error"],
                "report_sha256": sha256(scene_report_path),
            })
    all_candidate = all(
        record["status"] == "controlled_intake_scene_candidate"
        for record in scene_records)
    report = {
        "schema": "excavation_mixed_material_r2_controlled_intake_report_v1",
        "status": ("two_scene_controlled_intake_candidate" if all_candidate
                   else "controlled_intake_needs_review"),
        "scene_count": len(scene_records),
        "scenes": scene_records,
        "all_six_scene_qualification": False,
        "lift_hold_qualified": False,
        "dump_qualified": False,
        "mpm_coupling_qualified": False,
        "fill_rate_qualified": False,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "r2_config_sha256": sha256(r2_config_path),
            "stable_report_sha256": sha256(stable_report_path),
            "bridge_source_sha256": sha256(bridge_source),
            "inventory_sha256": sha256(inventory_path),
            "proxy_manifest_sha256": sha256(proxy_path),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository / "src/v4/run_mixed_material_r2_controlled_intake.py"),
            "compile_command": compile_record,
        },
        "scope_limit": (
            "two selected stable scenes and one fixed low-speed intake trajectory; "
            "does not establish six-scene success rate, fill rate, lift/hold, dump, "
            "MPM coupling, hydraulic capacity or training readiness"),
    }
    write_json(output / "report.json", report)
    readme = (
        "# R2受控低速多石进斗首轮报告\n\n"
        f"根状态：`{report['status']}`。本目录只覆盖一个普通场景和一个"
        "稀疏大石场景的固定低速接近、卷斗和1秒静置。\n\n"
        "力和力矩是未做真实液压能力标定的诊断量；无论本轮是否通过，都不能"
        "据此宣称六场成功率、满斗率、举升保持、卸料、MPM耦合或训练资格。\n")
    (output / "README.md").write_text(readme, encoding="utf-8")
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
