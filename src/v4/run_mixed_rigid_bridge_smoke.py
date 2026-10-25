"""Reproducible load and one-second free-fall smoke for the R1 rigid bridge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from mixed_material_r1_mechanism import build_candidate, sha256
from mixed_material_r0 import normalized_prototypes


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _compile(cxx: str, source: Path, output: Path, include_dirs,
             mujoco_library_dir: Path) -> list[str]:
    command = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra"]
    command += ["-I" + str(Path(value).resolve()) for value in include_dirs]
    command += [
        str(source.resolve()), "-L" + str(mujoco_library_dir.resolve()),
        "-Wl,-rpath," + str(mujoco_library_dir.resolve()),
        "-l:libmujoco.so.2.3.8", "-o", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("compile_failed:" + completed.stderr)
    return command


def _command_schedule(scenario: dict, dt: float) -> tuple[list[dict], list[list[float]]]:
    phases = scenario.get("phases")
    if phases is None:
        phases = [{
            "name": "constant",
            "duration_s": float(scenario["duration_s"]),
            "command": scenario.get("command", [0.0, 0.0, 0.0, 0.0]),
        }]
    normalized = []
    commands = []
    elapsed_steps = 0
    for phase in phases:
        command = phase["command"]
        if (len(command) != 4 or
                not all(math.isfinite(float(value)) for value in command)):
            raise ValueError("invalid_scenario_command")
        phase_steps = round(float(phase["duration_s"]) / dt)
        if phase_steps <= 0:
            raise ValueError("invalid_scenario_phase_duration")
        normalized.append({
            "name": str(phase["name"]),
            "duration_s": float(phase["duration_s"]),
            "steps": phase_steps,
            "start_step": elapsed_steps,
            "end_step_exclusive": elapsed_steps + phase_steps,
            "command": [float(value) for value in command],
        })
        commands.extend([normalized[-1]["command"]] * phase_steps)
        elapsed_steps += phase_steps
    expected_steps = round(float(scenario["duration_s"]) / dt)
    if elapsed_steps != expected_steps:
        raise ValueError("scenario_phase_duration_mismatch")
    return normalized, commands


def _rock_center_bucket_coordinates(state: dict) -> list[float]:
    delta = [
        state["rock_position"][axis] - state["bucket_origin"][axis]
        for axis in range(3)
    ]
    rotation = state["bucket_rotation"]
    return [
        sum(rotation[3 * row + axis] * delta[row] for row in range(3))
        for axis in range(3)
    ]


def _point_on_segment(point, first, second, tolerance=1e-12) -> bool:
    cross = ((point[0] - first[0]) * (second[1] - first[1]) -
             (point[1] - first[1]) * (second[0] - first[0]))
    if abs(cross) > tolerance:
        return False
    return (min(first[0], second[0]) - tolerance <= point[0] <=
            max(first[0], second[0]) + tolerance and
            min(first[1], second[1]) - tolerance <= point[1] <=
            max(first[1], second[1]) + tolerance)


def _point_in_polygon(point, polygon) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        if ((current[1] > point[1]) != (previous[1] > point[1])):
            intersection_x = (previous[0] +
                              (point[1] - previous[1]) *
                              (current[0] - previous[0]) /
                              (current[1] - previous[1]))
            if point[0] < intersection_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(point, first, second) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-24:
        return math.hypot(point[0] - first[0], point[1] - first[1])
    projection = max(0.0, min(1.0,
        ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
        / length_squared))
    return math.hypot(point[0] - (first[0] + projection * dx),
                      point[1] - (first[1] + projection * dy))


def _point_in_polygon_with_tolerance(point, polygon, tolerance_m: float) -> bool:
    if _point_in_polygon(point, polygon):
        return True
    return min(_point_segment_distance(point, polygon[index - 1], polygon[index])
               for index in range(len(polygon))) <= tolerance_m


def _entry_observation(state: dict, manifest: dict, inventory: dict) -> dict:
    local = _rock_center_bucket_coordinates(state)
    center = manifest["mouth_center_bucket_m"]
    inward = manifest["mouth_inward_direction_bucket"]
    depth = sum((local[axis] - center[axis]) * inward[axis]
                for axis in range(3))
    bucket = inventory["bucket"]
    xz_inside = _point_in_polygon([local[0], local[2]],
                                  bucket["polygon_xz_m"])
    y_inside = bucket["y_limits_m"][0] <= local[1] <= bucket["y_limits_m"][1]
    return {
        "rock_center_bucket_m": local,
        "mouth_depth_m": depth,
        "center_inside_bucket_prism": xz_inside and y_inside,
    }


def _quaternion_rotate_wxyz(quaternion, vector) -> list[float]:
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


def _rock_vertices_world(state: dict, manifest: dict) -> list[list[float]]:
    rock = manifest["rock"]
    prototype = normalized_prototypes([rock["prototype_id"]])[
        rock["prototype_id"]]
    scale = float(rock["equivalent_diameter_m"])
    position = state["rock_qpos"][:3]
    quaternion = state["rock_qpos"][3:7]
    result = []
    for source in prototype["vertices"]:
        rotated = _quaternion_rotate_wxyz(
            quaternion, [float(value) * scale for value in source])
        result.append([position[axis] + rotated[axis] for axis in range(3)])
    return result


def _full_rock_observation(state: dict, manifest: dict, inventory: dict,
                           tolerance_m: float = 0.0) -> dict:
    vertices_world = _rock_vertices_world(state, manifest)
    bucket = inventory["bucket"]
    center = manifest["mouth_center_bucket_m"]
    inward = manifest["mouth_inward_direction_bucket"]
    vertices_bucket = []
    depths = []
    inside = []
    for world in vertices_world:
        local_state = dict(state)
        local_state["rock_position"] = world
        local = _rock_center_bucket_coordinates(local_state)
        vertices_bucket.append(local)
        depths.append(sum((local[axis] - center[axis]) * inward[axis]
                          for axis in range(3)))
        inside.append(
            _point_in_polygon_with_tolerance(
                [local[0], local[2]], bucket["polygon_xz_m"], tolerance_m)
            and bucket["y_limits_m"][0] - tolerance_m <= local[1] <=
            bucket["y_limits_m"][1] + tolerance_m)
    return {
        "all_vertices_inside_bucket_prism": all(inside),
        "inside_vertex_count": sum(inside),
        "vertex_count": len(inside),
        "minimum_vertex_mouth_depth_m": min(depths),
        "maximum_vertex_mouth_depth_m": max(depths),
        "vertices_world_m": vertices_world,
        "vertices_bucket_m": vertices_bucket,
        "containment_tolerance_m": tolerance_m,
    }


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    source_model = Path(args.source_model).resolve()
    source_hash_before = sha256(source_model)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    scenario = config["mechanism_candidate"]["unit_scenarios"][args.scenario]
    envelope_tolerance = float(scenario.get("acceptance", {}).get(
        "maximum_full_rock_envelope_tolerance_m", 0.0))
    command_schedule, scheduled_commands = _command_schedule(
        scenario, float(args.dt))
    step_count = (int(args.steps) if args.steps is not None else
                  len(scheduled_commands))
    if step_count > len(scheduled_commands):
        raise ValueError("steps_exceed_scenario_schedule")
    expected_mass = (
        2600.0 * math.pi / 6.0
        * config["mechanism_candidate"]["first_rock"]["equivalent_diameter_m"] ** 3)
    with tempfile.TemporaryDirectory(prefix="mixed-rigid-bridge-smoke-") as temporary:
        work = Path(temporary)
        candidate_model = work / "mechanism-r1.xml"
        manifest = build_candidate(
            source_model, candidate_model, repository, args.inventory, args.config,
            args.scenario)
        inspector_binary = work / "model-inspect"
        bridge_binary = work / "mixed-rigid-bridge"
        inspector_command = _compile(
            args.cxx, repository / "src/v4/mixed_material_r1_model_inspect.cc",
            inspector_binary, [args.mujoco_include], Path(args.mujoco_library_dir))
        bridge_command = _compile(
            args.cxx, repository / "src/v4/mixed_rigid_bridge.cc",
            bridge_binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        inspected = subprocess.run(
            [str(inspector_binary), str(candidate_model)], text=True,
            capture_output=True, check=False)
        if inspected.returncode:
            raise RuntimeError("model_inspect_failed:" + inspected.stderr)
        inspection = json.loads(inspected.stdout)

        process = subprocess.Popen(
            [str(bridge_binary), str(candidate_model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        ready_line = process.stdout.readline().strip()
        if not ready_line.startswith("@READY "):
            raise RuntimeError("bridge_start_failed:" + ready_line)
        ready = json.loads(ready_line[7:])
        process.stdin.write("init_default\n")
        process.stdin.flush()
        initial_line = process.stdout.readline().strip()
        if not initial_line.startswith("@STATE "):
            raise RuntimeError("bridge_init_failed:" + initial_line)
        state = json.loads(initial_line[7:])
        initial = state
        total_ground_contact_substeps = 0
        total_ramp_contact_substeps = 0
        total_bucket_contact_substeps = 0
        total_rock_rock_contact_substeps = 0
        total_truck_contact_substeps = 0
        maximum_closure = 0.0
        maximum_forbidden = 0
        minimum_rock_origin_z = state["rock_position"][2]
        maximum_mechanism_qpos_delta = 0.0
        maximum_mechanism_speed = 0.0
        total_bucket_rock_impulse = [0.0, 0.0, 0.0]
        total_bucket_rock_angular_impulse = [0.0, 0.0, 0.0]
        total_bucket_rock_contact_torque_impulse = [0.0, 0.0, 0.0]
        total_bucket_rock_moment_arm_impulse = [0.0, 0.0, 0.0]
        accumulated_orientation_change = 0.0
        previous_quaternion = state["rock_qpos"][3:7]
        initial_entry = _entry_observation(state, manifest, inventory)
        entry = initial_entry
        initial_full_rock = _full_rock_observation(
            state, manifest, inventory, envelope_tolerance)
        full_rock = initial_full_rock
        maximum_mouth_depth = initial_entry["mouth_depth_m"]
        inside_output_steps = int(initial_entry["center_inside_bucket_prism"])
        consecutive_inside_output_steps = inside_output_steps
        maximum_consecutive_inside_output_steps = consecutive_inside_output_steps
        consecutive_full_inside_output_steps = int(
            initial_full_rock["all_vertices_inside_bucket_prism"])
        maximum_consecutive_full_inside_output_steps = (
            consecutive_full_inside_output_steps)
        phase_end_observations = {}
        minimum_hold_rock_origin_z = None
        for step_index in range(step_count):
            command = scheduled_commands[step_index]
            process.stdin.write(
                "step %.17g %.17g %d %.17g %.17g %.17g %.17g 0\n" %
                (state["time_s"], float(args.dt), int(args.substeps),
                 float(command[0]), float(command[1]), float(command[2]),
                 float(command[3])))
            process.stdin.flush()
            line = process.stdout.readline().strip()
            if not line.startswith("@STATE "):
                raise RuntimeError("bridge_step_failed:" + line + ":" +
                                   process.stderr.read())
            state = json.loads(line[7:])
            entry = _entry_observation(state, manifest, inventory)
            full_rock = _full_rock_observation(
                state, manifest, inventory, envelope_tolerance)
            maximum_mouth_depth = max(maximum_mouth_depth,
                                      entry["mouth_depth_m"])
            if entry["center_inside_bucket_prism"]:
                inside_output_steps += 1
                consecutive_inside_output_steps += 1
            else:
                consecutive_inside_output_steps = 0
            maximum_consecutive_inside_output_steps = max(
                maximum_consecutive_inside_output_steps,
                consecutive_inside_output_steps)
            if full_rock["all_vertices_inside_bucket_prism"]:
                consecutive_full_inside_output_steps += 1
            else:
                consecutive_full_inside_output_steps = 0
            maximum_consecutive_full_inside_output_steps = max(
                maximum_consecutive_full_inside_output_steps,
                consecutive_full_inside_output_steps)
            phase = next(item for item in command_schedule
                         if item["start_step"] <= step_index <
                         item["end_step_exclusive"])
            if phase["name"] == "hold":
                minimum_hold_rock_origin_z = (
                    state["rock_position"][2]
                    if minimum_hold_rock_origin_z is None else
                    min(minimum_hold_rock_origin_z,
                        state["rock_position"][2]))
            if step_index + 1 == phase["end_step_exclusive"]:
                phase_end_observations[phase["name"]] = {
                    "time_s": state["time_s"],
                    "rock_position_m": state["rock_position"],
                    "entry_observation": entry,
                    "full_rock_observation": full_rock,
                    "rock_translational_speed_mps": math.sqrt(sum(
                        value * value for value in state["rock_qvel"][:3])),
                }
            current_quaternion = state["rock_qpos"][3:7]
            step_dot = min(1.0, abs(sum(
                first * second for first, second in zip(
                    previous_quaternion, current_quaternion))))
            accumulated_orientation_change += 2.0 * math.acos(step_dot)
            previous_quaternion = current_quaternion
            total_ground_contact_substeps += state["contact_substeps"]["rock_ground"]
            total_ramp_contact_substeps += state["contact_substeps"]["rock_ramp"]
            total_bucket_contact_substeps += state["contact_substeps"]["rock_bucket"]
            total_rock_rock_contact_substeps += state["contact_substeps"]["rock_rock"]
            total_truck_contact_substeps += state["contact_substeps"]["rock_truck"]
            for axis in range(3):
                total_bucket_rock_impulse[axis] += (
                    state["bucket_rock_impulse_world_ns"][axis])
                total_bucket_rock_angular_impulse[axis] += (
                    state["bucket_rock_angular_impulse_world_nms"][axis])
                total_bucket_rock_contact_torque_impulse[axis] += (
                    state["bucket_rock_contact_torque_impulse_world_nms"][axis])
                total_bucket_rock_moment_arm_impulse[axis] += (
                    state["bucket_rock_moment_arm_impulse_world_nms"][axis])
            maximum_closure = max(maximum_closure, state["closure_m"])
            maximum_forbidden = max(
                maximum_forbidden, state["forbidden_contact_count"])
            minimum_rock_origin_z = min(
                minimum_rock_origin_z, state["rock_position"][2])
            maximum_mechanism_qpos_delta = max(
                maximum_mechanism_qpos_delta,
                max(abs(a - b) for a, b in zip(
                    state["mechanism_qpos"], initial["mechanism_qpos"])))
            maximum_mechanism_speed = max(
                maximum_mechanism_speed,
                max(abs(value) for value in state["mechanism_qvel"]))
        process.stdin.write("close\n")
        process.stdin.flush()
        process.wait(timeout=5)
        stderr = process.stderr.read()
        if process.returncode:
            raise RuntimeError("bridge_exit_failed:" + stderr)

    final_speed = math.sqrt(sum(value * value for value in state["rock_qvel"][:3]))
    initial_quaternion = initial["rock_qpos"][3:7]
    final_quaternion = state["rock_qpos"][3:7]
    quaternion_dot = min(1.0, abs(sum(
        first * second for first, second in zip(
            initial_quaternion, final_quaternion))))
    orientation_change = 2.0 * math.acos(quaternion_dot)
    bucket_origin_displacement = [
        state["bucket_origin"][axis] - initial["bucket_origin"][axis]
        for axis in range(3)]
    rock_position_displacement = [
        state["rock_position"][axis] - initial["rock_position"][axis]
        for axis in range(3)]
    angular_impulse_decomposition_residual = math.sqrt(sum(
        (total_bucket_rock_angular_impulse[axis]
         - total_bucket_rock_contact_torque_impulse[axis]
         - total_bucket_rock_moment_arm_impulse[axis]) ** 2
        for axis in range(3)))
    common_checks = {
        "model_inspection_passed": bool(inspection["passed"]),
        "bridge_schema_passed": ready["schema"] == "excavation_mixed_rigid_bridge_v1",
        "bridge_rock_count_matches_manifest": (
            ready["rock_count"] == len(manifest["rocks"])),
        "source_model_unchanged": sha256(source_model) == source_hash_before,
        "mass_matches_r0_geometry": abs(inspection["rock_mass_kg"] - expected_mass) < 1e-12,
        "total_mass_matches_manifest": abs(
            inspection["total_rock_mass_kg"] - manifest["total_rock_mass_kg"]
        ) < 1e-12,
        "forbidden_contacts_zero": maximum_forbidden == 0,
        "closure_below_1mm": maximum_closure < 0.001,
    }
    if args.scenario == "freefall":
        scenario_checks = {
            "rock_fell_more_than_0_25m": (
                initial["rock_position"][2] - state["rock_position"][2] > 0.25),
            "rock_ground_contact_observed": total_ground_contact_substeps > 0,
            "rock_settled_below_1e_6": final_speed < 1e-6,
        }
        schema = "excavation_mixed_rigid_bridge_smoke_v1"
        scope_limit = (
            "qualifies model loading, state protocol, classified free fall and "
            "ground settling only; bucket entry, lift, hold and dump remain pending"
        )
    elif args.scenario == "slope_roll":
        acceptance = scenario["acceptance"]
        scenario_checks = {
            "rock_ramp_contact_observed": total_ramp_contact_substeps > 0,
            "rock_moved_down_slope": (
                state["rock_position"][0] - initial["rock_position"][0] >
                acceptance["minimum_down_slope_displacement_m"]),
            "rock_accumulated_rotation_passed": (
                accumulated_orientation_change >
                acceptance["minimum_accumulated_rotation_rad"]),
            "rock_lost_height": (
                initial["rock_position"][2] - minimum_rock_origin_z >
                acceptance["minimum_height_loss_m"]),
        }
        schema = "excavation_mixed_rigid_bridge_slope_roll_v1"
        scope_limit = (
            "qualifies one irregular rock contacting and moving down the fixed "
            "inclined fixture only; rock-rock and bucket tests remain pending"
        )
    elif args.scenario == "rock_collision":
        acceptance = scenario["acceptance"]
        initial_second = initial["rocks"][1]["position"]
        final_second = state["rocks"][1]["position"]
        second_planar_displacement = math.hypot(
            final_second[0] - initial_second[0],
            final_second[1] - initial_second[1])
        scenario_checks = {
            "two_rock_contract_loaded": len(state["rocks"]) == 2,
            "rock_rock_contact_observed": total_rock_rock_contact_substeps >= (
                acceptance["minimum_rock_rock_contact_substeps"]),
            "second_rock_planar_displacement_passed": (
                second_planar_displacement >
                acceptance["minimum_second_rock_lateral_displacement_m"]),
            "two_rock_mass_conserved": abs(
                inspection["total_rock_mass_kg"] - manifest["total_rock_mass_kg"]
            ) <= acceptance["mass_conservation_tolerance_kg"],
        }
        schema = "excavation_mixed_rigid_bridge_rock_collision_v1"
        scope_limit = (
            "qualifies two fixed-ID irregular rocks loading, contacting each other, "
            "and retaining total mass only; bucket tests remain pending"
        )
    elif args.scenario == "bucket_push":
        acceptance = scenario["acceptance"]
        scenario_checks = {
            "bucket_contact_observed": total_bucket_contact_substeps >= (
                acceptance["minimum_bucket_contact_substeps"]),
            "bucket_moved_positive_y": bucket_origin_displacement[1] > (
                acceptance["minimum_bucket_positive_y_displacement_m"]),
            "rock_pushed_positive_y": rock_position_displacement[1] > (
                acceptance["minimum_rock_positive_y_displacement_m"]),
            "bucket_reaction_opposes_push": (
                total_bucket_rock_impulse[1] <
                -acceptance["minimum_bucket_reaction_impulse_magnitude_ns"]),
            "angular_impulse_decomposition_passed": (
                angular_impulse_decomposition_residual <=
                acceptance[
                    "maximum_angular_impulse_decomposition_residual_nms"]),
        }
        schema = "excavation_mixed_rigid_bridge_bucket_push_v1"
        scope_limit = (
            "qualifies one controlled positive-y bucket push, reaction impulse "
            "sign, and torque decomposition only; entry and hold remain pending"
        )
    elif args.scenario == "bucket_push_clear":
        acceptance = scenario["acceptance"]
        scenario_checks = {
            "bucket_contact_absent": total_bucket_contact_substeps <= (
                acceptance["maximum_bucket_contact_substeps"]),
            "bucket_moved_positive_y": bucket_origin_displacement[1] > (
                acceptance["minimum_bucket_positive_y_displacement_m"]),
            "rock_remained_clear_in_y": abs(rock_position_displacement[1]) < (
                acceptance["maximum_rock_y_displacement_m"]),
            "bucket_rock_impulse_zero": max(
                abs(value) for value in total_bucket_rock_impulse) < 1e-12,
        }
        schema = "excavation_mixed_rigid_bridge_bucket_push_clear_v1"
        scope_limit = (
            "no-contact baseline for the bucket-push command; it is only valid "
            "when paired with the matching contact scenario"
        )
    elif args.scenario == "mechanism_entry":
        acceptance = scenario["acceptance"]
        final_inside_duration = (consecutive_inside_output_steps *
                                 float(args.dt))
        scenario_checks = {
            "rock_center_started_outside_mouth": (
                initial_entry["mouth_depth_m"] <=
                acceptance["maximum_initial_mouth_depth_m"]),
            "rock_center_reached_required_mouth_depth": (
                maximum_mouth_depth >=
                acceptance["minimum_maximum_mouth_depth_m"]),
            "rock_center_finished_at_required_mouth_depth": (
                entry["mouth_depth_m"] >=
                acceptance["minimum_final_mouth_depth_m"]),
            "rock_center_finished_inside_bucket_prism": (
                entry["center_inside_bucket_prism"]),
            "full_rock_finished_inside_bucket_prism": (
                full_rock["all_vertices_inside_bucket_prism"]),
            "final_continuous_inside_duration_passed": (
                final_inside_duration >=
                acceptance["minimum_final_inside_duration_s"]),
            "final_continuous_full_rock_inside_duration_passed": (
                consecutive_full_inside_output_steps * float(args.dt) >=
                acceptance["minimum_final_inside_duration_s"]),
            "bucket_contact_observed": total_bucket_contact_substeps >= (
                acceptance["minimum_bucket_contact_substeps"]),
            "final_rock_translational_speed_passed": final_speed <= (
                acceptance["maximum_final_rock_speed_mps"]),
        }
        schema = "excavation_mixed_rigid_bridge_mechanism_entry_v1"
        scope_limit = (
            "qualifies all vertices of one 0.12 m rigid rock entering the real-"
            "mechanism bucket prism with 1 mm containment tolerance; 10 s hold "
            "and dump remain separate qualifications"
        )
    elif args.scenario == "mechanism_hold":
        acceptance = scenario["acceptance"]
        lift_end = phase_end_observations["lift"]
        entry_end = phase_end_observations["curl_entry"]
        lifted_height = (lift_end["rock_position_m"][2] -
                         entry_end["rock_position_m"][2])
        fixture = scenario["fixture"]
        shelf_top = (fixture["position_world_m"][2] +
                     fixture["half_size_m"][2])
        center_clearance = lift_end["rock_position_m"][2] - shelf_top
        hold_height_loss = (lift_end["rock_position_m"][2] -
                            minimum_hold_rock_origin_z)
        final_inside_duration = (consecutive_inside_output_steps *
                                 float(args.dt))
        scenario_checks = {
            "rock_center_started_outside_mouth": (
                initial_entry["mouth_depth_m"] <=
                acceptance["maximum_initial_mouth_depth_m"]),
            "rock_lifted_from_entry_end": lifted_height >= (
                acceptance["minimum_lift_from_entry_end_m"]),
            "rock_center_cleared_support_top": center_clearance >= (
                acceptance["minimum_center_clearance_above_shelf_top_m"]),
            "rock_center_finished_at_required_mouth_depth": (
                entry["mouth_depth_m"] >=
                acceptance["minimum_final_mouth_depth_m"]),
            "rock_center_finished_inside_bucket_prism": (
                entry["center_inside_bucket_prism"]),
            "full_rock_finished_inside_bucket_prism": (
                full_rock["all_vertices_inside_bucket_prism"]),
            "final_continuous_inside_duration_passed": (
                final_inside_duration >=
                acceptance["minimum_final_inside_duration_s"]),
            "final_continuous_full_rock_inside_duration_passed": (
                consecutive_full_inside_output_steps * float(args.dt) >=
                acceptance["minimum_final_inside_duration_s"]),
            "hold_height_loss_bounded": hold_height_loss <= (
                acceptance["maximum_hold_height_loss_m"]),
            "bucket_contact_observed": total_bucket_contact_substeps >= (
                acceptance["minimum_bucket_contact_substeps"]),
            "final_rock_translational_speed_passed": final_speed <= (
                acceptance["maximum_final_rock_speed_mps"]),
        }
        schema = "excavation_mixed_rigid_bridge_mechanism_hold_v1"
        scope_limit = (
            "qualifies all vertices of one 0.12 m rock entering, lifting and "
            "remaining in the real-mechanism bucket prism for 10 s with 1 mm "
            "containment tolerance; multi-rock transport and dump remain pending"
        )
    elif args.scenario == "mechanism_dump":
        acceptance = scenario["acceptance"]
        bounds = scenario["receiver_inner_bounds_world_m"]
        final_position = state["rock_position"]
        center_in_receiver = all(
            bounds[axis][0] <= final_position[axis] <= bounds[axis][1]
            for axis in range(3))
        full_rock_in_receiver = all(
            bounds[axis][0] <= vertex[axis] <= bounds[axis][1]
            for vertex in full_rock["vertices_world_m"] for axis in range(3))
        lift_end = phase_end_observations["lift"]
        release_drop = lift_end["rock_position_m"][2] - final_position[2]
        scenario_checks = {
            "rock_center_started_outside_mouth": (
                initial_entry["mouth_depth_m"] <=
                acceptance["maximum_initial_mouth_depth_m"]),
            "bucket_contact_observed": total_bucket_contact_substeps >= (
                acceptance["minimum_bucket_contact_substeps"]),
            "truck_contact_observed": total_truck_contact_substeps >= (
                acceptance["minimum_truck_contact_substeps"]),
            "rock_center_finished_in_receiver": center_in_receiver,
            "full_rock_finished_in_receiver": full_rock_in_receiver,
            "full_rock_hold_duration_passed": (
                maximum_consecutive_full_inside_output_steps * float(args.dt) >=
                10.0),
            "rock_center_finished_outside_bucket_prism": not (
                entry["center_inside_bucket_prism"]),
            "full_rock_finished_outside_bucket_prism": (
                full_rock["inside_vertex_count"] == 0),
            "release_drop_passed": release_drop >= (
                acceptance["minimum_release_drop_m"]),
            "final_rock_translational_speed_passed": final_speed <= (
                acceptance["maximum_final_rock_speed_mps"]),
        }
        schema = "excavation_mixed_rigid_bridge_mechanism_dump_v1"
        scope_limit = (
            "qualifies all vertices of one 0.12 m rock entering, lifting, holding, "
            "swinging and settling in a fixed receiver with 1 mm containment "
            "tolerance; multi-rock dumping, actual truck CAD and receiver mass "
            "accounting remain pending"
        )
    else:
        raise ValueError("unsupported_scenario:" + args.scenario)
    checks = {**common_checks, **scenario_checks}
    report = {
        "schema": schema,
        "unit_scenario": args.scenario,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "source_model_sha256_before": source_hash_before,
        "source_model_sha256_after": sha256(source_model),
        "candidate_manifest": manifest,
        "model_inspection": inspection,
        "bridge_ready": ready,
        "run": {
            "steps": step_count,
            "dt_s": float(args.dt),
            "substeps": int(args.substeps),
            "simulated_duration_s": state["time_s"],
            "initial_rock_position_m": initial["rock_position"],
            "final_rock_position_m": state["rock_position"],
            "rock_position_displacement_m": rock_position_displacement,
            "initial_bucket_origin_m": initial["bucket_origin"],
            "final_bucket_origin_m": state["bucket_origin"],
            "bucket_origin_displacement_m": bucket_origin_displacement,
            "minimum_rock_origin_z_m": minimum_rock_origin_z,
            "final_rock_qvel": state["rock_qvel"],
            "final_rock_speed_norm": final_speed,
            "rock_ground_contact_substeps": total_ground_contact_substeps,
            "rock_ramp_contact_substeps": total_ramp_contact_substeps,
            "rock_bucket_contact_substeps": total_bucket_contact_substeps,
            "rock_rock_contact_substeps": total_rock_rock_contact_substeps,
            "rock_truck_contact_substeps": total_truck_contact_substeps,
            "maximum_forbidden_contact_count": maximum_forbidden,
            "maximum_closure_m": maximum_closure,
            "maximum_mechanism_qpos_delta": maximum_mechanism_qpos_delta,
            "maximum_mechanism_abs_qvel": maximum_mechanism_speed,
            "rock_orientation_change_rad": orientation_change,
            "rock_accumulated_orientation_change_rad": (
                accumulated_orientation_change),
            "initial_rocks": initial["rocks"],
            "final_rocks": state["rocks"],
            "command_schedule": command_schedule,
            "initial_entry_observation": initial_entry,
            "final_entry_observation": entry,
            "initial_full_rock_observation": initial_full_rock,
            "final_full_rock_observation": full_rock,
            "maximum_mouth_depth_m": maximum_mouth_depth,
            "inside_output_steps": inside_output_steps,
            "maximum_continuous_inside_duration_s": (
                maximum_consecutive_inside_output_steps * float(args.dt)),
            "final_continuous_inside_duration_s": (
                consecutive_inside_output_steps * float(args.dt)),
            "maximum_continuous_full_rock_inside_duration_s": (
                maximum_consecutive_full_inside_output_steps * float(args.dt)),
            "final_continuous_full_rock_inside_duration_s": (
                consecutive_full_inside_output_steps * float(args.dt)),
            "phase_end_observations": phase_end_observations,
            "minimum_hold_rock_origin_z_m": minimum_hold_rock_origin_z,
            "total_bucket_rock_impulse_world_ns": total_bucket_rock_impulse,
            "total_bucket_rock_angular_impulse_world_nms": (
                total_bucket_rock_angular_impulse),
            "total_bucket_rock_contact_torque_impulse_world_nms": (
                total_bucket_rock_contact_torque_impulse),
            "total_bucket_rock_moment_arm_impulse_world_nms": (
                total_bucket_rock_moment_arm_impulse),
            "angular_impulse_decomposition_residual_nms": (
                angular_impulse_decomposition_residual),
            "initial_mechanism_qpos": initial["mechanism_qpos"],
            "final_mechanism_qpos": state["mechanism_qpos"],
            "final_mechanism_qvel": state["mechanism_qvel"],
        },
        "build": {
            "inspector_command": inspector_command,
            "bridge_command": bridge_command,
            "model_inspector_sha256": sha256(
                repository / "src/v4/mixed_material_r1_model_inspect.cc"),
            "bridge_sha256": sha256(repository / "src/v4/mixed_rigid_bridge.cc"),
        },
        "scope_limit": scope_limit,
    }
    if args.report:
        write_json(args.report, report)
    if args.parent_report:
        parent_path = Path(args.parent_report)
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent["admission"]["mechanism_model_load_passed"] = bool(
            checks["model_inspection_passed"])
        if args.scenario == "freefall":
            parent["admission"]["rigid_bridge_freefall_passed"] = (
                report["status"] == "passed")
            parent["mechanism_freefall_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "slope_roll":
            parent["admission"]["rigid_bridge_slope_roll_passed"] = (
                report["status"] == "passed")
            parent["mechanism_slope_roll_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "rock_collision":
            parent["admission"]["rigid_bridge_rock_collision_passed"] = (
                report["status"] == "passed")
            parent["mechanism_rock_collision_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "bucket_push":
            parent["admission"]["rigid_bridge_bucket_push_passed"] = (
                report["status"] == "passed")
            parent["mechanism_bucket_push_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "bucket_push_clear":
            parent["admission"]["rigid_bridge_bucket_push_clear_passed"] = (
                report["status"] == "passed")
            parent["mechanism_bucket_push_clear_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "mechanism_entry":
            parent["admission"]["rigid_bridge_mechanism_entry_passed"] = (
                report["status"] == "passed")
            parent["mechanism_entry_report_sha256"] = (
                sha256(args.report) if args.report else None)
        elif args.scenario == "mechanism_hold":
            parent["admission"]["rigid_bridge_mechanism_hold_passed"] = (
                report["status"] == "passed")
            parent["mechanism_hold_report_sha256"] = (
                sha256(args.report) if args.report else None)
        else:
            parent["admission"]["rigid_bridge_mechanism_dump_passed"] = (
                report["status"] == "passed")
            complete_single_rock_chain = bool(
                report["status"] == "passed" and
                parent["admission"].get("rigid_bridge_mechanism_entry_passed") and
                parent["admission"].get("rigid_bridge_mechanism_hold_passed"))
            parent["admission"][
                "mechanism_attached_entry_hold_dump_passed"] = (
                    complete_single_rock_chain)
            parent["admission"]["r1_complete"] = complete_single_rock_chain
            parent["mechanism_dump_report_sha256"] = (
                sha256(args.report) if args.report else None)
        if report["status"] != "passed":
            parent["status"] = "mechanism_" + args.scenario + "_failed"
        elif parent["admission"].get("rigid_bridge_mechanism_dump_passed"):
            parent["status"] = "r1_full_rock_single_chain_passed_r2_pending"
        elif parent["admission"].get("rigid_bridge_mechanism_hold_passed"):
            parent["status"] = "mechanism_hold_passed_dump_pending"
        elif parent["admission"].get("rigid_bridge_mechanism_entry_passed"):
            parent["status"] = "mechanism_entry_passed_hold_dump_pending"
        elif parent["admission"].get("rigid_bridge_bucket_push_comparison_passed"):
            parent["status"] = (
                "standalone_probe_rigid_units_through_bucket_push_passed_"
                "entry_hold_dump_pending")
        elif (parent["admission"].get("rigid_bridge_bucket_push_passed") and
              parent["admission"].get("rigid_bridge_bucket_push_clear_passed")):
            parent["status"] = "bucket_push_pair_passed_comparison_pending"
        elif parent["admission"].get("rigid_bridge_rock_collision_passed"):
            parent["status"] = (
                "standalone_probe_mechanism_freefall_slope_roll_rock_collision_"
                "passed_bucket_tests_pending")
        elif parent["admission"].get("rigid_bridge_slope_roll_passed"):
            parent["status"] = (
                "standalone_probe_mechanism_freefall_slope_roll_passed_"
                "remaining_bucket_tests_pending")
        elif parent["admission"].get("rigid_bridge_freefall_passed"):
            parent["status"] = (
                "standalone_probe_and_mechanism_freefall_passed_"
                "bucket_tests_pending")
        write_json(parent_path, parent)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--report")
    parser.add_argument("--parent-report")
    parser.add_argument("--scenario", choices=(
        "freefall", "slope_roll", "rock_collision", "bucket_push",
        "bucket_push_clear", "mechanism_entry", "mechanism_hold",
        "mechanism_dump"),
                        default="freefall")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dt", type=float, default=1 / 120)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
