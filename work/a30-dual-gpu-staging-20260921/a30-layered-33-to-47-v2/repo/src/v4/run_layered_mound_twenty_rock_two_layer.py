#!/usr/bin/env python3
"""Construct and qualify the first two rock/soil layers (20 rocks)."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_fullscale_graded_gravel_slope import _inside_convex, sat_axis_gap
from build_mixed_material_capacity_preflight import render_capacity_bridge, sha256
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz
from mpm_rigid_free_velocity import predict_free_linear_velocity
from preview_layered_mound_existing_viewer import _mound_mask
from run_layered_mixed_scene_local_dynamics_smoke import _build_provisional_model
from run_layered_mound_single_rock_press_smoke import _finite_tree, _resolve, _rotation_matrix
from run_layered_mound_three_rock_adaptive_press import (
    MultiRockFixture, RuntimeBridge, _angle_between, _zero)
from run_layered_mound_three_rock_cover_release import ParticleActivation, _stable_particle
from run_mixed_material_r3_a_probe import _compile_bridge, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_twenty_rock_two_layer_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_twenty_rock_two_layer_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "twenty_rocks_two_sequential_soil_covers_then_free_release"):
        raise ValueError("twenty_two_layer_config_schema_or_status")
    for group, fields in (
            ("predecessor", ("config", "runner", "report")),
            ("runtime_dependencies", ("fixture", "activation", "predictor")),
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("source", ("visual_config", "source_bank"))):
        binding = value.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if not target.is_file() or sha256(target) != binding.get(field + "_sha256"):
                raise ValueError(f"twenty_two_layer_{group}_{field}_binding")
    predecessor = json.loads(_resolve(
        root, value["predecessor"]["report_path"]).read_text(encoding="utf-8"))
    if predecessor.get("status") != "nine_rock_lower_layer_passed" or not all(
            predecessor.get("checks", {}).values()):
        raise ValueError("twenty_two_layer_predecessor_not_passed")
    targets = value.get("targets", [])
    if ([item["manifest_rock_id"] for item in targets[:9]] !=
            [f"lower-{i:03d}" for i in range(9)]
            or [item["manifest_rock_id"] for item in targets[9:]] !=
            [f"lower_middle-{i:03d}" for i in range(11)]):
        raise ValueError("twenty_two_layer_target_order")
    bands = value.get("particle_bands")
    if bands != [
            {"stage": "foundation", "maximum_center_height_m": 0.32,
             "raw_count": 85620, "removed_count": 0, "active_count": 85620},
            {"stage": "lower_cover", "maximum_center_height_m": 0.56,
             "raw_count": 49704, "removed_count": 6443, "active_count": 43261},
            {"stage": "lower_middle_cover", "maximum_center_height_m": 0.84,
             "raw_count": 58320, "removed_count": 6647, "active_count": 51673}
            ] or value.get("combined_particle_count") != 180554:
        raise ValueError("twenty_two_layer_particle_contract")
    if value.get("scope") != {
            "active_rock_count": 20, "other_45_rocks_active": False,
            "completed_layers": 2, "full_scene": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("twenty_two_layer_scope_contract")
    return value


def _prepare(config: dict, root: Path, records: list[dict]) -> dict:
    visual = json.loads(_resolve(
        root, config["source"]["visual_config_path"]).read_text(encoding="utf-8"))
    with np.load(_resolve(root, config["source"]["source_bank_path"]),
                 allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    mound = _mound_mask(source_q, visual)
    limits = [float(item["maximum_center_height_m"])
              for item in config["particle_bands"]]
    masks = [mound & (source_q[:, 2] <= limits[0] + 1e-9)]
    for lower, upper in zip(limits, limits[1:]):
        masks.append(mound & (source_q[:, 2] > lower + 1e-9)
                     & (source_q[:, 2] <= upper + 1e-9))
    spacing = float(config["mpm"]["particle_spacing_m"])
    surfaces = [float(source_q[mask, 2].max()) + 0.45 * spacing
                for mask in masks[:-1]]
    prototypes = normalized_prototypes(sorted({r["prototype_id"] for r in records}))
    removes = [np.zeros(len(source_q), dtype=bool) for _ in range(2)]
    removed_by_rock = {}
    starts, pressed, geometry = [], [], []
    for target, record in zip(config["targets"], records, strict=True):
        stage = 0 if target["stage"] == "lower" else 1
        vertices, indices = _read_obj(
            root / "assets/v4/mixed-material-r0" / f"{record['prototype_id']}.obj")
        rotation = _rotation_matrix(record["orientation_quaternion_wxyz"])
        local = (np.asarray(vertices, dtype=np.float64)
                 * float(record["equivalent_diameter_m"])) @ rotation.T
        start = np.asarray(record["position_world_m"], dtype=np.float64)
        start[2] = (surfaces[stage]
                    + float(config["construction"]["start_surface_clearance_m"])
                    - float(local[:, 2].min())
                    + float(target["vertical_clearance_lift_m"]))
        end = start.copy()
        end[2] -= float(target["press_distance_m"])
        starts.append(start)
        pressed.append(end)
        transform = quaternion_matrix_wxyz(record["orientation_quaternion_wxyz"])
        prototype = prototypes[record["prototype_id"]]
        world_vertices = (prototype["vertices"]
                          * float(record["equivalent_diameter_m"]) @ transform.T + end)
        candidates = np.flatnonzero(
            masks[stage + 1]
            & np.all(source_q >= world_vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(source_q <= world_vertices.max(axis=0) + 1e-10, axis=1))
        local_candidates = ((source_q[candidates].astype(np.float64) - end)
                            @ transform / float(record["equivalent_diameter_m"]))
        selected = candidates[_inside_convex(
            local_candidates, prototype["vertices"], prototype["faces"])]
        removes[stage][selected] = True
        removed_by_rock[record["rock_id"]] = int(len(selected))
        geometry.append({"id": record["rock_id"], "vertices": world_vertices,
                         "faces": prototype["faces"]})
    gaps, intersections = [], []
    for first, second in combinations(geometry, 2):
        intersects, gap = sat_axis_gap(
            first["vertices"], first["faces"], second["vertices"], second["faces"])
        gaps.append(float(gap))
        if intersects:
            intersections.append([first["id"], second["id"]])
    minimum_gap = min(gaps)
    required_gap = float(config["physical_geometry"][
        "required_minimum_pressed_sat_gap_m"])
    if intersections or minimum_gap < required_gap - 1e-12:
        raise ValueError("twenty_two_layer_pressed_geometry_intersection")
    arrays = [source_q[masks[0]], source_q[masks[1] & ~removes[0]],
              source_q[masks[2] & ~removes[1]]]
    expected = config["particle_bands"]
    for index, (mask, remove, array) in enumerate(zip(
            masks, [np.zeros(len(source_q), bool), *removes], arrays)):
        if (int(np.count_nonzero(mask)) != int(expected[index]["raw_count"])
                or int(np.count_nonzero(mask & remove)) != int(
                    expected[index]["removed_count"])
                or len(array) != int(expected[index]["active_count"])):
            raise ValueError(f"twenty_two_layer_particle_band_{index}")
    q = np.concatenate(arrays, axis=0)
    if len(q) != int(config["combined_particle_count"]):
        raise ValueError("twenty_two_layer_combined_count")
    activation = np.concatenate([
        np.full(len(array), index, dtype=np.int8)
        for index, array in enumerate(arrays)])
    return {
        "q": q, "activation": activation, "starts": starts, "pressed": pressed,
        "visual": visual, "minimum_sat_gap_m": minimum_gap,
        "removed_by_rock": removed_by_rock,
        "band_counts": [len(a) for a in arrays],
        "q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
    }


def run(args) -> dict:
    root = Path(args.repository_root).resolve()
    config_path, output = Path(args.config).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("twenty_two_layer_output_exists")
    config = load_config(config_path, root)
    manifest_path = _resolve(root, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
    records = [by_id[item["manifest_rock_id"]] for item in config["targets"]]
    prepared = _prepare(config, root, records)
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    fixture = MultiRockFixture(config, root, records, prepared["q"], args.device)
    activation = ParticleActivation(fixture)
    dt = float(config["probe"]["output_dt_s"])
    stage_ids = [np.flatnonzero(prepared["activation"] <= stage) for stage in range(3)]
    activation_reports = [activation.set_active(stage_ids[0])]
    positions = [value.copy() for value in prepared["starts"]]
    parking = np.asarray(config["construction"][
        "park_later_stage_position_world_m"], dtype=np.float64)
    for index in range(9, 20):
        positions[index] = parking.copy()
    kinematic_steps = 0
    kinematic_finite = True
    parked_upper_force_max_n = 0.0
    press_reports = []

    def held_step() -> tuple[dict, dict]:
        nonlocal kinematic_steps, kinematic_finite, parked_upper_force_max_n
        fixture.assign(positions, [np.zeros(3) for _ in records])
        wrenches, metrics = fixture.step(dt, positions)
        kinematic_steps += 1
        kinematic_finite &= metrics["finite"]
        if all(np.allclose(positions[i], parking) for i in range(9, 20)):
            parked_upper_force_max_n = max(
                parked_upper_force_max_n,
                max(float(np.linalg.norm(wrenches[f"r1-rock-{i:03d}"]["force"]))
                    for i in range(9, 20)))
        return wrenches, metrics

    for _ in range(int(config["construction"]["foundation_presettle_frames"])):
        held_step()

    def press(indices: range) -> None:
        nonlocal kinematic_steps, kinematic_finite, parked_upper_force_max_n
        for index in indices:
            target = config["targets"][index]
            start = positions[index].copy()
            distance, frames = float(target["press_distance_m"]), int(target["press_frames"])
            speed = distance / (frames * dt)
            samples = 0
            for frame in range(frames):
                positions[index] = start.copy()
                positions[index][2] -= distance * (frame + 1) / frames
                velocities = [np.zeros(3) for _ in records]
                velocities[index][2] = -speed
                fixture.assign(positions, velocities)
                wrenches, metrics = fixture.step(dt, positions)
                kinematic_steps += 1
                kinematic_finite &= metrics["finite"]
                samples += metrics["contact_samples"]
                if index < 9:
                    parked_upper_force_max_n = max(
                        parked_upper_force_max_n,
                        max(float(np.linalg.norm(wrenches[f"r1-rock-{i:03d}"]["force"]))
                            for i in range(9, 20)))
            press_reports.append({
                "manifest_rock_id": target["manifest_rock_id"],
                "stage": target["stage"], "press_distance_m": distance,
                "press_frames": frames, "press_speed_m_s": speed,
                "vertical_clearance_lift_m": target["vertical_clearance_lift_m"],
                "contact_samples": samples,
                "held_position_world_m": positions[index].tolist()})

    press(range(0, 9))
    activation_reports.append(activation.set_active(stage_ids[1]))
    settle_reports = []

    def settle(stage_name: str) -> bool:
        stable_count, contact_samples, peak, finite = 0, 0, 0.0, True
        trace = []
        for frame in range(int(config["construction"]["cover_maximum_settle_frames"])):
            _, metrics = held_step()
            contact_samples += metrics["contact_samples"]
            peak = max(peak, metrics["particle_max_speed_m_s"])
            finite &= metrics["finite"]
            stable_count = stable_count + 1 if _stable_particle(
                metrics, config["probe"]) else 0
            if frame == 0 or (frame + 1) % 30 == 0:
                trace.append({"frame": frame, "stable_window_frames": stable_count,
                              **metrics})
            if stable_count >= int(config["construction"]["cover_stable_window_frames"]):
                break
        passed = finite and stable_count >= int(
            config["construction"]["cover_stable_window_frames"])
        settle_reports.append({
            "stage": stage_name, "passed": bool(passed), "steps": frame + 1,
            "stable_window_frames": stable_count, "contact_samples": contact_samples,
            "peak_particle_speed_m_s": peak, "trace": trace})
        return bool(passed)

    first_cover_passed = settle("lower_cover")
    for index in range(9, 20):
        positions[index] = prepared["starts"][index].copy()
    if first_cover_passed:
        press(range(9, 20))
        activation_reports.append(activation.set_active(stage_ids[2]))
        second_cover_passed = settle("lower_middle_cover")
    else:
        second_cover_passed = False

    release = {"executed": False, "steps": 0, "stable_window_frames": 0,
               "all_finite": True, "trace": [], "rock_rock_contact_substeps": 0,
               "external_rigid_support_substeps": 0, "maximum_forbidden_contacts": 0,
               "maximum_global_momentum_residual_ns": 0.0,
               "maximum_rock_linear_speed_m_s": 0.0,
               "maximum_rock_angular_speed_rad_s": 0.0,
               "maximum_particle_speed_m_s": 0.0}
    hold = {"requested_frames": int(config["probe"]["additional_uncontrolled_hold_frames"]),
            "executed_frames": 0, "all_frames_stable": False, "trace": []}
    model_manifest = bridge_manifest = compile_command = None
    gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
    dynamic_positions = []
    if second_cover_passed:
        release_records = []
        for record, position in zip(records, positions):
            item = dict(record)
            item["position_world_m"] = position.tolist()
            release_records.append(item)
        outer = np.asarray(prepared["visual"]["mound_envelope"][
            "outer_xy_bounds_m"], dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="twenty-rock-two-layer-") as raw:
            work = Path(raw)
            model = work / "release.xml"
            model_manifest = _build_provisional_model(
                _resolve(root, config["capacity_bridge"]["config_path"]),
                root, {"rocks": release_records}, model)
            source = work / "bridge.cc"
            code, bridge_manifest = render_capacity_bridge(
                _resolve(root, config["capacity_bridge"]["config_path"]), root)
            source.write_text(code, encoding="utf-8")
            binary = work / "bridge"
            compile_command = _compile_bridge(
                args.cxx, source, binary,
                [args.mechanism_header_dir, args.controller_header_dir,
                 args.mujoco_include], Path(args.mujoco_library_dir))
            bridge = RuntimeBridge(binary, model, len(records))
            release["executed"] = True

            def dynamic_step(frame: int) -> tuple[bool, dict]:
                before = bridge.state
                fixture.assign(
                    [np.asarray(r["position"], dtype=float) for r in before["rocks"]],
                    [predict_free_linear_velocity(
                        np.asarray(r["qvel"][:3], dtype=float), gravity, dt)
                     for r in before["rocks"]],
                    [np.asarray(r["angular_velocity_world_rad_s"], dtype=float)
                     for r in before["rocks"]],
                    [r["qpos"][3:7] for r in before["rocks"]])
                origins = [np.asarray(r["position"], dtype=float) for r in before["rocks"]]
                measured, particles = fixture.step(dt, origins)
                after = bridge.step(
                    dt, int(config["probe"]["mujoco_internal_substeps"]),
                    {"bucket": _zero(before["bucket_origin"]), **measured})
                delta = sum(float(record["mass_kg"]) * (
                    np.asarray(current["qvel"][:3]) - np.asarray(prior["qvel"][:3]))
                    for record, prior, current in zip(records, before["rocks"], after["rocks"]))
                expected = dt * sum(
                    np.asarray(measured[prior["rock_id"]]["force"])
                    + float(record["mass_kg"]) * gravity
                    for record, prior in zip(records, before["rocks"]))
                residual = float(np.linalg.norm(delta - expected))
                linear = max(float(np.linalg.norm(r["qvel"][:3])) for r in after["rocks"])
                angular = max(float(np.linalg.norm(r["angular_velocity_world_rad_s"]))
                              for r in after["rocks"])
                contacts = after.get("contact_substeps", {})
                external = sum(int(contacts.get(k, 0)) for k in
                               ("rock_ground", "rock_bucket", "rock_ramp", "rock_truck"))
                stable = (linear <= config["probe"]["maximum_stable_rock_linear_speed_m_s"]
                          and angular <= config["probe"]["maximum_stable_rock_angular_speed_rad_s"]
                          and _stable_particle(particles, config["probe"]))
                finite = bool(particles["finite"] and _finite_tree(after))
                centers_inside = all(
                    outer[0, 0] <= r["position"][0] <= outer[1, 0]
                    and outer[0, 1] <= r["position"][1] <= outer[1, 1]
                    and r["position"][2] >= config["probe"]["minimum_world_z_m"]
                    for r in after["rocks"])
                row = {"frame": frame, "stable": bool(stable), "finite": finite,
                       "centers_inside": bool(centers_inside),
                       "max_raw_linear_speed_m_s": linear,
                       "max_angular_speed_rad_s": angular,
                       "global_momentum_residual_ns": residual,
                       "contact_substeps": contacts,
                       "positions_world_m": [r["qpos"][:3] for r in after["rocks"]],
                       **particles}
                dynamic_positions.append(row)
                release["rock_rock_contact_substeps"] += int(contacts.get("rock_rock", 0))
                release["external_rigid_support_substeps"] += external
                release["maximum_forbidden_contacts"] = max(
                    release["maximum_forbidden_contacts"],
                    int(after.get("forbidden_contact_count", 0)))
                release["maximum_global_momentum_residual_ns"] = max(
                    release["maximum_global_momentum_residual_ns"], residual)
                release["maximum_rock_linear_speed_m_s"] = max(
                    release["maximum_rock_linear_speed_m_s"], linear)
                release["maximum_rock_angular_speed_rad_s"] = max(
                    release["maximum_rock_angular_speed_rad_s"], angular)
                release["maximum_particle_speed_m_s"] = max(
                    release["maximum_particle_speed_m_s"], particles["particle_max_speed_m_s"])
                release["all_finite"] &= finite
                return stable, row

            try:
                consecutive = 0
                for frame in range(int(config["probe"]["maximum_release_frames"])):
                    stable, row = dynamic_step(frame)
                    release["steps"] += 1
                    consecutive = consecutive + 1 if stable else 0
                    release["stable_window_frames"] = consecutive
                    if frame == 0 or (frame + 1) % 30 == 0:
                        release["trace"].append(row)
                    if not row["finite"] or not row["centers_inside"]:
                        break
                    if consecutive >= int(config["probe"]["release_stable_window_frames"]):
                        break
                if consecutive >= int(config["probe"]["release_stable_window_frames"]):
                    for frame in range(hold["requested_frames"]):
                        stable, row = dynamic_step(release["steps"] + frame)
                        hold["executed_frames"] += 1
                        hold["trace"].append(row)
                        if not stable or not row["finite"] or not row["centers_inside"]:
                            break
                    hold["all_frames_stable"] = (
                        hold["executed_frames"] == hold["requested_frames"]
                        and all(row["stable"] for row in hold["trace"]))
            finally:
                bridge.close(kill=not release["all_finite"])

    tail = dynamic_positions[-120:]
    checks = {
        "nine_rock_predecessor_passed": True,
        "pressed_geometry_sat_gap_passed": prepared["minimum_sat_gap_m"] >= float(
            config["physical_geometry"]["required_minimum_pressed_sat_gap_m"]),
        "particle_partition_counts_match": prepared["band_counts"] == [85620, 43261, 51673],
        "later_stage_rocks_parked_without_mpm_force": parked_upper_force_max_n == 0.0,
        "kinematic_construction_finite": bool(kinematic_finite),
        "both_cover_histories_restored": len(activation_reports) == 3 and all(
            item["dormant_history_restored"] for item in activation_reports[1:]),
        "both_cover_stable_windows_reached": len(settle_reports) == 2 and all(
            item["passed"] for item in settle_reports),
        "twenty_rock_release_executed": release["executed"],
        "release_state_finite": release["all_finite"],
        "release_continuous_stable_window_reached": release["stable_window_frames"] >= int(
            config["probe"]["release_stable_window_frames"]),
        "additional_uncontrolled_hold_all_frames_stable": hold["all_frames_stable"],
        "no_external_rigid_support": release["external_rigid_support_substeps"] == 0,
        "forbidden_contacts_zero": release["maximum_forbidden_contacts"] == 0,
        "gravity_and_mpm_global_impulse_accounted_once":
            release["maximum_global_momentum_residual_ns"] < 1e-4,
        "release_speed_safety_bounds": (
            release["maximum_rock_linear_speed_m_s"] <= config["probe"]["maximum_rock_linear_speed_m_s"]
            and release["maximum_rock_angular_speed_rad_s"] <= config["probe"]["maximum_rock_angular_speed_rad_s"]
            and release["maximum_particle_speed_m_s"] <= config["probe"]["maximum_particle_speed_m_s"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "twenty_rock_two_layer_passed" if all(checks.values()) else
                  "twenty_rock_two_layer_failed",
        "checks": checks,
        "geometry": {"minimum_pressed_sat_gap_m": prepared["minimum_sat_gap_m"],
                     "pressed_pair_count": math.comb(20, 2)},
        "particles": {"band_counts": prepared["band_counts"],
                      "combined_count": len(prepared["q"]),
                      "combined_q_sha256": prepared["q_sha256"],
                      "removed_by_rock": prepared["removed_by_rock"],
                      "activation_reports": activation_reports},
        "construction": {"presses": press_reports, "settles": settle_reports,
                         "kinematic_steps": kinematic_steps,
                         "parked_upper_force_max_n": parked_upper_force_max_n},
        "release": release,
        "additional_uncontrolled_hold": hold,
        "last_120_frames": {
            "maximum_raw_linear_speed_m_s": max(
                (row["max_raw_linear_speed_m_s"] for row in tail), default=None),
            "position_change_m": [float(np.linalg.norm(
                np.asarray(tail[-1]["positions_world_m"][i])
                - np.asarray(tail[0]["positions_world_m"][i])))
                for i in range(20)] if tail else []},
        "inputs": {"config_sha256": sha256(config_path),
                   "runner_sha256": sha256(Path(__file__).resolve()),
                   "manifest_sha256": sha256(manifest_path),
                   "model_manifest": model_manifest,
                   "bridge_manifest": bridge_manifest},
        "environment": {"python": platform.python_version(),
                        "newton": newton.__version__, "warp": wp.__version__,
                        "numpy": np.__version__, "device": str(wp.get_device())},
        "execution": {"compile_command": compile_command,
                      "newton_solver_step_calls": kinematic_steps + len(dynamic_positions),
                      "mujoco_step_calls": len(dynamic_positions),
                      "wall_seconds": time.perf_counter() - started},
        "scope": config["scope"], "stable_scene_claimed": False,
        "next_gate": "review_two_layers_before_adding_middle_layer",
    }
    if not _finite_tree(report):
        raise RuntimeError("twenty_two_layer_nonfinite_report")
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "geometry": report["geometry"], "particles": report["particles"],
                      "release": {k: v for k, v in report["release"].items() if k != "trace"},
                      "last_120_frames": report["last_120_frames"],
                      "execution": report["execution"]}, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "twenty_rock_two_layer_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
