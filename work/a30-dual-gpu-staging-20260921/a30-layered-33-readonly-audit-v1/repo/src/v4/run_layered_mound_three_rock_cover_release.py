#!/usr/bin/env python3
"""Hold three variably pressed rocks, settle cover soil, then release them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_fullscale_graded_gravel_slope import _inside_convex
from build_mixed_material_capacity_preflight import render_capacity_bridge, sha256
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz
from preview_layered_mound_existing_viewer import _mound_mask
from run_layered_mixed_scene_local_dynamics_smoke import _build_provisional_model
from run_layered_mound_single_rock_press_smoke import _finite_tree, _resolve, _rotation_matrix
from run_layered_mound_three_rock_adaptive_press import (
    MultiRockFixture,
    RuntimeBridge,
    _angle_between,
    _zero,
)
from run_mixed_material_r3_a_probe import _compile_bridge, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_three_rock_cover_release_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_three_rock_cover_release_report_v1"
PARTICLE_HISTORY = (
    "particle_elastic_strain", "particle_transform", "particle_qd_grad",
    "particle_stress", "particle_Jp")


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "three_variable_pressed_rocks_cover_settle_then_free_release"):
        raise ValueError("three_cover_config_schema_or_status")
    for group, fields in (
            ("failed_bare_release_gate", ("config", "runner", "report")),
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("source", ("visual_config", "source_bank"))):
        binding = value.get(group, {})
        for field in fields:
            target = _resolve(repository, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"three_cover_{group}_{field}_binding")
    failure = json.loads(_resolve(
        repository, value["failed_bare_release_gate"]["report_path"]
    ).read_text(encoding="utf-8"))
    failure_checks = failure.get("checks", {})
    if (failure.get("status") !=
            "three_lower_rocks_variable_press_free_settle_failed"
            or failure_checks.get("continuous_free_stable_window_reached") is not False
            or not all(result for name, result in failure_checks.items()
                       if name != "continuous_free_stable_window_reached")):
        raise ValueError("three_cover_bare_failure_contract")
    if [item.get("manifest_rock_id") for item in value.get("targets", [])] != [
            "lower-000", "lower-001", "lower-002"]:
        raise ValueError("three_cover_target_ids")
    if [(float(item["press_distance_m"]), int(item["press_frames"]))
            for item in value["targets"]] != [(0.012, 18), (0.008, 12), (0.016, 24)]:
        raise ValueError("three_cover_target_press_contract")
    if value.get("particles") != {
            "foundation_maximum_center_height_m": 0.32,
            "cover_maximum_center_height_m": 0.56,
            "foundation_particle_count": 85620,
            "raw_cover_particle_count": 49704,
            "cover_particle_count_after_pressed_rock_carve": 47459,
            "combined_particle_count": 133079,
            "cover_carve_policy":
            "remove_cover_particle_centers_inside_three_pressed_rocks_only",
            "activation_policy":
            "foundation_active_from_start_cover_dormant_then_history_restored"}:
        raise ValueError("three_cover_particle_contract")
    if value.get("construction") != {
            "start_surface_clearance_m": 0.005,
            "foundation_presettle_frames": 24,
            "cover_maximum_settle_frames": 240,
            "cover_stable_window_frames": 60,
            "rocks_held_during_cover_settle": True,
            "simultaneous_free_release_after_cover": True,
            "temporary_mpm_damping": 0.0,
            "temporary_rigid_damping": 0.0,
            "free_fall_from_height": False}:
        raise ValueError("three_cover_construction_contract")
    if value.get("acceptance") != {
            "press_distances_may_differ": True,
            "rock_translation_during_release_allowed": True,
            "rock_rotation_during_release_allowed": True,
            "rock_rock_contact_allowed": True,
            "final_manifest_pose_required": False,
            "cover_must_settle_before_release": True,
            "continuous_release_stable_window_required": True}:
        raise ValueError("three_cover_acceptance_contract")
    if value.get("scope") != {
            "active_rock_count": 3, "other_62_rocks_active": False,
            "full_lower_layer": False, "full_scene": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("three_cover_scope_contract")
    return value


def _prepare_particles(config: dict, repository: Path,
                       records: list[dict]) -> dict:
    source = config["source"]
    visual = json.loads(_resolve(
        repository, source["visual_config_path"]).read_text(encoding="utf-8"))
    with np.load(_resolve(repository, source["source_bank_path"]),
                 allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    mound = _mound_mask(source_q, visual)
    particle = config["particles"]
    foundation = mound & (source_q[:, 2] <= float(
        particle["foundation_maximum_center_height_m"]) + 1e-9)
    cover = (mound & (source_q[:, 2] > float(
        particle["foundation_maximum_center_height_m"]) + 1e-9)
        & (source_q[:, 2] <= float(
            particle["cover_maximum_center_height_m"]) + 1e-9))
    foundation_q = source_q[foundation]
    spacing = float(config["mpm"]["particle_spacing_m"])
    surface_extent = float(foundation_q[:, 2].max()) + 0.45 * spacing
    starts, pressed = [], []
    prototypes = normalized_prototypes(sorted({record["prototype_id"]
                                               for record in records}))
    remove_cover = np.zeros(len(source_q), dtype=bool)
    removed_by_rock = {}
    for target, record in zip(config["targets"], records, strict=True):
        vertices, _ = _read_obj(
            repository / "assets/v4/mixed-material-r0" /
            f"{record['prototype_id']}.obj")
        rotation = _rotation_matrix(record["orientation_quaternion_wxyz"])
        local = ((np.asarray(vertices, dtype=np.float64)
                  * float(record["equivalent_diameter_m"])) @ rotation.T)
        start = np.asarray(record["position_world_m"], dtype=np.float64)
        start[2] = (surface_extent
                    + float(config["construction"]["start_surface_clearance_m"])
                    - float(local[:, 2].min()))
        end = start.copy()
        end[2] -= float(target["press_distance_m"])
        starts.append(start)
        pressed.append(end)

        transform = quaternion_matrix_wxyz(record["orientation_quaternion_wxyz"])
        prototype = prototypes[record["prototype_id"]]
        world_vertices = (prototype["vertices"]
                          * float(record["equivalent_diameter_m"]) @ transform.T + end)
        candidates = np.flatnonzero(
            cover
            & np.all(source_q >= world_vertices.min(axis=0) - 1e-10, axis=1)
            & np.all(source_q <= world_vertices.max(axis=0) + 1e-10, axis=1))
        local_candidates = ((source_q[candidates].astype(np.float64) - end)
                            @ transform / float(record["equivalent_diameter_m"]))
        selected = candidates[_inside_convex(
            local_candidates, prototype["vertices"], prototype["faces"])]
        remove_cover[selected] = True
        removed_by_rock[record["rock_id"]] = int(len(selected))
    cover_q = source_q[cover & ~remove_cover]
    q = np.concatenate((foundation_q, cover_q), axis=0)
    activation = np.concatenate((
        np.zeros(len(foundation_q), dtype=np.int8),
        np.ones(len(cover_q), dtype=np.int8)))
    expected = config["particles"]
    if (len(foundation_q) != int(expected["foundation_particle_count"])
            or int(np.count_nonzero(cover)) != int(expected["raw_cover_particle_count"])
            or len(cover_q) != int(expected["cover_particle_count_after_pressed_rock_carve"])
            or len(q) != int(expected["combined_particle_count"])):
        raise ValueError("three_cover_prepared_particle_counts")
    return {
        "q": q, "activation": activation, "starts": starts, "pressed": pressed,
        "removed_cover_particle_count_by_rock": removed_by_rock,
        "foundation_q_sha256": hashlib.sha256(foundation_q.tobytes()).hexdigest(),
        "cover_q_sha256": hashlib.sha256(cover_q.tobytes()).hexdigest(),
        "combined_q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        "visual": visual,
    }


class ParticleActivation:
    def __init__(self, fixture: MultiRockFixture):
        self.fixture = fixture
        self.active_flag = int(newton.ParticleFlags.ACTIVE)
        self.history = {
            name: getattr(fixture.state.mpm, name).numpy().copy()
            for name in PARTICLE_HISTORY}

    def set_active(self, indices: np.ndarray) -> dict:
        flags = self.fixture.model.particle_flags.numpy().copy()
        current = (flags & self.active_flag) != 0
        requested = np.zeros(len(flags), dtype=bool)
        requested[indices] = True
        deactivated = np.flatnonzero(current & ~requested)
        activated = np.flatnonzero(~current & requested)
        for name in PARTICLE_HISTORY:
            state_array = getattr(self.fixture.state.mpm, name)
            values = state_array.numpy().copy()
            if len(deactivated):
                self.history[name][deactivated] = values[deactivated]
            if len(activated):
                values[activated] = self.history[name][activated]
                state_array.assign(values)
        flags &= ~self.active_flag
        flags[indices] |= self.active_flag
        self.fixture.model.particle_flags.assign(flags)
        self.fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        restored = all(np.array_equal(
            getattr(self.fixture.state.mpm, name).numpy()[activated],
            self.history[name][activated]) for name in PARTICLE_HISTORY)
        return {
            "activated_count": int(len(activated)),
            "deactivated_count": int(len(deactivated)),
            "active_count": int(len(indices)),
            "dormant_history_restored": bool(restored),
            "history_fields": list(PARTICLE_HISTORY),
        }


def _stable_particle(metrics: dict, probe: dict) -> bool:
    return (metrics["particle_max_speed_m_s"] <=
            float(probe["maximum_stable_particle_speed_m_s"])
            and metrics["particle_p99_speed_m_s"] <=
            float(probe["maximum_stable_particle_p99_speed_m_s"]))


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("three_cover_output_exists")
    config = load_config(config_path, repository)
    manifest_path = _resolve(repository, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
    records = [by_id[item["manifest_rock_id"]] for item in config["targets"]]
    prepared = _prepare_particles(config, repository, records)
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    fixture = MultiRockFixture(
        config, repository, records, prepared["q"], args.device)
    fixture.start_positions = [position.copy() for position in prepared["starts"]]
    activation = ParticleActivation(fixture)
    foundation_ids = np.flatnonzero(prepared["activation"] == 0)
    all_ids = np.arange(len(prepared["q"]), dtype=np.int64)
    initial_activation = activation.set_active(foundation_ids)
    positions = [position.copy() for position in prepared["starts"]]
    velocities = [np.zeros(3) for _ in records]
    dt = float(config["probe"]["output_dt_s"])
    kinematic_steps = 0
    kinematic_finite = True
    press_samples = 0
    for _ in range(int(config["construction"]["foundation_presettle_frames"])):
        fixture.assign(positions, velocities)
        _, metrics = fixture.step(dt, positions)
        kinematic_steps += 1
        kinematic_finite &= metrics["finite"]
    press_summary = []
    for index, target in enumerate(config["targets"]):
        start = positions[index].copy()
        distance = float(target["press_distance_m"])
        frames = int(target["press_frames"])
        speed = distance / (frames * dt)
        samples = 0
        for frame in range(frames):
            positions[index] = start.copy()
            positions[index][2] -= distance * (frame + 1) / frames
            velocities = [np.zeros(3) for _ in records]
            velocities[index][2] = -speed
            fixture.assign(positions, velocities)
            _, metrics = fixture.step(dt, positions)
            kinematic_steps += 1
            samples += metrics["contact_samples"]
            press_samples += metrics["contact_samples"]
            kinematic_finite &= metrics["finite"]
        press_summary.append({
            "manifest_rock_id": target["manifest_rock_id"],
            "press_distance_m": distance, "press_frames": frames,
            "press_speed_m_s": speed, "contact_samples": samples,
            "held_position_world_m": positions[index].tolist(),
        })
    if any(not np.allclose(actual, expected, rtol=0.0, atol=1e-9)
           for actual, expected in zip(positions, prepared["pressed"])):
        raise RuntimeError("three_cover_pressed_position_contract")

    cover_activation = activation.set_active(all_ids)
    cover_trace = []
    cover_stable_count = 0
    cover_steps = 0
    cover_contact_samples = 0
    cover_finite = True
    cover_peak_speed = 0.0
    for frame in range(int(config["construction"]["cover_maximum_settle_frames"])):
        fixture.assign(positions, [np.zeros(3) for _ in records])
        _, metrics = fixture.step(dt, positions)
        cover_steps += 1
        cover_contact_samples += metrics["contact_samples"]
        cover_peak_speed = max(cover_peak_speed, metrics["particle_max_speed_m_s"])
        cover_finite &= metrics["finite"]
        cover_stable_count = cover_stable_count + 1 if _stable_particle(
            metrics, config["probe"]) else 0
        if frame == 0 or (frame + 1) % 30 == 0 or cover_stable_count == int(
                config["construction"]["cover_stable_window_frames"]):
            cover_trace.append({"frame": frame,
                                "stable_window_frames": cover_stable_count,
                                **metrics})
        if cover_stable_count >= int(
                config["construction"]["cover_stable_window_frames"]):
            break

    cover_passed = (cover_finite and cover_stable_count >= int(
        config["construction"]["cover_stable_window_frames"]))
    release = {
        "executed": False, "steps": 0, "stable_window_frames": 0,
        "contact_samples": 0, "maximum_rock_linear_speed_m_s": 0.0,
        "maximum_rock_angular_speed_rad_s": 0.0,
        "maximum_particle_speed_m_s": 0.0,
        "maximum_forbidden_contact_count": 0,
        "all_finite": True, "centers_inside": True,
        "final_rocks": [], "trace": [],
    }
    model_manifest = None
    bridge_manifest = None
    compile_command = None
    if cover_passed:
        release_records = []
        for record, position in zip(records, positions):
            item = dict(record)
            item["position_world_m"] = position.tolist()
            release_records.append(item)
        capacity_path = _resolve(repository, config["capacity_bridge"]["config_path"])
        outer = np.asarray(prepared["visual"]["mound_envelope"][
            "outer_xy_bounds_m"], dtype=np.float64)
        initial_positions = [position.copy() for position in positions]
        initial_quaternions = [record["orientation_quaternion_wxyz"]
                               for record in records]
        with tempfile.TemporaryDirectory(prefix="three-rock-cover-release-") as raw:
            work = Path(raw)
            model = work / "release.xml"
            model_manifest = _build_provisional_model(
                capacity_path, repository, {"rocks": release_records}, model)
            source_path = work / "bridge.cc"
            source_code, bridge_manifest = render_capacity_bridge(
                capacity_path, repository)
            source_path.write_text(source_code, encoding="utf-8")
            binary = work / "bridge"
            compile_command = _compile_bridge(
                args.cxx, source_path, binary,
                [args.mechanism_header_dir, args.controller_header_dir,
                 args.mujoco_include], Path(args.mujoco_library_dir))
            bridge = RuntimeBridge(binary, model, len(records))
            release["executed"] = True
            try:
                stable_count = 0
                for frame in range(int(config["probe"]["maximum_release_frames"])):
                    state = bridge.state
                    fixture.assign_bridge(state)
                    origins = [np.asarray(rock["position"], dtype=np.float64)
                               for rock in state["rocks"]]
                    measured, particle_metrics = fixture.step(dt, origins)
                    state = bridge.step(
                        dt, int(config["probe"]["mujoco_internal_substeps"]),
                        {"bucket": _zero(state["bucket_origin"]), **measured})
                    release["steps"] += 1
                    release["contact_samples"] += particle_metrics["contact_samples"]
                    linear = [float(np.linalg.norm(
                        rock["linear_velocity_world_m_s"])) for rock in state["rocks"]]
                    angular = [float(np.linalg.norm(
                        rock["angular_velocity_world_rad_s"])) for rock in state["rocks"]]
                    max_linear, max_angular = max(linear), max(angular)
                    release["maximum_rock_linear_speed_m_s"] = max(
                        release["maximum_rock_linear_speed_m_s"], max_linear)
                    release["maximum_rock_angular_speed_rad_s"] = max(
                        release["maximum_rock_angular_speed_rad_s"], max_angular)
                    release["maximum_particle_speed_m_s"] = max(
                        release["maximum_particle_speed_m_s"],
                        particle_metrics["particle_max_speed_m_s"])
                    release["maximum_forbidden_contact_count"] = max(
                        release["maximum_forbidden_contact_count"],
                        int(state.get("forbidden_contact_count", 0)))
                    release["all_finite"] &= bool(
                        particle_metrics["finite"] and _finite_tree(state))
                    for rock in state["rocks"]:
                        point = np.asarray(rock["position"], dtype=np.float64)
                        release["centers_inside"] &= bool(
                            outer[0, 0] <= point[0] <= outer[1, 0]
                            and outer[0, 1] <= point[1] <= outer[1, 1]
                            and point[2] >= float(config["probe"]["minimum_world_z_m"]))
                    stable = (
                        max_linear <= float(config["probe"][
                            "maximum_stable_rock_linear_speed_m_s"])
                        and max_angular <= float(config["probe"][
                            "maximum_stable_rock_angular_speed_rad_s"])
                        and _stable_particle(particle_metrics, config["probe"]))
                    stable_count = stable_count + 1 if stable else 0
                    release["stable_window_frames"] = stable_count
                    if frame == 0 or (frame + 1) % 30 == 0 or stable_count == int(
                            config["probe"]["release_stable_window_frames"]):
                        fastest = int(np.argmax(linear))
                        release["trace"].append({
                            "frame": frame, "stable_window_frames": stable_count,
                            "max_rock_linear_speed_m_s": max_linear,
                            "max_rock_angular_speed_rad_s": max_angular,
                            "fastest_runtime_rock_id": f"r1-rock-{fastest:03d}",
                            **particle_metrics})
                    if stable_count >= int(config["probe"][
                            "release_stable_window_frames"]):
                        break
                final_state = bridge.state
                for index, rock in enumerate(final_state["rocks"]):
                    final_position = np.asarray(rock["position"], dtype=np.float64)
                    release["final_rocks"].append({
                        "manifest_rock_id": records[index]["rock_id"],
                        "runtime_rock_id": rock["rock_id"],
                        "final_position_world_m": final_position.tolist(),
                        "translation_from_release_m": float(np.linalg.norm(
                            final_position - initial_positions[index])),
                        "rotation_from_release_rad": _angle_between(
                            initial_quaternions[index], rock["qpos"][3:7]),
                        "final_linear_speed_m_s": float(np.linalg.norm(
                            rock["linear_velocity_world_m_s"])),
                        "final_angular_speed_rad_s": float(np.linalg.norm(
                            rock["angular_velocity_world_rad_s"])),
                    })
            finally:
                bridge.close(kill=not release["all_finite"])

    probe = config["probe"]
    checks = {
        "bare_release_failure_bound": True,
        "particle_partition_counts_match": len(prepared["q"]) == int(
            config["particles"]["combined_particle_count"]),
        "foundation_only_initial_activation": initial_activation["active_count"] == int(
            config["particles"]["foundation_particle_count"]),
        "unequal_press_schedule_preserved":
            [item["press_distance_m"] for item in press_summary] == [0.012, 0.008, 0.016],
        "kinematic_press_finite": bool(kinematic_finite),
        "cover_history_restored": bool(
            cover_activation["dormant_history_restored"]
            and cover_activation["activated_count"] == int(
                config["particles"]["cover_particle_count_after_pressed_rock_carve"])),
        "cover_settle_finite": bool(cover_finite),
        "cover_continuous_stable_window_reached": bool(cover_passed),
        "release_executed_only_after_cover_pass": bool(
            release["executed"] == cover_passed),
        "release_state_finite": bool(release["executed"] and release["all_finite"]),
        "release_contact_observed": bool(
            release["executed"] and release["contact_samples"] > 0),
        "release_speed_safety_bounds": bool(
            release["maximum_rock_linear_speed_m_s"] <= float(
                probe["maximum_rock_linear_speed_m_s"])
            and release["maximum_rock_angular_speed_rad_s"] <= float(
                probe["maximum_rock_angular_speed_rad_s"])
            and release["maximum_particle_speed_m_s"] <= float(
                probe["maximum_particle_speed_m_s"])),
        "release_forbidden_contacts_zero": release[
            "maximum_forbidden_contact_count"] <= int(
                probe["maximum_forbidden_contact_count"]),
        "release_centers_remain_in_bounds": bool(
            release["executed"] and release["centers_inside"]),
        "release_continuous_stable_window_reached": bool(
            release["stable_window_frames"] >= int(
                probe["release_stable_window_frames"])),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("three_rocks_cover_then_free_release_passed"
                   if all(checks.values()) else
                   "three_rocks_cover_then_free_release_failed"),
        "checks": checks,
        "particles": {
            "foundation_count": int(np.count_nonzero(prepared["activation"] == 0)),
            "cover_count": int(np.count_nonzero(prepared["activation"] == 1)),
            "combined_count": len(prepared["q"]),
            "removed_cover_particle_count_by_rock":
                prepared["removed_cover_particle_count_by_rock"],
            "foundation_q_sha256": prepared["foundation_q_sha256"],
            "cover_q_sha256": prepared["cover_q_sha256"],
            "combined_q_sha256": prepared["combined_q_sha256"],
            "initial_activation": initial_activation,
            "cover_activation": cover_activation,
        },
        "construction": {
            "presses": press_summary,
            "foundation_and_press_steps": kinematic_steps,
            "press_contact_samples": press_samples,
            "cover_settle_steps": cover_steps,
            "cover_settle_simulated_s": cover_steps * dt,
            "cover_stable_window_frames": cover_stable_count,
            "cover_contact_samples": cover_contact_samples,
            "cover_peak_particle_speed_m_s": cover_peak_speed,
            "cover_trace": cover_trace,
        },
        "release": release,
        "inputs": {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "physical_manifest_sha256": sha256(manifest_path),
            "failed_bare_release_report_sha256": sha256(_resolve(
                repository, config["failed_bare_release_gate"]["report_path"])),
            "model_manifest": model_manifest,
            "bridge_manifest": bridge_manifest,
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "compile_command": compile_command,
            "newton_solver_step_calls": kinematic_steps + cover_steps + release["steps"],
            "mujoco_step_calls": release["steps"],
            "wall_seconds": time.perf_counter() - started,
        },
        "acceptance": config["acceptance"],
        "scope": config["scope"],
        "stable_scene_claimed": False,
        "next_gate": "review_three_rock_cover_release_before_any_full_lower_layer",
    }
    if not _finite_tree(report):
        raise RuntimeError("three_cover_nonfinite_report")
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
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "particles": report["particles"],
        "construction": {key: value for key, value in report["construction"].items()
                         if key != "cover_trace"},
        "release": {key: value for key, value in report["release"].items()
                    if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
