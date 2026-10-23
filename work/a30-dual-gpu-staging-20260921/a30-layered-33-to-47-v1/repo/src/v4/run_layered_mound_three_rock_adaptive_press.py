#!/usr/bin/env python3
"""Press three lower rocks by unequal short distances, release, and settle.

Translation and rotation after release are observations, not failures.  The
gate asks whether all three freely moving rocks and the foundation reach a
continuous low-speed window without non-finite state or leaving the mound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from build_fullscale_graded_gravel_slope import sat_axis_gap
from build_mixed_material_capacity_preflight import render_capacity_bridge, sha256
from run_layered_mixed_scene_local_dynamics_smoke import _build_provisional_model
from run_layered_mound_single_rock_press_smoke import (
    _finite_tree,
    _foundation_particles,
    _resolve,
    _rotation_matrix,
)
from run_mixed_material_r3_a_probe import _compile_bridge, _make_transform, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_three_rock_adaptive_press_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_three_rock_adaptive_press_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "three_lower_rocks_variable_short_press_then_free_settle"):
        raise ValueError("three_press_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("config", "report", "manifest")),
            ("single_press_gate", ("config", "runner", "report")),
            ("capacity_bridge", ("config",)),
            ("unexcavated_soil", ("visual_config", "source_bank"))):
        binding = value.get(group, {})
        for field in fields:
            target = _resolve(repository, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"three_press_{group}_{field}_binding")
    single = json.loads(_resolve(
        repository, value["single_press_gate"]["report_path"]
    ).read_text(encoding="utf-8"))
    if (single.get("status") != "single_lower_rock_controlled_press_smoke_passed"
            or not all(single.get("checks", {}).values())):
        raise ValueError("three_press_single_gate_not_passed")
    manifest = json.loads(_resolve(
        repository, value["physical_geometry"]["manifest_path"]
    ).read_text(encoding="utf-8"))
    targets = value.get("targets", [])
    if ([item.get("manifest_rock_id") for item in targets] !=
            ["lower-000", "lower-001", "lower-002"]
            or [float(item.get("press_distance_m", -1)) for item in targets] !=
            [0.012, 0.008, 0.016]
            or [int(item.get("press_frames", -1)) for item in targets] != [18, 12, 24]):
        raise ValueError("three_press_target_contract")
    records = {rock["rock_id"]: rock for rock in manifest.get("rocks", [])}
    if (manifest.get("rock_count") != 65
            or any(identifier not in records or records[identifier]["layer_id"] != "lower"
                   for identifier in ("lower-000", "lower-001", "lower-002"))):
        raise ValueError("three_press_manifest_contract")
    if value.get("construction") != {
            "start_surface_clearance_m": 0.005,
            "foundation_presettle_frames": 24,
            "press_sequence": "one_at_a_time_other_rocks_held",
            "orientation_locked_during_press": True,
            "simultaneous_free_release_after_press": True,
            "free_fall_from_height": False}:
        raise ValueError("three_press_construction_contract")
    if value.get("acceptance") != {
            "press_distances_may_differ": True,
            "rock_translation_during_release_allowed": True,
            "rock_rotation_during_release_allowed": True,
            "rock_rock_contact_allowed": True,
            "final_manifest_pose_required": False,
            "continuous_stable_window_required": True}:
        raise ValueError("three_press_acceptance_contract")
    if value.get("scope") != {
            "active_soil": "unexcavated_foundation_stage_only",
            "active_rock_count": 3, "other_62_rocks_active": False,
            "full_depth_press": False, "full_lower_layer": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("three_press_scope_contract")
    return value


class RuntimeBridge:
    def __init__(self, binary: Path, model: Path, count: int):
        self.count = count
        self.ids = [f"r1-rock-{index:03d}" for index in range(count)]
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@READY "):
            self._fail("ready", line)
        self.ready = json.loads(line[7:])
        if (self.ready.get("rock_count") != count
                or self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", *self.ids]):
            self.close(kill=True)
            raise RuntimeError("three_press_bridge_ready_contract")
        self._write("init_default\n")
        self.state = self._read_state("init")

    def _write(self, value: str) -> None:
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def _fail(self, stage: str, stdout: str = "") -> None:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise RuntimeError(f"three_press_bridge_{stage}_failed:{stdout}:{stderr}")

    def _read_state(self, stage: str) -> dict:
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            self._fail(stage, line)
        state = json.loads(line[7:])
        if len(state.get("rocks", [])) != self.count:
            self._fail(stage, line)
        return state

    def step(self, dt: float, substeps: int, wrenches: dict[str, dict]) -> dict:
        fields = ["step", f"{float(self.state['time_s']):.17g}", f"{dt:.17g}",
                  str(substeps), "0", "0", "0", "0", str(self.count + 1)]
        for body_name in ["bucket", *self.ids]:
            wrench = wrenches[body_name]
            fields.append(body_name)
            for key in ("force", "torque", "point"):
                fields.extend(f"{float(value):.17g}" for value in wrench[key])
        self._write(" ".join(fields) + "\n")
        self.state = self._read_state("step")
        return self.state

    def close(self, kill: bool = False) -> None:
        if self.process.poll() is None:
            if kill:
                self.process.kill()
            else:
                self._write("close\n")
        stdout, stderr = self.process.communicate(timeout=20)
        if not kill and (self.process.returncode or stdout.strip() or stderr.strip()):
            raise RuntimeError(
                f"three_press_bridge_close_failed:{self.process.returncode}:"
                f"{stdout}:{stderr}")


class MultiRockFixture:
    def __init__(self, config: dict, repository: Path, records: list[dict],
                 particle_q: np.ndarray, device: str):
        self.records = records
        self.ids = [f"r1-rock-{index:03d}" for index in range(len(records))]
        self.device = device
        material = config["mpm"]["material"]
        spacing = float(config["mpm"]["particle_spacing_m"])
        radius = 0.45 * spacing
        surface_extent = float(particle_q[:, 2].max()) + radius
        clearance = float(config["construction"]["start_surface_clearance_m"])
        collider = newton.ModelBuilder()
        self.body_indices = {}
        self.start_positions = []
        self.faces = []
        self.local_vertices = []
        for index, record in enumerate(records):
            vertices, indices = _read_obj(
                repository / "assets/v4/mixed-material-r0" /
                f"{record['prototype_id']}.obj")
            rotation = _rotation_matrix(record["orientation_quaternion_wxyz"])
            local = ((np.asarray(vertices, dtype=np.float64)
                      * float(record["equivalent_diameter_m"])) @ rotation.T)
            position = np.asarray(record["position_world_m"], dtype=np.float64)
            position[2] = surface_extent + clearance - float(local[:, 2].min())
            self.start_positions.append(position)
            self.local_vertices.append(local)
            self.faces.append(np.asarray(indices, dtype=np.int32).reshape(-1, 3))
            inertia = np.asarray(record["inertia_about_com_body_kg_m2"], dtype=np.float32)
            body = int(collider.add_body(
                xform=_make_transform(position, record["orientation_quaternion_wxyz"]),
                com=wp.vec3(0.0), inertia=wp.mat33(inertia),
                mass=float(record["mass_kg"]), label=self.ids[index],
                lock_inertia=True))
            self.body_indices[self.ids[index]] = body
            collider.add_shape_mesh(
                body, mesh=newton.Mesh(vertices, indices, compute_inertia=False,
                                       is_solid=True),
                scale=wp.vec3(float(record["equivalent_diameter_m"])),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(material["friction"]),
                    has_particle_collision=True),
                label=self.ids[index] + "-mpm-collider")
        collider.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(material["friction"])))
        self.collider_model = collider.finalize(device=device)

        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        self.particle_mass = float(config["mpm"]["particle_density_kg_m3"]) * spacing ** 3
        sand.add_particles(
            pos=particle_q.tolist(), vel=np.zeros_like(particle_q).tolist(),
            mass=[self.particle_mass] * len(particle_q), radius=[radius] * len(particle_q),
            custom_attributes={
                "mpm:friction": [float(material["friction"])] * len(particle_q)})
        self.model = sand.finalize(device=device)
        self.model.set_gravity(tuple(config["mpm"]["gravity_m_s2"]))
        for name, value in material.items():
            if hasattr(self.model.mpm, name):
                getattr(self.model.mpm, name).fill_(float(value))
        options = SolverImplicitMPM.Config()
        options.voxel_size = float(config["mpm"]["voxel_size_m"])
        options.grid_type = "fixed"
        options.grid_padding = 12
        options.max_active_cell_count = 1 << 18
        options.max_iterations = 50
        options.tolerance = 1e-4
        options.transfer_scheme = "pic"
        options.integration_scheme = "pic"
        options.strain_basis = "P0"
        options.critical_fraction = 0.0
        options.collider_basis = "S2"
        options.velocity_basis = "Q1"
        options.collider_velocity_mode = "forward"
        self.solver = SolverImplicitMPM(self.model, config=options)
        self.solver.setup_collider(model=self.collider_model)
        self.state = self.model.state()
        collider_state = self.collider_model.state()
        self.state.body_q = wp.clone(collider_state.body_q)
        self.state.body_qd = wp.clone(collider_state.body_qd)
        self.state.body_f = wp.zeros_like(collider_state.body_f)
        self.assign(self.start_positions, [np.zeros(3)] * len(records))
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy().copy()
        self.collider_ids = {
            rock_id: np.flatnonzero(body_map == body)
            for rock_id, body in self.body_indices.items()}
        if any(ids.size != 1 for ids in self.collider_ids.values()):
            raise RuntimeError("three_press_collider_mapping")

    def assign(self, positions: list[np.ndarray], velocities: list[np.ndarray],
               angular: list[np.ndarray] | None = None,
               quaternions: list[list[float]] | None = None) -> None:
        angular = angular or [np.zeros(3)] * len(self.records)
        quaternions = quaternions or [record["orientation_quaternion_wxyz"]
                                      for record in self.records]
        self.state.body_q.assign(wp.array([
            _make_transform(position, quaternion)
            for position, quaternion in zip(positions, quaternions)
        ], dtype=wp.transform, device=self.device))
        self.state.body_qd.assign(wp.array([
            wp.spatial_vector(float(linear[0]), float(linear[1]), float(linear[2]),
                              float(omega[0]), float(omega[1]), float(omega[2]))
            for linear, omega in zip(velocities, angular)
        ], dtype=wp.spatial_vector, device=self.device))

    def assign_bridge(self, state: dict) -> None:
        self.assign(
            [np.asarray(rock["position"], dtype=np.float64) for rock in state["rocks"]],
            [np.asarray(rock["linear_velocity_world_m_s"], dtype=np.float64)
             for rock in state["rocks"]],
            [np.asarray(rock["angular_velocity_world_rad_s"], dtype=np.float64)
             for rock in state["rocks"]],
            [rock["qpos"][3:7] for rock in state["rocks"]])

    def step(self, dt: float, origins: list[np.ndarray]) -> tuple[dict, dict]:
        self.solver.step(self.state, self.state, contacts=None, control=None, dt=dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        wrenches = {}
        total_samples = 0
        for index, rock_id in enumerate(self.ids):
            mask = np.isin(ids, self.collider_ids[rock_id])
            mask &= np.isfinite(impulse).all(axis=1)
            mask &= np.linalg.norm(impulse, axis=1) > 1e-12
            selected = impulse[mask]
            selected_position = position[mask]
            total_samples += int(np.count_nonzero(mask))
            if selected.size:
                force = np.sum(selected, axis=0, dtype=np.float64) / dt
                torque = np.sum(np.cross(
                    selected_position - origins[index], selected), axis=0,
                    dtype=np.float64) / dt
            else:
                force, torque = np.zeros(3), np.zeros(3)
            wrenches[rock_id] = {
                "force": force, "torque": torque, "point": origins[index]}
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        speed = np.linalg.norm(qd, axis=1)
        return wrenches, {
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "contact_samples": total_samples,
            "particle_max_speed_m_s": float(speed.max()),
            "particle_p99_speed_m_s": float(np.quantile(speed, 0.99)),
            "particle_minimum_z_m": float(q[:, 2].min()),
        }


def _zero(point) -> dict:
    return {"force": np.zeros(3), "torque": np.zeros(3),
            "point": np.asarray(point, dtype=np.float64)}


def _angle_between(first: list[float], second: list[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    value = min(1.0, max(-1.0, abs(float(np.dot(a, b)))))
    return 2.0 * math.acos(value)


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("three_press_output_exists")
    config = load_config(config_path, repository)
    manifest_path = _resolve(repository, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
    records = [by_id[item["manifest_rock_id"]] for item in config["targets"]]
    particle_q = _foundation_particles(config, repository)
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    fixture = MultiRockFixture(config, repository, records, particle_q, args.device)
    dt = float(config["probe"]["output_dt_s"])
    positions = [position.copy() for position in fixture.start_positions]
    velocities = [np.zeros(3) for _ in records]
    kinematic_contact_samples = 0
    kinematic_peak_particle_speed = 0.0
    kinematic_steps = 0

    for _ in range(int(config["construction"]["foundation_presettle_frames"])):
        fixture.assign(positions, velocities)
        _, metrics = fixture.step(dt, positions)
        kinematic_steps += 1
        kinematic_contact_samples += metrics["contact_samples"]
        kinematic_peak_particle_speed = max(
            kinematic_peak_particle_speed, metrics["particle_max_speed_m_s"])
        if not metrics["finite"]:
            raise RuntimeError("three_press_nonfinite_during_presettle")

    press_summary = []
    for target_index, target in enumerate(config["targets"]):
        distance = float(target["press_distance_m"])
        frames = int(target["press_frames"])
        speed = distance / (frames * dt)
        start = positions[target_index].copy()
        samples = 0
        for frame in range(frames):
            positions[target_index] = start.copy()
            positions[target_index][2] -= distance * (frame + 1) / frames
            velocities = [np.zeros(3) for _ in records]
            velocities[target_index][2] = -speed
            fixture.assign(positions, velocities)
            _, metrics = fixture.step(dt, positions)
            kinematic_steps += 1
            samples += metrics["contact_samples"]
            kinematic_contact_samples += metrics["contact_samples"]
            kinematic_peak_particle_speed = max(
                kinematic_peak_particle_speed, metrics["particle_max_speed_m_s"])
            if not metrics["finite"]:
                raise RuntimeError("three_press_nonfinite_during_press")
        velocities[target_index][:] = 0.0
        press_summary.append({
            "manifest_rock_id": target["manifest_rock_id"],
            "runtime_rock_id": f"r1-rock-{target_index:03d}",
            "press_distance_m": distance, "press_frames": frames,
            "press_speed_m_s": speed, "contact_samples": samples,
            "release_position_world_m": positions[target_index].tolist(),
        })

    initial_intersections = []
    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            intersects, gap = sat_axis_gap(
                fixture.local_vertices[first] + positions[first], fixture.faces[first],
                fixture.local_vertices[second] + positions[second], fixture.faces[second])
            if intersects:
                initial_intersections.append([first, second, gap])

    release_records = []
    for record, position in zip(records, positions):
        item = dict(record)
        item["position_world_m"] = position.tolist()
        release_records.append(item)
    release_manifest = {"rocks": release_records}
    capacity_path = _resolve(repository, config["capacity_bridge"]["config_path"])
    bridge = None
    release_trace = []
    stable_count = 0
    release_steps = 0
    release_contact_samples = 0
    maximum_rock_linear = 0.0
    maximum_rock_angular = 0.0
    maximum_particle_speed = 0.0
    maximum_forbidden = 0
    all_finite = True
    center_inside = True
    visual = json.loads(_resolve(
        repository, config["unexcavated_soil"]["visual_config_path"]
    ).read_text(encoding="utf-8"))
    outer = np.asarray(visual["mound_envelope"]["outer_xy_bounds_m"], dtype=np.float64)
    release_initial_positions = [position.copy() for position in positions]
    release_initial_quaternions = [record["orientation_quaternion_wxyz"]
                                   for record in records]
    model_manifest = None
    bridge_manifest = None
    compile_command = None
    with tempfile.TemporaryDirectory(prefix="three-rock-release-") as temporary:
        work = Path(temporary)
        model = work / "release.xml"
        model_manifest = _build_provisional_model(
            capacity_path, repository, release_manifest, model)
        source_path = work / "bridge.cc"
        source, bridge_manifest = render_capacity_bridge(capacity_path, repository)
        source_path.write_text(source, encoding="utf-8")
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source_path, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        bridge = RuntimeBridge(binary, model, len(records))
        try:
            initial_bridge_positions = [np.asarray(rock["position"], dtype=np.float64)
                                        for rock in bridge.state["rocks"]]
            if any(not np.allclose(actual, expected, rtol=0.0, atol=1e-6)
                   for actual, expected in zip(initial_bridge_positions, positions)):
                raise RuntimeError("three_press_release_position_handoff")
            for frame in range(int(config["probe"]["maximum_release_frames"])):
                state = bridge.state
                fixture.assign_bridge(state)
                origins = [np.asarray(rock["position"], dtype=np.float64)
                           for rock in state["rocks"]]
                measured, particle = fixture.step(dt, origins)
                wrenches = {"bucket": _zero(state["bucket_origin"]), **measured}
                state = bridge.step(
                    dt, int(config["probe"]["mujoco_internal_substeps"]), wrenches)
                release_steps += 1
                release_contact_samples += particle["contact_samples"]
                linear_speeds = [float(np.linalg.norm(
                    rock["linear_velocity_world_m_s"])) for rock in state["rocks"]]
                angular_speeds = [float(np.linalg.norm(
                    rock["angular_velocity_world_rad_s"])) for rock in state["rocks"]]
                max_linear = max(linear_speeds)
                max_angular = max(angular_speeds)
                maximum_rock_linear = max(maximum_rock_linear, max_linear)
                maximum_rock_angular = max(maximum_rock_angular, max_angular)
                maximum_particle_speed = max(
                    maximum_particle_speed, particle["particle_max_speed_m_s"])
                maximum_forbidden = max(
                    maximum_forbidden, int(state.get("forbidden_contact_count", 0)))
                all_finite &= bool(particle["finite"] and _finite_tree(state))
                for rock in state["rocks"]:
                    point = np.asarray(rock["position"], dtype=np.float64)
                    center_inside &= bool(
                        outer[0, 0] <= point[0] <= outer[1, 0]
                        and outer[0, 1] <= point[1] <= outer[1, 1]
                        and point[2] >= float(config["probe"]["minimum_world_z_m"]))
                stable = (
                    max_linear <= float(config["probe"][
                        "maximum_stable_rock_linear_speed_m_s"])
                    and max_angular <= float(config["probe"][
                        "maximum_stable_rock_angular_speed_rad_s"])
                    and particle["particle_max_speed_m_s"] <= float(config["probe"][
                        "maximum_stable_particle_speed_m_s"])
                    and particle["particle_p99_speed_m_s"] <= float(config["probe"][
                        "maximum_stable_particle_p99_speed_m_s"]))
                stable_count = stable_count + 1 if stable else 0
                fastest = int(np.argmax(linear_speeds))
                if frame == 0 or (frame + 1) % 30 == 0 or stable_count == int(
                        config["probe"]["stable_window_frames"]):
                    release_trace.append({
                        "frame": frame, "stable_window_frames": stable_count,
                        "max_rock_linear_speed_m_s": max_linear,
                        "max_rock_angular_speed_rad_s": max_angular,
                        "fastest_runtime_rock_id": f"r1-rock-{fastest:03d}",
                        **particle})
                if stable_count >= int(config["probe"]["stable_window_frames"]):
                    break
        finally:
            bridge.close(kill=not all_finite)

    final_state = bridge.state
    final_rocks = []
    for index, rock in enumerate(final_state["rocks"]):
        final_position = np.asarray(rock["position"], dtype=np.float64)
        final_rocks.append({
            "manifest_rock_id": records[index]["rock_id"],
            "runtime_rock_id": rock["rock_id"],
            "final_position_world_m": final_position.tolist(),
            "translation_from_release_m": float(np.linalg.norm(
                final_position - release_initial_positions[index])),
            "rotation_from_release_rad": _angle_between(
                release_initial_quaternions[index], rock["qpos"][3:7]),
            "final_linear_speed_m_s": float(np.linalg.norm(
                rock["linear_velocity_world_m_s"])),
            "final_angular_speed_rad_s": float(np.linalg.norm(
                rock["angular_velocity_world_rad_s"])),
        })
    checks = {
        "upstream_single_press_gate_bound": True,
        "unequal_short_press_distances_executed":
            [item["press_distance_m"] for item in press_summary] == [0.012, 0.008, 0.016],
        "release_geometry_has_no_initial_rock_intersection": not initial_intersections,
        "kinematic_phase_finite": True,
        "release_state_finite": all_finite,
        "release_contact_observed": release_contact_samples > 0,
        "rock_speed_safety_bound": maximum_rock_linear <= float(
            config["probe"]["maximum_rock_linear_speed_m_s"]),
        "rock_angular_safety_bound": maximum_rock_angular <= float(
            config["probe"]["maximum_rock_angular_speed_rad_s"]),
        "particle_speed_safety_bound": maximum_particle_speed <= float(
            config["probe"]["maximum_particle_speed_m_s"]),
        "forbidden_contacts_zero": maximum_forbidden <= int(
            config["probe"]["maximum_forbidden_contact_count"]),
        "rock_centers_remain_in_foundation_bounds": center_inside,
        "continuous_free_stable_window_reached": stable_count >= int(
            config["probe"]["stable_window_frames"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("three_lower_rocks_variable_press_free_settle_passed"
                   if all(checks.values()) else
                   "three_lower_rocks_variable_press_free_settle_failed"),
        "checks": checks,
        "construction": {
            "presses": press_summary,
            "kinematic_steps": kinematic_steps,
            "kinematic_contact_samples": kinematic_contact_samples,
            "kinematic_peak_particle_speed_m_s": kinematic_peak_particle_speed,
            "release_initial_sat_intersections": initial_intersections,
        },
        "release": {
            "steps": release_steps, "wall_simulated_s": release_steps * dt,
            "stable_window_frames": stable_count,
            "contact_samples": release_contact_samples,
            "maximum_rock_linear_speed_m_s": maximum_rock_linear,
            "maximum_rock_angular_speed_rad_s": maximum_rock_angular,
            "maximum_particle_speed_m_s": maximum_particle_speed,
            "maximum_forbidden_contact_count": maximum_forbidden,
            "final_rocks": final_rocks, "trace": release_trace,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "physical_manifest_sha256": sha256(manifest_path),
            "single_press_report_sha256": sha256(_resolve(
                repository, config["single_press_gate"]["report_path"])),
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
            "newton_solver_step_calls": kinematic_steps + release_steps,
            "mujoco_step_calls": release_steps,
            "wall_seconds": time.perf_counter() - started,
        },
        "acceptance": config["acceptance"],
        "scope": config["scope"],
        "stable_scene_claimed": False,
        "next_gate": "review_three_rock_free_settle_before_expanding_lower_layer",
    }
    if not _finite_tree(report):
        raise RuntimeError("three_press_nonfinite_report")
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
        "construction": report["construction"],
        "release": {key: value for key, value in report["release"].items()
                    if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
