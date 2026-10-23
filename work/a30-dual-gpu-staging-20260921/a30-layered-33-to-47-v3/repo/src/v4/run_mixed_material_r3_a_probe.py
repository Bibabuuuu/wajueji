#!/usr/bin/env python3
"""Run the isolated R3-A one-rock MuJoCo/Newton MPM coupling probe.

This is deliberately a qualification fixture, not the official excavation
scene.  It derives a one-rock MJCF and bridge from frozen R1 inputs, places the
rock above a small MPM matrix, and exchanges one named rock wrench per outer
step.  The bucket is intentionally absent from the MPM collider model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from build_mixed_coupled_bridge_r3_a import (
    sha256,
    validate_config,
    write_bridge,
)
from mixed_material_r1_mechanism import build_candidate


REPORT_SCHEMA = "excavation_mixed_material_r3_a_probe_report_v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compile_bridge(cxx: str, source: Path, output: Path, include_dirs,
                    library_dir: Path) -> list[str]:
    command = [cxx, "-O2", "-std=c++17", "-Wall", "-Wextra"]
    command += ["-I" + str(Path(value).resolve()) for value in include_dirs]
    command += [
        str(source.resolve()),
        "-L" + str(library_dir.resolve()),
        "-Wl,-rpath," + str(library_dir.resolve()),
        "-l:libmujoco.so.2.3.8",
        "-o", str(output.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("r3_a_bridge_compile_failed:" + completed.stderr)
    return command


def _patch_rock_pose(model_path: Path, position, quaternion) -> None:
    tree = ET.parse(model_path)
    matches = [element for element in tree.getroot().iter("body")
               if element.get("name") == "r1-rock-000"]
    if len(matches) != 1:
        raise ValueError("r3_a_one_rock_body_contract")
    matches[0].set("pos", " ".join(f"{float(value):.17g}" for value in position))
    matches[0].set("quat", " ".join(f"{float(value):.17g}" for value in quaternion))
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode", xml_declaration=False)
    with model_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    indices: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError("r3_a_obj_vertex_contract")
            vertices.append([float(value) for value in fields[1:]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("r3_a_obj_triangles_only")
            indices.extend(int(value.split("/")[0]) - 1 for value in fields[1:])
    vertex_array = np.asarray(vertices, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not index_array.size:
        raise ValueError("r3_a_obj_empty")
    if np.min(index_array) < 0 or np.max(index_array) >= len(vertex_array):
        raise ValueError("r3_a_obj_index_contract")
    return vertex_array, index_array


def _rotation_wxyz(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid_rock_quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _correct_velocity(state: dict, previous_force: np.ndarray,
                      previous_torque: np.ndarray, dt: float, mass: float,
                      inertia_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear = np.asarray(state["rock_linear_velocity_world_m_s"], dtype=np.float64)
    angular = np.asarray(state["rock_angular_velocity_world_rad_s"], dtype=np.float64)
    rotation = _rotation_wxyz(state["rock_qpos"][3:7])
    inverse_world_inertia = rotation @ np.linalg.inv(inertia_local) @ rotation.T
    return (
        linear - dt * previous_force / mass,
        angular - dt * (inverse_world_inertia @ previous_torque),
    )


class Bridge:
    def __init__(self, binary: Path, model: Path, expected_ready_schema: str):
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready_line = self.process.stdout.readline().rstrip("\n")
        if not ready_line.startswith("@READY "):
            self._fail("ready", ready_line)
        self.ready = json.loads(ready_line[7:])
        if (self.ready.get("schema") != expected_ready_schema or
                self.ready.get("rock_count") != 1 or
                self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", "r1-rock-000"]):
            self.close(kill=True)
            raise RuntimeError("r3_a_bridge_ready_contract")
        self._write("init_default\n")
        self.initial = self._read_state("init")

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("r3_a_bridge_stdin_closed")
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def _fail(self, stage: str, stdout: str = "") -> None:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise RuntimeError(f"r3_a_bridge_{stage}_failed:{stdout}:{stderr}")

    def _read_state(self, stage: str) -> dict:
        if self.process.stdout is None:
            self._fail(stage)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            self._fail(stage, line)
        return json.loads(line[7:])

    def step(self, dt: float, substeps: int, force: np.ndarray,
             torque: np.ndarray, point: np.ndarray, apply_wrench: bool) -> dict:
        # The caller tracks time from the most recently returned state.
        time_s = self.last_state["time_s"] if hasattr(self, "last_state") else self.initial["time_s"]
        fields = ["step", f"{float(time_s):.17g}", f"{dt:.17g}", str(substeps),
                  "0", "0", "0", "0", "1" if apply_wrench else "0"]
        if apply_wrench:
            fields.append("r1-rock-000")
            fields += [f"{float(value):.17g}" for value in force]
            fields += [f"{float(value):.17g}" for value in torque]
            fields += [f"{float(value):.17g}" for value in point]
        self._write(" ".join(fields) + "\n")
        self.last_state = self._read_state("step")
        return self.last_state

    def close(self, kill: bool = False) -> None:
        if self.process.poll() is None:
            if kill:
                self.process.kill()
            else:
                self._write("close\n")
        stdout, stderr = self.process.communicate(timeout=10)
        if not kill and (self.process.returncode or stdout.strip() or stderr.strip()):
            raise RuntimeError(
                f"r3_a_bridge_close_failed:{self.process.returncode}:{stdout}:{stderr}")


def _make_transform(position, quaternion_wxyz):
    w, x, y, z = [float(value) for value in quaternion_wxyz]
    return wp.transform(wp.vec3(*[float(value) for value in position]),
                        wp.quat(x, y, z, w))


def _assign_body_state(state, bridge_state: dict, linear: np.ndarray,
                       angular: np.ndarray, device: str) -> None:
    state.body_q.assign(wp.array([
        _make_transform(bridge_state["rock_qpos"][:3],
                        bridge_state["rock_qpos"][3:7])
    ], dtype=wp.transform, device=device))
    state.body_qd.assign(wp.array([
        wp.spatial_vector(float(linear[0]), float(linear[1]), float(linear[2]),
                          float(angular[0]), float(angular[1]), float(angular[2]))
    ], dtype=wp.spatial_vector, device=device))


class MpmFixture:
    def __init__(self, config: dict, repository: Path, device: str,
                 initial_bridge_state: dict):
        matrix = config["matrix"]
        frozen = config["frozen_inputs"]
        rock_position = initial_bridge_state["rock_qpos"][:3]
        rock_quaternion = initial_bridge_state["rock_qpos"][3:7]
        inertia = np.asarray(frozen["rock_inertia_about_com_kg_m2"], dtype=np.float64)
        vertices, indices = _read_obj(repository / frozen["rock_asset_path"])

        collider_builder = newton.ModelBuilder()
        self.rock_body = collider_builder.add_body(
            xform=_make_transform(rock_position, rock_quaternion),
            com=wp.vec3(0.0),
            inertia=wp.mat33(inertia.astype(np.float32)),
            mass=float(frozen["rock_mass_kg"]),
            label="r1-rock-000",
            lock_inertia=True,
        )
        mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True)
        self.rock_shape = collider_builder.add_shape_mesh(
            self.rock_body,
            mesh=mesh,
            scale=wp.vec3(float(frozen["rock_equivalent_diameter_m"])),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(matrix["material"]["friction"])),
            label="r1-rock-000-mpm-collider",
        )
        collider_builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(matrix["material"]["friction"])))
        self.collider_model = collider_builder.finalize(device=device)

        center = np.asarray(matrix["center_xy_m"], dtype=np.float64)
        size = np.asarray(matrix["size_xyz_m"], dtype=np.float64)
        lower = np.asarray([center[0] - 0.5 * size[0],
                            center[1] - 0.5 * size[1], 0.0], dtype=np.float64)
        spacing = float(matrix["particle_spacing_m"])
        dimensions = np.rint(size / spacing).astype(int) + 1
        if not np.allclose((dimensions - 1) * spacing, size, atol=1.0e-12):
            raise ValueError("r3_a_matrix_spacing_contract")

        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        particle_count = int(np.prod(dimensions, dtype=np.int64))
        target_mass = float(np.prod(size) * float(matrix["density_kg_m3"]))
        sand_builder.add_particle_grid(
            pos=wp.vec3(lower), rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=int(dimensions[0]), dim_y=int(dimensions[1]),
            dim_z=int(dimensions[2]),
            cell_x=spacing, cell_y=spacing, cell_z=spacing,
            mass=target_mass / particle_count,
            jitter=0.0, radius_mean=0.45 * spacing,
            custom_attributes={"mpm:friction": float(matrix["material"]["friction"])},
        )
        self.model = sand_builder.finalize(device=device)
        self.model.set_gravity(tuple(float(value) for value in matrix["gravity_m_s2"]))
        for name, value in matrix["material"].items():
            if hasattr(self.model.mpm, name):
                getattr(self.model.mpm, name).fill_(float(value))

        options = SolverImplicitMPM.Config()
        options.voxel_size = float(matrix["voxel_size_m"])
        options.grid_type = "fixed"
        options.grid_padding = 12
        options.max_active_cell_count = 1 << 16
        options.max_iterations = 50
        options.tolerance = 1.0e-4
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
        self.device = device
        _assign_body_state(self.state, initial_bridge_state, np.zeros(3), np.zeros(3), device)
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy()
        self.rock_collider_ids = np.flatnonzero(body_map == self.rock_body)
        if self.rock_collider_ids.size != 1:
            raise RuntimeError(
                "r3_a_rock_collider_map:" + json.dumps(body_map.tolist()))
        self.initial_particle_q = self.state.particle_q.numpy().copy()
        self.initial_particle_qd = self.state.particle_qd.numpy().copy()
        self.initial_mass_kg = float(np.sum(self.model.particle_mass.numpy(), dtype=np.float64))
        self.target_mass_kg = target_mass
        self.particle_count = int(self.model.particle_count)

    def assign_rock(self, bridge_state: dict, linear: np.ndarray,
                    angular: np.ndarray) -> None:
        _assign_body_state(self.state, bridge_state, linear, angular, self.device)

    def step_and_collect(self, dt: float, rock_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        self.solver.step(self.state, self.state, contacts=None, control=None, dt=dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        collider_id = collider_ids.numpy()
        mask = np.isin(collider_id, self.rock_collider_ids)
        mask &= np.isfinite(impulse).all(axis=1)
        mask &= np.linalg.norm(impulse, axis=1) > 1.0e-12
        selected = impulse[mask]
        selected_position = position[mask]
        force = np.sum(selected, axis=0, dtype=np.float64) / dt if selected.size else np.zeros(3)
        if selected.size:
            torque_impulse = np.sum(
                np.cross(selected_position - rock_position, selected), axis=0,
                dtype=np.float64)
            torque = torque_impulse / dt
        else:
            torque = np.zeros(3)
        return force, torque, int(np.count_nonzero(mask))

    def metrics(self) -> dict:
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        return {
            "particle_count": int(q.shape[0]),
            "total_mass_kg": float(np.sum(self.model.particle_mass.numpy(), dtype=np.float64)),
            "target_mass_kg": self.target_mass_kg,
            "target_mass_error_kg": float(
                np.sum(self.model.particle_mass.numpy(), dtype=np.float64) -
                self.target_mass_kg),
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "minimum_particle_z_m": float(np.min(q[:, 2])),
            "maximum_particle_speed_m_s": float(np.max(np.linalg.norm(qd, axis=1))),
            "particle_q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            "particle_qd_sha256": hashlib.sha256(qd.tobytes()).hexdigest(),
        }

    def reset_audit(self) -> dict:
        self.state.particle_q.assign(wp.array(
            self.initial_particle_q, dtype=wp.vec3, device=self.device))
        self.state.particle_qd.assign(wp.array(
            self.initial_particle_qd, dtype=wp.vec3, device=self.device))
        self.solver.reset(self.state)
        wp.synchronize()
        mpm = self.state.mpm
        identity = np.eye(3, dtype=np.float32)
        fields = {
            "particle_elastic_strain_identity": np.array_equal(
                mpm.particle_elastic_strain.numpy(),
                np.broadcast_to(identity, mpm.particle_elastic_strain.numpy().shape)),
            "particle_transform_identity": np.array_equal(
                mpm.particle_transform.numpy(),
                np.broadcast_to(identity, mpm.particle_transform.numpy().shape)),
            "particle_qd_grad_zero": not np.any(mpm.particle_qd_grad.numpy()),
            "particle_stress_zero": not np.any(mpm.particle_stress.numpy()),
            "particle_Jp_one": np.all(mpm.particle_Jp.numpy() == 1.0),
            "particle_q_exact": np.array_equal(
                self.state.particle_q.numpy(), self.initial_particle_q),
            "particle_qd_exact": np.array_equal(
                self.state.particle_qd.numpy(), self.initial_particle_qd),
        }
        return {"checks": {name: bool(value) for name, value in fields.items()},
                "passed": bool(all(fields.values()))}


def _run_episode(config: dict, repository: Path, binary: Path, model: Path,
                 device: str, frames: int, feedback: bool) -> dict:
    probe = config["probe"]
    frozen = config["frozen_inputs"]
    dt = float(probe["output_dt_s"])
    substeps = int(probe["mujoco_internal_substeps"])
    mass = float(frozen["rock_mass_kg"])
    inertia = np.asarray(frozen["rock_inertia_about_com_kg_m2"], dtype=np.float64)
    bridge = Bridge(binary, model, config["bridge"]["ready_schema"])
    fixture = MpmFixture(config, repository, device, bridge.initial)
    previous_force = np.zeros(3, dtype=np.float64)
    previous_torque = np.zeros(3, dtype=np.float64)
    peak_force = 0.0
    peak_torque = 0.0
    contact_frames = 0
    contact_samples = 0
    maximum_closure = 0.0
    maximum_forbidden = 0
    rock_z = [float(bridge.initial["rock_position"][2])]
    trace = []
    start = time.perf_counter()
    try:
        for frame in range(frames):
            point = np.asarray(
                bridge.last_state["rock_position"] if hasattr(bridge, "last_state")
                else bridge.initial["rock_position"], dtype=np.float64)
            state = bridge.step(
                dt, substeps, previous_force, previous_torque, point,
                apply_wrench=feedback,
            )
            linear, angular = _correct_velocity(
                state,
                previous_force if feedback else np.zeros(3),
                previous_torque if feedback else np.zeros(3),
                dt, mass, inertia,
            )
            fixture.assign_rock(state, linear, angular)
            next_force, next_torque, samples = fixture.step_and_collect(
                dt, np.asarray(state["rock_position"], dtype=np.float64))
            previous_force = next_force
            previous_torque = next_torque
            force_norm = float(np.linalg.norm(next_force))
            torque_norm = float(np.linalg.norm(next_torque))
            peak_force = max(peak_force, force_norm)
            peak_torque = max(peak_torque, torque_norm)
            contact_samples += samples
            contact_frames += int(samples > 0)
            maximum_closure = max(maximum_closure, float(state["closure_m"]))
            maximum_forbidden = max(maximum_forbidden,
                                    int(state["forbidden_contact_count"]))
            rock_z.append(float(state["rock_position"][2]))
            trace.append([
                frame + 1, float(state["time_s"]), rock_z[-1],
                *[float(value) for value in next_force],
                *[float(value) for value in next_torque], samples,
            ])
    finally:
        bridge.close()
    final_metrics = fixture.metrics()
    reset = fixture.reset_audit()
    trace_bytes = json.dumps(trace, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "feedback_enabled": feedback,
        "frames": frames,
        "simulated_time_s": frames * dt,
        "wall_seconds": time.perf_counter() - start,
        "initial_rock_z_m": rock_z[0],
        "final_rock_z_m": rock_z[-1],
        "minimum_rock_z_m": min(rock_z),
        "maximum_rock_z_m": max(rock_z),
        "rock_vertical_displacement_m": rock_z[-1] - rock_z[0],
        "peak_mpm_force_n": peak_force,
        "peak_mpm_torque_nm": peak_torque,
        "mpm_contact_frames": contact_frames,
        "mpm_contact_samples": contact_samples,
        "maximum_mechanism_closure_m": maximum_closure,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "trace": trace,
        "matrix": final_metrics,
        "reset_audit": reset,
    }


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    probe = config["probe"]
    default_frames = int(round(float(probe["duration_s"]) /
                               float(probe["output_dt_s"])))
    frames = default_frames if args.frames is None else int(args.frames)
    if frames < 1 or frames > default_frames:
        raise ValueError(f"frames_must_be_within_1_and_{default_frames}")

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mixed-material-r3-a-") as temporary:
        work = Path(temporary)
        generated_source = work / "mixed-coupled-bridge-r3-a.cc"
        source_manifest = write_bridge(config_path, repository, generated_source)
        bridge_binary = work / "mixed-coupled-bridge-r3-a"
        compile_command = _compile_bridge(
            args.cxx, generated_source, bridge_binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        model = work / "model.xml"
        r1_config = repository / config["frozen_inputs"]["r1_config_path"]
        model_manifest = build_candidate(
            repository / config["frozen_inputs"]["mechanism_model_path"],
            model, repository, Path(args.inventory), r1_config, "freefall")
        _patch_rock_pose(
            model, probe["rock_initial_position_world_m"],
            probe["rock_initial_orientation_quaternion_wxyz"])
        model_sha256 = sha256(model)
        coupled_first = _run_episode(
            config, repository, bridge_binary, model, args.device, frames, True)
        coupled_repeat = _run_episode(
            config, repository, bridge_binary, model, args.device, frames, True)
        uncoupled = _run_episode(
            config, repository, bridge_binary, model, args.device, frames, False)

    deterministic = (
        coupled_first["trace_sha256"] == coupled_repeat["trace_sha256"] and
        coupled_first["matrix"]["particle_q_sha256"] ==
        coupled_repeat["matrix"]["particle_q_sha256"] and
        coupled_first["matrix"]["particle_qd_sha256"] ==
        coupled_repeat["matrix"]["particle_qd_sha256"])
    support_gain = (
        coupled_first["final_rock_z_m"] - uncoupled["final_rock_z_m"])
    checks = {
        "coupled_finite": coupled_first["matrix"]["finite"],
        "repeat_finite": coupled_repeat["matrix"]["finite"],
        "uncoupled_finite": uncoupled["matrix"]["finite"],
        "nonzero_rock_mpm_contact": coupled_first["mpm_contact_frames"] > 0,
        "nonzero_rock_mpm_wrench": coupled_first["peak_mpm_force_n"] > 0.0,
        "feedback_changes_rock_support": support_gain > 0.01,
        "forbidden_contacts_zero": max(
            coupled_first["maximum_forbidden_contact_count"],
            coupled_repeat["maximum_forbidden_contact_count"],
            uncoupled["maximum_forbidden_contact_count"]) == 0,
        "closure_within_one_mm": max(
            coupled_first["maximum_mechanism_closure_m"],
            coupled_repeat["maximum_mechanism_closure_m"],
            uncoupled["maximum_mechanism_closure_m"]) <=
            float(probe["maximum_closure_m"]),
        "matrix_count_conserved": all(
            episode["matrix"]["particle_count"] ==
            coupled_first["matrix"]["particle_count"]
            for episode in (coupled_first, coupled_repeat, uncoupled)),
        "matrix_mass_conserved": all(
            episode["matrix"]["total_mass_kg"] ==
            coupled_first["matrix"]["total_mass_kg"]
            for episode in (coupled_first, coupled_repeat, uncoupled)),
        "matrix_target_mass_matched": all(
            abs(episode["matrix"]["target_mass_error_kg"]) <= 1.0e-5
            for episode in (coupled_first, coupled_repeat, uncoupled)),
        "deterministic_repeat": deterministic,
        "complete_reset": all(
            episode["reset_audit"]["passed"]
            for episode in (coupled_first, coupled_repeat, uncoupled)),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "r3_a_one_rock_coupling_probe_passed" if all(checks.values())
        else "r3_a_one_rock_coupling_probe_failed",
        "physics_qualified": bool(all(checks.values())),
        "checks": {name: bool(value) for name, value in checks.items()},
        "comparison": {
            "coupled_minus_uncoupled_final_rock_z_m": support_gain,
            "deterministic_repeat": deterministic,
        },
        "episodes": {
            "coupled_first": coupled_first,
            "coupled_repeat": coupled_repeat,
            "uncoupled_no_feedback": uncoupled,
        },
        "backend": {
            "python": platform.python_version(),
            "newton": newton.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "device_requested": args.device,
            "device_resolved": str(wp.get_device()),
        },
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "derived_model_sha256": model_sha256,
            "r1_model_candidate_manifest": model_manifest,
            "bridge_source_manifest": source_manifest,
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": [
                "<temporary>/mixed-coupled-bridge-r3-a"
                if value == str(bridge_binary.resolve()) else
                "<temporary>/mixed-coupled-bridge-r3-a.cc"
                if value == str(generated_source.resolve()) else value
                for value in compile_command
            ],
            "total_wall_seconds": time.perf_counter() - started,
        },
        "scope_limit": (
            "isolated one frozen R0 rock over a 0.5 m x 0.5 m x 0.25 m MPM "
            "matrix; per-rock wrench feedback only; bucket-to-MPM contact, multiple "
            "rocks, official 2.2 m envelope, excavation and training are not qualified"
        ),
    }
    _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "physics_qualified": report["physics_qualified"],
        "checks": report["checks"],
        "comparison": report["comparison"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
