#!/usr/bin/env python3
"""Run a short final-assembled dynamics smoke for the D-068 local geometry.

This is deliberately not the staged-settling gate.  It activates the final
carved MPM patch and all 22 rocks at frame zero to catch immediate coupling,
mapping, finite-state, or contact explosions before implementing the longer
foundation/lower/middle/top activation schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from build_mixed_material_capacity_preflight import (
    build_capacity_model,
    render_capacity_bridge,
    sha256,
)
from run_mixed_material_r3_a_probe import _compile_bridge, _make_transform, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mixed_scene_local_dynamics_smoke_config_v1"
REPORT_SCHEMA = "excavation_layered_mixed_scene_local_dynamics_smoke_report_v1"


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "final_assembled_short_mixed_dynamics_smoke_only"):
        raise ValueError("layered_smoke_config_schema_or_status")
    for group, fields in (("geometry", ("config", "manifest", "bank")),
                          ("capacity_bridge", ("config",))):
        binding = config.get(group, {})
        for field in fields:
            bound_path = repository / binding.get(field + "_path", "")
            expected = binding.get(field + "_sha256")
            if (not bound_path.is_file() or not isinstance(expected, str)
                    or sha256(bound_path) != expected):
                raise ValueError(f"layered_smoke_{group}_{field}_binding")
    manifest = json.loads((repository / config["geometry"]["manifest_path"])
                          .read_text(encoding="utf-8"))
    if (manifest.get("status") != "reduced_three_layer_geometry_candidate"
            or manifest.get("rock_count") != 22
            or manifest.get("stable_scene_claimed") is not False
            or manifest.get("physics_step_executed") is not False):
        raise ValueError("layered_smoke_geometry_scope")
    mpm = config.get("mpm", {})
    if (float(mpm.get("particle_spacing_m", -1)) != 0.025
            or float(mpm.get("particle_density_kg_m3", -1)) != 1670.0
            or float(mpm.get("voxel_size_m", -1)) != 0.05
            or mpm.get("gravity_m_s2") != [0.0, 0.0, -9.81]
            or mpm.get("material") != {
                "friction": 0.68, "young_modulus": 2000000.0,
                "poisson_ratio": 0.3, "yield_pressure": 100000.0,
                "yield_stress": 12000.0, "tensile_yield_ratio": 0.0025,
                "hardening": 0.0, "hardening_rate": 1.0,
                "softening_rate": 1.0, "dilatancy": 0.0,
                "viscosity": 0.0, "damping": 0.0,
            }):
        raise ValueError("layered_smoke_mpm_contract")
    if config.get("probe") != {
        "output_dt_s": 1.0 / 120.0, "mujoco_internal_substeps": 40,
        "frames": 12, "zero_command": [0.0, 0.0, 0.0, 0.0],
        "maximum_mechanism_closure_m": 0.001,
        "maximum_forbidden_contact_count": 0,
        "maximum_particle_speed_m_s": 5.0,
        "maximum_rock_linear_speed_m_s": 5.0,
        "maximum_rock_angular_speed_rad_s": 25.0,
    }:
        raise ValueError("layered_smoke_probe_contract")
    if config.get("scope") != {
        "all_final_particles_active_from_frame_zero": True,
        "all_22_rocks_active_from_frame_zero": True,
        "per_rock_mpm_wrench_feedback": True,
        "bucket_mpm_collision": False, "gravity_ramp": False,
        "staged_activation": False, "construction_walls": False,
        "short_smoke_only": True, "settling_qualified": False,
        "wall_free_hold_qualified": False, "formal_scene_qualified": False,
        "training": False,
    }:
        raise ValueError("layered_smoke_scope_expanded")
    return config


def _build_provisional_model(capacity_config: Path, repository: Path,
                             manifest: dict, output: Path) -> dict:
    base = output.with_name(output.stem + "-capacity-base.xml")
    base_manifest = build_capacity_model(
        capacity_config, repository, len(manifest["rocks"]), base)
    tree = ET.parse(base)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("layered_smoke_model_structure")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") == "capacity_rock_mesh":
            asset.remove(mesh)
    for index, rock in enumerate(manifest["rocks"]):
        mesh_name = f"layered_local_rock_mesh_{index:03d}"
        ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "file": str(repository / "assets/v4/mixed-material-r0"
                        / f"{rock['prototype_id']}.obj"),
            "scale": _numbers([rock["equivalent_diameter_m"]] * 3),
        })
        body = worldbody.find(f"body[@name='r1-rock-{index:03d}']")
        if body is None:
            raise ValueError("layered_smoke_model_rock_body")
        body.set("pos", _numbers(rock["position_world_m"]))
        body.set("quat", _numbers(rock["orientation_quaternion_wxyz"]))
        inertial, geom = body.find("inertial"), body.find("geom")
        if inertial is None or geom is None:
            raise ValueError("layered_smoke_model_rock_structure")
        inertia = np.asarray(rock["inertia_about_com_body_kg_m2"], dtype=float)
        inertial.set("mass", f"{float(rock['mass_kg']):.17g}")
        inertial.set("fullinertia", _numbers([
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
        ]))
        geom.set("mesh", mesh_name)
    resource_manifest = _make_model_resources_portable(root, repository)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    base.unlink()
    return {
        "base_capacity_model_manifest": base_manifest,
        "model_sha256": sha256(output),
        "rock_count": len(manifest["rocks"]),
        "model_resources": resource_manifest,
        "provisional_geometry_only_input": True,
        "stable_scene_claimed": False,
    }


def _make_model_resources_portable(root: ET.Element,
                                   repository: Path) -> dict:
    """Bind MuJoCo file assets to this repository or an offline bundle."""
    compiler = root.find("compiler")
    asset = root.find("asset")
    if compiler is None or asset is None:
        raise ValueError("layered_smoke_model_resource_structure")
    mesh_override = os.environ.get("EXCAVATOR_MUJOCO_MESH_DIR")
    texture_override = os.environ.get("EXCAVATOR_MUJOCO_TEXTURE_DIR")
    if bool(mesh_override) != bool(texture_override):
        raise ValueError("layered_smoke_model_resource_override_pair")
    if mesh_override:
        mesh_dir = Path(mesh_override).resolve()
        texture_dir = Path(texture_override).resolve()
        if not mesh_dir.is_dir() or not texture_dir.is_dir():
            raise ValueError("layered_smoke_model_resource_override_missing")
        compiler.set("meshdir", str(mesh_dir))
        compiler.set("texturedir", str(texture_dir))
    else:
        mesh_dir = Path(compiler.get("meshdir", "")).resolve()
        texture_dir = Path(compiler.get("texturedir", "")).resolve()

    rewritten = 0
    checked = 0
    for kind, base_dir in (("mesh", mesh_dir), ("texture", texture_dir),
                           ("hfield", mesh_dir)):
        for item in asset.findall(kind):
            raw = item.get("file")
            if not raw:
                continue
            path = Path(raw)
            if path.is_absolute():
                parts = path.parts
                if "assets" in parts:
                    relative = Path(*parts[parts.index("assets"):])
                    candidate = (repository / relative).resolve()
                    if candidate.is_file() and candidate != path:
                        item.set("file", str(candidate))
                        path = candidate
                        rewritten += 1
            resolved = path if path.is_absolute() else base_dir / path
            if not resolved.is_file():
                raise ValueError(
                    f"layered_smoke_model_resource_missing:{kind}:{raw}")
            checked += 1
    return {
        "offline_override_applied": bool(mesh_override),
        "absolute_repository_paths_rewritten": rewritten,
        "external_files_checked": checked,
    }


class Bridge:
    def __init__(self, binary: Path, model: Path, config: dict):
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@READY "):
            self._fail("ready", line)
        self.ready = json.loads(line[7:])
        count = 22
        expected_ids = [f"r1-rock-{index:03d}" for index in range(count)]
        if (self.ready.get("rock_count") != count
                or self.ready.get("maximum_external_wrench_count") != count + 1
                or self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", *expected_ids]):
            self.close(kill=True)
            raise RuntimeError("layered_smoke_bridge_ready_contract")
        self._write("init_default\n")
        self.initial = self._read_state("init")

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("layered_smoke_bridge_stdin_closed")
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def _fail(self, stage: str, stdout: str = "") -> None:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise RuntimeError(f"layered_smoke_bridge_{stage}_failed:{stdout}:{stderr}")

    def _read_state(self, stage: str) -> dict:
        if self.process.stdout is None:
            self._fail(stage)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            self._fail(stage, line)
        state = json.loads(line[7:])
        if len(state.get("rocks", [])) != 22:
            self._fail(stage, line)
        return state

    def step(self, dt: float, substeps: int, command: list[float],
             wrenches: dict[str, dict]) -> dict:
        previous = self.last_state if hasattr(self, "last_state") else self.initial
        fields = ["step", f"{float(previous['time_s']):.17g}", f"{dt:.17g}",
                  str(substeps), *[f"{float(value):.17g}" for value in command]]
        ordered = ["bucket", *[f"r1-rock-{index:03d}" for index in range(22)]]
        fields.append(str(len(ordered)))
        for body_name in ordered:
            wrench = wrenches[body_name]
            fields.append(body_name)
            for key in ("force", "torque", "point"):
                fields.extend(f"{float(value):.17g}" for value in wrench[key])
        self._write(" ".join(fields) + "\n")
        self.last_state = self._read_state("step")
        return self.last_state

    def close(self, kill: bool = False) -> None:
        if self.process.poll() is None:
            if kill:
                self.process.kill()
            else:
                self._write("close\n")
        stdout, stderr = self.process.communicate(timeout=20)
        if not kill and (self.process.returncode or stdout.strip() or stderr.strip()):
            raise RuntimeError(
                f"layered_smoke_bridge_close_failed:{self.process.returncode}:"
                f"{stdout}:{stderr}")


class Fixture:
    def __init__(self, config: dict, repository: Path, manifest: dict,
                 bank_path: Path, initial: dict, device: str):
        material = config["mpm"]["material"]
        state_rocks = {rock["rock_id"]: rock for rock in initial["rocks"]}
        builder = newton.ModelBuilder()
        self.body_indices = {}
        mesh_cache = {}
        for index, record in enumerate(manifest["rocks"]):
            rock_id = f"r1-rock-{index:03d}"
            state = state_rocks[rock_id]
            inertia = np.asarray(record["inertia_about_com_body_kg_m2"], dtype=np.float32)
            body = builder.add_body(
                xform=_make_transform(state["qpos"][:3], state["qpos"][3:7]),
                com=wp.vec3(0.0), inertia=wp.mat33(inertia),
                mass=float(record["mass_kg"]), label=rock_id, lock_inertia=True)
            self.body_indices[rock_id] = int(body)
            prototype = record["prototype_id"]
            if prototype not in mesh_cache:
                vertices, indices = _read_obj(
                    repository / "assets/v4/mixed-material-r0" / f"{prototype}.obj")
                mesh_cache[prototype] = newton.Mesh(
                    vertices, indices, compute_inertia=False, is_solid=True)
            builder.add_shape_mesh(
                body, mesh=mesh_cache[prototype],
                scale=wp.vec3(float(record["equivalent_diameter_m"])),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(material["friction"]),
                    has_particle_collision=True), label=rock_id + "-mpm-collider")
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(material["friction"])))
        self.collider_model = builder.finalize(device=device)

        with np.load(bank_path, allow_pickle=False) as archive:
            q = archive["q"].astype(np.float32)
        spacing = float(config["mpm"]["particle_spacing_m"])
        particle_mass = float(config["mpm"]["particle_density_kg_m3"]) * spacing ** 3
        count = len(q)
        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        sand.add_particles(
            pos=q.tolist(), vel=np.zeros_like(q).tolist(), mass=[particle_mass] * count,
            radius=[0.45 * spacing] * count,
            custom_attributes={"mpm:friction": [float(material["friction"])] * count})
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
        options.tolerance = 1.0e-4
        options.transfer_scheme = "pic"
        options.integration_scheme = "pic"
        options.strain_basis = "P0"
        options.critical_fraction = 0.0
        options.collider_basis = "S2"
        options.velocity_basis = "Q1"
        options.collider_velocity_mode = "backward"
        self.solver = SolverImplicitMPM(self.model, config=options)
        self.solver.setup_collider(model=self.collider_model)
        self.state = self.model.state()
        collider_state = self.collider_model.state()
        self.state.body_q = wp.clone(collider_state.body_q)
        self.state.body_qd = wp.clone(collider_state.body_qd)
        self.state.body_f = wp.zeros_like(collider_state.body_f)
        self.device = device
        self.assign(initial)
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy().copy()
        self.collider_ids = {
            rock_id: np.flatnonzero(body_map == body)
            for rock_id, body in self.body_indices.items()
        }
        if any(len(ids) != 1 for ids in self.collider_ids.values()):
            raise RuntimeError("layered_smoke_collider_mapping")
        self.particle_count = count
        self.particle_mass = particle_mass

    def assign(self, bridge_state: dict) -> None:
        rocks = bridge_state["rocks"]
        self.state.body_q.assign(wp.array([
            _make_transform(rock["qpos"][:3], rock["qpos"][3:7]) for rock in rocks
        ], dtype=wp.transform, device=self.device))
        self.state.body_qd.assign(wp.array([
            wp.spatial_vector(
                *[float(value) for value in rock["linear_velocity_world_m_s"]],
                *[float(value) for value in rock["angular_velocity_world_rad_s"]])
            for rock in rocks
        ], dtype=wp.spatial_vector, device=self.device))

    def step(self, dt: float, origins: dict[str, np.ndarray]) -> tuple[dict, int]:
        self.solver.step(self.state, self.state, None, None, dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        result = {}
        total_samples = 0
        for rock_id, selected_ids in self.collider_ids.items():
            mask = np.isin(ids, selected_ids)
            mask &= np.isfinite(impulse).all(axis=1)
            mask &= np.linalg.norm(impulse, axis=1) > 1e-12
            selected = impulse[mask]
            selected_pos = position[mask]
            total_samples += int(np.count_nonzero(mask))
            if selected.size:
                force = np.sum(selected, axis=0, dtype=np.float64) / dt
                torque = np.sum(np.cross(
                    selected_pos - origins[rock_id], selected), axis=0,
                    dtype=np.float64) / dt
            else:
                force = np.zeros(3)
                torque = np.zeros(3)
            result[rock_id] = {"force": force, "torque": torque,
                               "point": origins[rock_id]}
        return result, total_samples

    def metrics(self) -> dict:
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        return {
            "particle_count": len(q),
            "total_mass_kg": self.particle_count * self.particle_mass,
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "maximum_particle_speed_m_s": float(np.linalg.norm(qd, axis=1).max()),
            "minimum_particle_z_m": float(q[:, 2].min()),
            "particle_q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            "particle_qd_sha256": hashlib.sha256(qd.tobytes()).hexdigest(),
        }


def _zero_wrench(point) -> dict:
    return {"force": np.zeros(3), "torque": np.zeros(3),
            "point": np.asarray(point, dtype=np.float64)}


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path, repository)
    manifest_path = repository / config["geometry"]["manifest_path"]
    bank_path = repository / config["geometry"]["bank_path"]
    capacity_path = repository / config["capacity_bridge"]["config_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = config["probe"]
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="layered-local-smoke-") as temporary:
        work = Path(temporary)
        model = work / "model.xml"
        model_manifest = _build_provisional_model(
            capacity_path, repository, manifest, model)
        bridge_source = work / "bridge.cc"
        source, bridge_manifest = render_capacity_bridge(capacity_path, repository)
        bridge_source.write_text(source, encoding="utf-8")
        bridge_binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, bridge_source, bridge_binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        bridge = Bridge(bridge_binary, model, config)
        fixture = None
        initial_positions = {
            rock["rock_id"]: np.asarray(rock["position"], dtype=np.float64)
            for rock in bridge.initial["rocks"]}
        wrenches = {"bucket": _zero_wrench(bridge.initial["bucket_origin"])}
        for rock in bridge.initial["rocks"]:
            wrenches[rock["rock_id"]] = _zero_wrench(rock["position"])
        maximum_linear = 0.0
        maximum_angular = 0.0
        maximum_displacement = 0.0
        maximum_closure = 0.0
        maximum_forbidden = 0
        total_contact_samples = 0
        trace = []
        try:
            fixture = Fixture(config, repository, manifest, bank_path,
                              bridge.initial, args.device)
            for frame in range(int(probe["frames"])):
                state = bridge.step(
                    float(probe["output_dt_s"]),
                    int(probe["mujoco_internal_substeps"]),
                    probe["zero_command"], wrenches)
                fixture.assign(state)
                origins = {rock["rock_id"]: np.asarray(
                    rock["position"], dtype=np.float64) for rock in state["rocks"]}
                measured, samples = fixture.step(
                    float(probe["output_dt_s"]), origins)
                wrenches = {"bucket": _zero_wrench(state["bucket_origin"]), **measured}
                linear = max(float(np.linalg.norm(
                    rock["linear_velocity_world_m_s"])) for rock in state["rocks"])
                angular = max(float(np.linalg.norm(
                    rock["angular_velocity_world_rad_s"])) for rock in state["rocks"])
                displacement = max(float(np.linalg.norm(
                    np.asarray(rock["position"]) - initial_positions[rock["rock_id"]]))
                    for rock in state["rocks"])
                maximum_linear = max(maximum_linear, linear)
                maximum_angular = max(maximum_angular, angular)
                maximum_displacement = max(maximum_displacement, displacement)
                maximum_closure = max(maximum_closure, float(state["closure_m"]))
                maximum_forbidden = max(
                    maximum_forbidden, int(state["forbidden_contact_count"]))
                total_contact_samples += samples
                trace.append({"frame": frame + 1, "time_s": state["time_s"],
                              "maximum_rock_linear_speed_m_s": linear,
                              "maximum_rock_angular_speed_rad_s": angular,
                              "maximum_rock_displacement_m": displacement,
                              "mpm_rock_contact_samples": samples})
        finally:
            bridge.close()
        if fixture is None:
            raise RuntimeError("layered_smoke_fixture_not_created")
        matrix = fixture.metrics()
    checks = {
        "finite_mpm_state": matrix["finite"],
        "particle_count_preserved": matrix["particle_count"] == 72415,
        "nonzero_mpm_rock_contact": total_contact_samples > 0,
        "mechanism_closure_within_limit": maximum_closure <=
            float(probe["maximum_mechanism_closure_m"]),
        "forbidden_contacts_within_limit": maximum_forbidden <=
            int(probe["maximum_forbidden_contact_count"]),
        "particle_speed_bounded": matrix["maximum_particle_speed_m_s"] <=
            float(probe["maximum_particle_speed_m_s"]),
        "rock_linear_speed_bounded": maximum_linear <=
            float(probe["maximum_rock_linear_speed_m_s"]),
        "rock_angular_speed_bounded": maximum_angular <=
            float(probe["maximum_rock_angular_speed_rad_s"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "final_assembled_short_mixed_dynamics_smoke_passed"
        if all(checks.values()) else "final_assembled_short_mixed_dynamics_smoke_failed",
        "short_smoke_qualified": bool(all(checks.values())),
        "settling_qualified": False,
        "wall_free_hold_qualified": False,
        "formal_scene_qualified": False,
        "checks": checks,
        "metrics": {
            "frames": int(probe["frames"]),
            "simulated_time_s": int(probe["frames"]) * float(probe["output_dt_s"]),
            "maximum_rock_linear_speed_m_s": maximum_linear,
            "maximum_rock_angular_speed_rad_s": maximum_angular,
            "maximum_rock_displacement_m": maximum_displacement,
            "maximum_mechanism_closure_m": maximum_closure,
            "maximum_forbidden_contact_count": maximum_forbidden,
            "mpm_rock_contact_samples": total_contact_samples,
            "matrix": matrix,
        },
        "trace": trace,
        "inputs": {
            "config_sha256": sha256(config_path),
            "geometry_manifest_sha256": sha256(manifest_path),
            "geometry_bank_sha256": sha256(bank_path),
            "model_manifest": model_manifest,
            "bridge_manifest": bridge_manifest,
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "wall_seconds": time.perf_counter() - started,
        },
        "scope_limit": (
            "all final particles and all 22 rocks active at frame zero for 0.1 s; "
            "no staged activation, gravity ramp, construction walls, settling "
            "qualification, one-second nominal hold, excavation, or training"),
    }
    _write_json(Path(args.output), report)
    return report


def main() -> None:
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
    print(json.dumps({"status": report["status"],
                      "short_smoke_qualified": report["short_smoke_qualified"],
                      "metrics": report["metrics"]}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
