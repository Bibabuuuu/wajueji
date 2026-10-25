#!/usr/bin/env python3
"""Qualify stable per-rock Newton collider routing for the R3-C two-rock fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from build_mixed_coupled_bridge_r3_c import write_bridge
from build_mixed_multirock_candidate_r3_c import (
    build_two_rock_candidate,
    sha256,
)
from mixed_material_r3_c_routing_contract import validate_config
from run_mixed_material_r3_a_probe import (
    _compile_bridge,
    _make_transform,
    _read_obj,
    _write_json,
)


REPORT_SCHEMA = "excavation_mixed_material_r3_c_routing_report_v1"


class R3CRoutingBridge:
    def __init__(self, binary: Path, model: Path, ready_schema: str,
                 state_schema: str):
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        ready_line = self.process.stdout.readline().rstrip("\n")
        if not ready_line.startswith("@READY "):
            self._fail("ready", ready_line)
        self.ready = json.loads(ready_line[7:])
        if (self.ready.get("schema") != ready_schema or
                self.ready.get("rock_count") != 2 or
                self.ready.get("maximum_external_wrench_count") != 3 or
                self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", "r1-rock-000", "r1-rock-001"]):
            self.close(kill=True)
            raise RuntimeError("r3_c_routing_bridge_ready_contract")
        self.state_schema = state_schema
        self._write("init_default\n")
        self.initial = self._read_state("init")

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("r3_c_routing_bridge_stdin_closed")
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def _fail(self, stage: str, stdout: str = "") -> None:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise RuntimeError(
            f"r3_c_routing_bridge_{stage}_failed:{stdout}:{stderr}")

    def _read_state(self, stage: str) -> dict:
        if self.process.stdout is None:
            self._fail(stage)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            self._fail(stage, line)
        state = json.loads(line[7:])
        if (state.get("schema") != self.state_schema or
                [item.get("rock_id") for item in state.get("rocks", [])] !=
                ["r1-rock-000", "r1-rock-001"]):
            self.close(kill=True)
            raise RuntimeError("r3_c_routing_bridge_state_contract")
        return state

    def step(self, dt: float, substeps: int, command, rock_id: str,
             force, torque, point) -> dict:
        previous = self.last_state if hasattr(self, "last_state") else self.initial
        fields = [
            "step", f"{float(previous['time_s']):.17g}", f"{dt:.17g}",
            str(substeps),
        ]
        fields += [f"{float(value):.17g}" for value in command]
        fields += ["1", rock_id]
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
                "r3_c_routing_bridge_close_failed:"
                f"{self.process.returncode}:{stdout}:{stderr}")


def _rock_map(state: dict) -> dict[str, dict]:
    rocks = state.get("rocks", [])
    result = {item["rock_id"]: item for item in rocks}
    if list(result) != ["r1-rock-000", "r1-rock-001"]:
        raise ValueError("r3_c_routing_state_rock_order")
    return result


class TwoRockRoutingFixture:
    def __init__(self, routing_config: dict, r3_c_config: dict,
                 r3_b_config: dict, repository: Path, device: str,
                 initial_state: dict, target_rock_id: str):
        probe = routing_config["routing_probe"]
        rock_ids = probe["rock_ids"]
        if target_rock_id not in rock_ids:
            raise ValueError("r3_c_routing_unknown_target")
        self.target_rock_id = target_rock_id
        self.target_index = rock_ids.index(target_rock_id)
        r3_c_rocks = r3_c_config["two_rock_fixture"]["rocks"]
        r3_b_frozen = r3_b_config["frozen_inputs"]
        patch = r3_b_config["matrix_patch"]
        vertices, indices = _read_obj(
            repository / r3_c_config["frozen_inputs"]["rock_asset_path"])
        inertia = np.asarray(
            r3_b_frozen["rock_inertia_about_com_kg_m2"], dtype=np.float64)
        bridge_rocks = _rock_map(initial_state)

        builder = newton.ModelBuilder()
        self.rock_body_indices: dict[str, int] = {}
        for index, rock in enumerate(r3_c_rocks):
            rock_id = rock["rock_id"]
            state = bridge_rocks[rock_id]
            body = builder.add_body(
                xform=_make_transform(state["qpos"][:3], state["qpos"][3:7]),
                com=wp.vec3(0.0), inertia=wp.mat33(inertia.astype(np.float32)),
                mass=float(r3_b_frozen["rock_mass_kg"]), label=rock_id,
                lock_inertia=True)
            self.rock_body_indices[rock_id] = int(body)
            builder.add_shape_mesh(
                body,
                mesh=newton.Mesh(
                    vertices, indices, compute_inertia=False, is_solid=True),
                scale=wp.vec3(float(rock["equivalent_diameter_m"])),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(patch["material"]["friction"]),
                    has_particle_collision=(index == self.target_index)),
                label=rock_id + "-mpm-collider")
        support = patch["support_fixture"]
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(*[float(value) for value in support["position_world_m"]]),
                wp.quat_identity()),
            hx=float(support["half_size_m"][0]),
            hy=float(support["half_size_m"][1]),
            hz=float(support["half_size_m"][2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(support["friction"])),
            label="r1_ground_push_shelf")
        self.collider_model = builder.finalize(device=device)

        target_position = np.asarray(
            bridge_rocks[target_rock_id]["position"], dtype=np.float64)
        lower = target_position + np.asarray(
            probe["matrix_patch_lower_offset_from_target_rock_m"],
            dtype=np.float64)
        size = np.asarray(probe["matrix_patch_size_xyz_m"], dtype=np.float64)
        spacing = float(probe["particle_spacing_m"])
        dimensions = np.rint(size / spacing).astype(int) + 1
        if not np.allclose((dimensions - 1) * spacing, size, atol=1.0e-12):
            raise ValueError("r3_c_routing_patch_spacing_contract")
        particle_count = int(np.prod(dimensions, dtype=np.int64))
        self.target_mass_kg = float(
            np.prod(size) * float(patch["density_kg_m3"]))
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        sand_builder.add_particle_grid(
            pos=wp.vec3(*[float(value) for value in lower]),
            rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=int(dimensions[0]), dim_y=int(dimensions[1]),
            dim_z=int(dimensions[2]), cell_x=spacing, cell_y=spacing,
            cell_z=spacing, mass=self.target_mass_kg / particle_count,
            jitter=0.0, radius_mean=0.45 * spacing,
            custom_attributes={
                "mpm:friction": float(patch["material"]["friction"]),
            })
        self.model = sand_builder.finalize(device=device)
        self.model.set_gravity(
            tuple(float(value) for value in patch["gravity_m_s2"]))
        for name, value in patch["material"].items():
            if hasattr(self.model.mpm, name):
                getattr(self.model.mpm, name).fill_(float(value))
        options = SolverImplicitMPM.Config()
        options.voxel_size = float(patch["voxel_size_m"])
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
        options.collider_velocity_mode = "backward"
        self.solver = SolverImplicitMPM(self.model, config=options)
        self.solver.setup_collider(model=self.collider_model)
        self.state = self.model.state()
        collider_state = self.collider_model.state()
        self.state.body_q = wp.clone(collider_state.body_q)
        self.state.body_qd = wp.clone(collider_state.body_qd)
        self.state.body_f = wp.zeros_like(collider_state.body_f)
        self.device = device
        self.assign(initial_state)
        self.solver.reset(self.state)
        self.initial_q = self.state.particle_q.numpy().copy()
        self.initial_qd = self.state.particle_qd.numpy().copy()

        body_map = self.solver.collider_body_index.numpy().copy()
        self.collider_body_map = body_map
        self.static_collider_ids = np.flatnonzero(body_map == -1)
        self.rock_id_to_collider_ids = {
            rock_id: np.flatnonzero(body_map == body_index)
            for rock_id, body_index in self.rock_body_indices.items()
        }
        if (self.rock_body_indices != {"r1-rock-000": 0, "r1-rock-001": 1} or
                self.rock_id_to_collider_ids[target_rock_id].size != 1 or
                any(self.rock_id_to_collider_ids[rock_id].size
                    for rock_id in rock_ids if rock_id != target_rock_id)):
            raise RuntimeError(
                "r3_c_routing_collider_map:" + json.dumps(body_map.tolist()))

    def assign(self, bridge_state: dict) -> None:
        rocks = _rock_map(bridge_state)
        transforms = []
        velocities = []
        for rock_id in ("r1-rock-000", "r1-rock-001"):
            rock = rocks[rock_id]
            transforms.append(_make_transform(rock["qpos"][:3], rock["qpos"][3:7]))
            linear = rock["linear_velocity_world_m_s"]
            angular = rock["angular_velocity_world_rad_s"]
            velocities.append(wp.spatial_vector(
                *[float(value) for value in linear],
                *[float(value) for value in angular]))
        self.state.body_q.assign(
            wp.array(transforms, dtype=wp.transform, device=self.device))
        self.state.body_qd.assign(
            wp.array(velocities, dtype=wp.spatial_vector, device=self.device))

    @staticmethod
    def _collect(impulses, positions, ids, selected_ids, origin, dt):
        mask = np.isin(ids, selected_ids)
        mask &= np.isfinite(impulses).all(axis=1)
        mask &= np.linalg.norm(impulses, axis=1) > 1.0e-12
        selected = impulses[mask]
        selected_positions = positions[mask]
        if not selected.size:
            return np.zeros(3), np.zeros(3), 0
        force = np.sum(selected, axis=0, dtype=np.float64) / dt
        torque = np.sum(
            np.cross(selected_positions - origin, selected),
            axis=0, dtype=np.float64) / dt
        return force, torque, int(np.count_nonzero(mask))

    def step_and_collect(self, dt: float, origins: dict[str, np.ndarray]):
        self.solver.step(self.state, self.state, None, None, dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(
            self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        finite = np.isfinite(impulse).all(axis=1) & np.isfinite(position).all(axis=1)
        active = finite & (np.linalg.norm(impulse, axis=1) > 1.0e-12)
        all_owned_ids = self.static_collider_ids.copy()
        for selected in self.rock_id_to_collider_ids.values():
            all_owned_ids = np.concatenate((all_owned_ids, selected))
        results = {
            rock_id: self._collect(
                impulse, position, ids, selected_ids, origins[rock_id], dt)
            for rock_id, selected_ids in self.rock_id_to_collider_ids.items()
        }
        ownership = {
            "sample_count": int(ids.size),
            "nonfinite_sample_count": int(np.count_nonzero(~finite)),
            "active_sample_count": int(np.count_nonzero(active)),
            "unmapped_active_sample_count": int(np.count_nonzero(
                active & ~np.isin(ids, all_owned_ids))),
        }
        return results, ownership

    def metrics(self) -> dict:
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        total_mass = float(np.sum(
            self.model.particle_mass.numpy(), dtype=np.float64))
        return {
            "particle_count": int(q.shape[0]),
            "target_mass_kg": self.target_mass_kg,
            "total_mass_kg": total_mass,
            "target_mass_error_kg": total_mass - self.target_mass_kg,
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "particle_q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            "particle_qd_sha256": hashlib.sha256(qd.tobytes()).hexdigest(),
        }

    def reset_audit(self) -> dict:
        self.state.particle_q.assign(
            wp.array(self.initial_q, dtype=wp.vec3, device=self.device))
        self.state.particle_qd.assign(
            wp.array(self.initial_qd, dtype=wp.vec3, device=self.device))
        self.solver.reset(self.state)
        wp.synchronize()
        mpm = self.state.mpm
        identity = np.eye(3, dtype=np.float32)
        checks = {
            "particle_q_exact": np.array_equal(
                self.state.particle_q.numpy(), self.initial_q),
            "particle_qd_exact": np.array_equal(
                self.state.particle_qd.numpy(), self.initial_qd),
            "particle_elastic_strain_identity": np.array_equal(
                mpm.particle_elastic_strain.numpy(),
                np.broadcast_to(
                    identity, mpm.particle_elastic_strain.numpy().shape)),
            "particle_transform_identity": np.array_equal(
                mpm.particle_transform.numpy(),
                np.broadcast_to(identity, mpm.particle_transform.numpy().shape)),
            "particle_qd_grad_zero": not np.any(mpm.particle_qd_grad.numpy()),
            "particle_stress_zero": not np.any(mpm.particle_stress.numpy()),
            "particle_Jp_one": np.all(mpm.particle_Jp.numpy() == 1.0),
        }
        return {
            "checks": {key: bool(value) for key, value in checks.items()},
            "passed": bool(all(checks.values())),
        }


def _episode(routing_config: dict, r3_c_config: dict, r3_b_config: dict,
             repository: Path, binary: Path, model: Path, device: str,
             target_rock_id: str) -> dict:
    probe = routing_config["routing_probe"]
    bridge = R3CRoutingBridge(
        binary, model, r3_c_config["bridge"]["ready_schema"],
        r3_c_config["bridge"]["state_schema"])
    fixture = None
    rock_ids = probe["rock_ids"]
    force = {rock_id: np.zeros(3) for rock_id in rock_ids}
    torque = {rock_id: np.zeros(3) for rock_id in rock_ids}
    accumulated_impulse = {rock_id: 0.0 for rock_id in rock_ids}
    contact_frames = {rock_id: 0 for rock_id in rock_ids}
    contact_samples = {rock_id: 0 for rock_id in rock_ids}
    maximum_echo_residual = 0.0
    maximum_reference_residual = 0.0
    maximum_forbidden = 0
    maximum_closure = 0.0
    ownership_totals = {
        "sample_count": 0,
        "nonfinite_sample_count": 0,
        "active_sample_count": 0,
        "unmapped_active_sample_count": 0,
    }
    trace = []
    dt = float(probe["output_dt_s"])
    try:
        fixture = TwoRockRoutingFixture(
            routing_config, r3_c_config, r3_b_config, repository, device,
            bridge.initial, target_rock_id)
        for frame in range(int(probe["frames"])):
            prior = bridge.last_state if hasattr(bridge, "last_state") else bridge.initial
            prior_rocks = _rock_map(prior)
            reference = np.asarray(
                prior_rocks[target_rock_id]["position"], dtype=np.float64)
            sent_force = force[target_rock_id].copy()
            sent_torque = torque[target_rock_id].copy()
            state = bridge.step(
                dt, int(probe["mujoco_internal_substeps"]), probe["command"],
                target_rock_id, sent_force, sent_torque, reference)
            applied = {
                item["rock_id"]: item for item in state["applied_rocks_mpm"]
            }
            for rock_id in rock_ids:
                expected_force = sent_force if rock_id == target_rock_id else np.zeros(3)
                expected_torque = sent_torque if rock_id == target_rock_id else np.zeros(3)
                maximum_echo_residual = max(
                    maximum_echo_residual,
                    float(np.linalg.norm(
                        np.asarray(applied[rock_id]["force_world_n"]) -
                        expected_force)),
                    float(np.linalg.norm(
                        np.asarray(applied[rock_id]["torque_world_nm"]) -
                        expected_torque)))
            maximum_reference_residual = max(
                maximum_reference_residual,
                float(np.linalg.norm(
                    reference - np.asarray(
                        prior_rocks[target_rock_id]["position"],
                        dtype=np.float64))))
            maximum_forbidden = max(
                maximum_forbidden, int(state["forbidden_contact_count"]))
            maximum_closure = max(
                maximum_closure, float(state["closure_m"]))
            fixture.assign(state)
            current_rocks = _rock_map(state)
            origins = {
                rock_id: np.asarray(
                    current_rocks[rock_id]["position"], dtype=np.float64)
                for rock_id in rock_ids
            }
            measured, ownership = fixture.step_and_collect(dt, origins)
            for rock_id in rock_ids:
                force[rock_id], torque[rock_id], samples = measured[rock_id]
                accumulated_impulse[rock_id] += (
                    float(np.linalg.norm(force[rock_id])) * dt)
                contact_frames[rock_id] += int(samples > 0)
                contact_samples[rock_id] += samples
            for key in ownership_totals:
                ownership_totals[key] += ownership[key]
            trace.append({
                "frame": frame + 1,
                "time_s": float(state["time_s"]),
                "target_rock_id": target_rock_id,
                "reference_point_world_m": reference.tolist(),
                "sent_force_world_n": sent_force.tolist(),
                "sent_torque_about_reference_world_nm": sent_torque.tolist(),
                "measured": {
                    rock_id: {
                        "force_world_n": force[rock_id].tolist(),
                        "torque_about_current_origin_world_nm":
                            torque[rock_id].tolist(),
                    } for rock_id in rock_ids
                },
                "ownership": ownership,
            })
    finally:
        bridge.close()
    if fixture is None:
        raise RuntimeError("r3_c_routing_fixture_not_created")
    trace_bytes = json.dumps(
        trace, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return {
        "round": "rock_000_mpm_only" if target_rock_id.endswith("000")
        else "rock_001_mpm_only",
        "target_rock_id": target_rock_id,
        "frames": int(probe["frames"]),
        "simulated_time_s": int(probe["frames"]) * dt,
        "rock_id_to_newton_body_index": fixture.rock_body_indices,
        "rock_id_to_collider_ids": {
            rock_id: selected.tolist()
            for rock_id, selected in fixture.rock_id_to_collider_ids.items()
        },
        "static_collider_ids": fixture.static_collider_ids.tolist(),
        "collider_body_map": fixture.collider_body_map.tolist(),
        "accumulated_direct_mpm_impulse_ns": accumulated_impulse,
        "contact_frames": contact_frames,
        "contact_samples": contact_samples,
        "maximum_wrench_echo_residual": maximum_echo_residual,
        "maximum_reference_point_residual_m": maximum_reference_residual,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "maximum_mechanism_closure_m": maximum_closure,
        "collider_ownership": ownership_totals,
        "matrix": fixture.metrics(),
        "mpm_history_reset": fixture.reset_audit(),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "trace": trace,
    }


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    inventory = Path(args.inventory).resolve()
    output = Path(args.output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    r3_c_config_path = repository / frozen["r3_c_config_path"]
    r3_c_config = json.loads(r3_c_config_path.read_text(encoding="utf-8"))
    r3_b_config = json.loads(
        (repository / frozen["r3_b_config_path"]).read_text(encoding="utf-8"))
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
            prefix="mixed-material-r3-c-routing-") as temporary:
        work = Path(temporary)
        model = work / "model.xml"
        model_manifest = build_two_rock_candidate(
            r3_c_config_path, repository, inventory, model)
        source = work / "bridge.cc"
        source_manifest = write_bridge(
            r3_c_config_path, repository, source)
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        episodes = {
            rock_id: _episode(
                config, r3_c_config, r3_b_config, repository, binary, model,
                args.device, rock_id)
            for rock_id in config["routing_probe"]["rock_ids"]
        }
        model_sha256 = sha256(model)
        source_sha256 = sha256(source)
        if args.artifact_dir:
            artifact_dir = Path(args.artifact_dir).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "generated-model.xml").write_bytes(model.read_bytes())
            (artifact_dir / "generated-bridge.cc").write_bytes(source.read_bytes())
            _write_json(
                artifact_dir / "generated-model-manifest.json", model_manifest)
            _write_json(
                artifact_dir / "generated-bridge-manifest.json", source_manifest)
    probe = config["routing_probe"]
    active_min = float(probe["minimum_target_direct_impulse_ns"])
    inactive_max = float(probe["maximum_inactive_direct_impulse_ns"])
    rock_ids = probe["rock_ids"]
    checks = {
        "stable_body_index_mapping": all(
            episode["rock_id_to_newton_body_index"] ==
            {"r1-rock-000": 0, "r1-rock-001": 1}
            for episode in episodes.values()),
        "target_collider_only_per_round": all(
            len(episode["rock_id_to_collider_ids"][target]) == 1 and
            all(not episode["rock_id_to_collider_ids"][rock_id]
                for rock_id in rock_ids if rock_id != target)
            for target, episode in episodes.items()),
        "rock_000_route_isolated": (
            episodes["r1-rock-000"]["accumulated_direct_mpm_impulse_ns"]
            ["r1-rock-000"] >= active_min and
            episodes["r1-rock-000"]["accumulated_direct_mpm_impulse_ns"]
            ["r1-rock-001"] <= inactive_max),
        "rock_001_route_isolated": (
            episodes["r1-rock-001"]["accumulated_direct_mpm_impulse_ns"]
            ["r1-rock-001"] >= active_min and
            episodes["r1-rock-001"]["accumulated_direct_mpm_impulse_ns"]
            ["r1-rock-000"] <= inactive_max),
        "per_id_wrench_echo_exact": all(
            episode["maximum_wrench_echo_residual"] <=
            float(probe["maximum_wrench_echo_residual"])
            for episode in episodes.values()),
        "per_id_reference_point_exact": all(
            episode["maximum_reference_point_residual_m"] <=
            float(probe["maximum_reference_point_residual_m"])
            for episode in episodes.values()),
        "all_mpm_states_finite": all(
            episode["matrix"]["finite"] for episode in episodes.values()),
        "matrix_mass_matched": all(
            abs(episode["matrix"]["target_mass_error_kg"]) <=
            float(probe["maximum_mass_error_kg"])
            for episode in episodes.values()),
        "no_unmapped_or_nonfinite_collider_samples": all(
            episode["collider_ownership"]["unmapped_active_sample_count"] == 0 and
            episode["collider_ownership"]["nonfinite_sample_count"] == 0
            for episode in episodes.values()),
        "forbidden_contacts_zero": all(
            episode["maximum_forbidden_contact_count"] == 0
            for episode in episodes.values()),
        "closure_within_one_mm": all(
            episode["maximum_mechanism_closure_m"] <= 0.001
            for episode in episodes.values()),
        "mpm_history_reset_complete": all(
            episode["mpm_history_reset"]["passed"]
            for episode in episodes.values()),
        "frozen_scope_preserved": (
            not config["scope"]
            ["frozen_r1_r2_r3_a_r3_b_and_r3_c_preflight_modified"] and
            not config["scope"]["r3_c_physics_qualified"] and
            not config["scope"]["official_envelope"] and
            not config["scope"]["training"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "r3_c_per_rock_routing_probe_passed" if all(checks.values())
        else "r3_c_per_rock_routing_probe_failed",
        "routing_physics_qualified": bool(all(checks.values())),
        "r3_c_physics_qualified": False,
        "training_qualified": False,
        "checks": {key: bool(value) for key, value in checks.items()},
        "episodes": episodes,
        "backend": {
            "python": platform.python_version(),
            "newton": newton.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "device_requested": args.device,
            "device_resolved": str(wp.get_device()),
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "r3_c_config_sha256": sha256(r3_c_config_path),
            "derived_model_sha256": model_sha256,
            "model_manifest": model_manifest,
            "bridge_source_sha256": source_sha256,
            "bridge_source_manifest": source_manifest,
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "contact_ownership": {
            "mujoco": ["rigid-rock-bucket", "rigid-rock-rock",
                       "rigid-rock-support"],
            "newton_mpm": ["target-rigid-rock-matrix",
                           "matrix-support"],
            "bucket_mpm_disabled_for_this_gate": True,
            "duplicate_contact_class_count": 0,
        },
        "scope_limit": (
            "two isolated single-rock MPM routing rounds in the exact-two-rock "
            "R3-C fixture; not the four-round force partition, not feedback or "
            "repeat/dt qualification, not a complete in-process mixed reset, "
            "not the official envelope, and not training-qualified"),
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    parser.add_argument("--artifact-dir")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"],
        "routing_physics_qualified": report["routing_physics_qualified"],
        "r3_c_physics_qualified": report["r3_c_physics_qualified"],
        "checks": report["checks"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
