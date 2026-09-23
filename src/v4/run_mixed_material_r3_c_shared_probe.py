#!/usr/bin/env python3
"""Run the R3-C exact-two-rock shared-matrix four-round qualification gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from build_mixed_coupled_bridge_r3_c_dynamics import (
    READY_SCHEMA,
    STATE_SCHEMA,
    write_bridge,
)
from build_mixed_multirock_candidate_r3_c import (
    build_two_rock_candidate,
    sha256,
)
from mixed_material_r1 import build_proxy
from mixed_material_r3_c_shared_contract import validate_config
from run_mixed_material_r3_a_probe import (
    _compile_bridge,
    _make_transform,
    _read_obj,
    _write_json,
)
from run_mixed_material_r3_b_probe import _bucket_transform, _quat_wxyz_to_xyzw


REPORT_SCHEMA = "excavation_mixed_material_r3_c_shared_four_round_report_v1"
ROCK_IDS = ("r1-rock-000", "r1-rock-001")


def _rock_map(state: dict) -> dict[str, dict]:
    rocks = state.get("rocks", [])
    if [item.get("rock_id") for item in rocks] != list(ROCK_IDS):
        raise ValueError("r3_c_shared_state_rock_order")
    return {item["rock_id"]: item for item in rocks}


def _disable_bucket_rigid_contact(source_model: Path, output_model: Path) -> None:
    tree = ET.parse(source_model)
    count = 0
    for geom in tree.getroot().iter("geom"):
        if geom.get("name", "").startswith("r1_bucket_"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            count += 1
    if count != 18:
        raise ValueError("r3_c_shared_bucket_proxy_count")
    ET.indent(tree, space="  ")
    tree.write(output_model, encoding="unicode", xml_declaration=False)
    with output_model.open("a", encoding="utf-8") as stream:
        stream.write("\n")


class SharedBridge:
    def __init__(self, binary: Path, model: Path):
        self.process = subprocess.Popen(
            [str(binary), str(model)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        ready_line = self.process.stdout.readline().rstrip("\n")
        if not ready_line.startswith("@READY "):
            self._fail("ready", ready_line)
        self.ready = json.loads(ready_line[7:])
        if (self.ready.get("schema") != READY_SCHEMA or
                self.ready.get("rock_count") != 2 or
                self.ready.get("maximum_external_wrench_count") != 3 or
                self.ready.get("allowed_external_wrench_bodies") !=
                ["bucket", *ROCK_IDS]):
            self.close(kill=True)
            raise RuntimeError("r3_c_shared_bridge_ready_contract")
        self._write("init_default\n")
        self.initial = self._read_state("init")

    def _write(self, value: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("r3_c_shared_bridge_stdin_closed")
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def _fail(self, stage: str, stdout: str = "") -> None:
        stderr = ""
        if self.process.poll() is not None and self.process.stderr is not None:
            stderr = self.process.stderr.read()
        raise RuntimeError(f"r3_c_shared_bridge_{stage}_failed:{stdout}:{stderr}")

    def _read_state(self, stage: str) -> dict:
        if self.process.stdout is None:
            self._fail(stage)
        line = self.process.stdout.readline().rstrip("\n")
        if not line.startswith("@STATE "):
            self._fail(stage, line)
        state = json.loads(line[7:])
        if state.get("schema") != STATE_SCHEMA:
            self.close(kill=True)
            raise RuntimeError("r3_c_shared_bridge_state_schema")
        _rock_map(state)
        return state

    def step(self, dt: float, substeps: int, command, wrenches: dict) -> dict:
        previous = self.last_state if hasattr(self, "last_state") else self.initial
        fields = [
            "step", f"{float(previous['time_s']):.17g}", f"{dt:.17g}",
            str(substeps),
        ]
        fields += [f"{float(value):.17g}" for value in command]
        ordered = ["bucket", *ROCK_IDS]
        fields.append(str(len(ordered)))
        for body_name in ordered:
            wrench = wrenches[body_name]
            fields.append(body_name)
            for key in ("force", "torque", "point"):
                fields += [f"{float(value):.17g}" for value in wrench[key]]
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
                "r3_c_shared_bridge_close_failed:"
                f"{self.process.returncode}:{stdout}:{stderr}")


class SharedMatrixFixture:
    def __init__(self, config: dict, r3_c_config: dict, r3_b_config: dict,
                 repository: Path, inventory: Path, device: str,
                 initial_state: dict, mode: str):
        if mode not in {"clear", "mpm_only", "rigid_only", "combined"}:
            raise ValueError("r3_c_shared_unknown_mode:" + mode)
        particle_collision = mode in {"mpm_only", "combined"}
        probe = config["probe"]
        patch = r3_b_config["matrix_patch"]
        r3_b_frozen = r3_b_config["frozen_inputs"]
        r1_config = json.loads(
            (repository / r3_c_config["frozen_inputs"]["r1_config_path"])
            .read_text(encoding="utf-8"))
        if sha256(inventory) != r1_config["bucket"]["inventory_contract_sha256"]:
            raise ValueError("r3_c_shared_inventory_sha256_mismatch")
        proxy = build_proxy(
            json.loads(inventory.read_text(encoding="utf-8")), r1_config)
        vertices, indices = _read_obj(
            repository / r3_c_config["frozen_inputs"]["rock_asset_path"])
        inertia = np.asarray(
            r3_b_frozen["rock_inertia_about_com_kg_m2"], dtype=np.float64)
        rocks = _rock_map(initial_state)

        builder = newton.ModelBuilder()
        self.rock_body_indices: dict[str, int] = {}
        for rock_id in ROCK_IDS:
            rock = rocks[rock_id]
            body = builder.add_body(
                xform=_make_transform(rock["qpos"][:3], rock["qpos"][3:7]),
                com=wp.vec3(0.0), inertia=wp.mat33(inertia.astype(np.float32)),
                mass=float(r3_b_frozen["rock_mass_kg"]), label=rock_id,
                lock_inertia=True)
            self.rock_body_indices[rock_id] = int(body)
            builder.add_shape_mesh(
                body,
                mesh=newton.Mesh(
                    vertices, indices, compute_inertia=False, is_solid=True),
                scale=wp.vec3(float(r3_b_frozen["rock_equivalent_diameter_m"])),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(patch["material"]["friction"]),
                    has_particle_collision=particle_collision),
                label=rock_id + "-mpm-collider")
        self.bucket_body_index = int(builder.add_body(
            xform=_bucket_transform(initial_state), mass=100.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)), label="bucket",
            lock_inertia=True, is_kinematic=True))
        bucket_shape_config = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(patch["material"]["friction"]),
            has_particle_collision=particle_collision)
        for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
            builder.add_shape_box(
                self.bucket_body_index,
                xform=wp.transform(
                    wp.vec3(*[float(value) for value in box["position_bucket_m"]]),
                    _quat_wxyz_to_xyzw(box["quaternion_wxyz"])),
                hx=float(box["half_size_m"][0]),
                hy=float(box["half_size_m"][1]),
                hz=float(box["half_size_m"][2]),
                cfg=bucket_shape_config, label="r3c_" + box["name"])
        for side in proxy["side_plate_meshes"]:
            builder.add_shape_mesh(
                self.bucket_body_index,
                mesh=newton.Mesh(
                    np.asarray(side["vertices_bucket_m"], dtype=np.float32),
                    np.asarray(side["faces"], dtype=np.int32).reshape(-1),
                    compute_inertia=False, is_solid=True),
                scale=wp.vec3(1.0), cfg=bucket_shape_config,
                label="r3c_" + side["name"])
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

        lower = np.asarray(probe["matrix_patch_lower_world_m"], dtype=np.float64)
        size = np.asarray(probe["matrix_patch_size_xyz_m"], dtype=np.float64)
        spacing = float(probe["particle_spacing_m"])
        dimensions = np.rint(size / spacing).astype(int) + 1
        if not np.allclose((dimensions - 1) * spacing, size, atol=1.0e-12):
            raise ValueError("r3_c_shared_patch_spacing_contract")
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

        self.collider_body_map = self.solver.collider_body_index.numpy().copy()
        self.static_collider_ids = np.flatnonzero(self.collider_body_map == -1)
        self.rock_id_to_collider_ids = {
            rock_id: np.flatnonzero(self.collider_body_map == body_index)
            for rock_id, body_index in self.rock_body_indices.items()
        }
        self.bucket_collider_ids = np.flatnonzero(
            self.collider_body_map == self.bucket_body_index)
        expected = int(particle_collision)
        if (self.rock_body_indices != {"r1-rock-000": 0, "r1-rock-001": 1} or
                self.bucket_body_index != 2 or
                any(selected.size != expected
                    for selected in self.rock_id_to_collider_ids.values()) or
                self.bucket_collider_ids.size != expected):
            raise RuntimeError(
                "r3_c_shared_collider_map:" +
                json.dumps(self.collider_body_map.tolist()))

    def assign(self, bridge_state: dict) -> None:
        rocks = _rock_map(bridge_state)
        transforms = [
            _make_transform(rocks[rock_id]["qpos"][:3],
                            rocks[rock_id]["qpos"][3:7])
            for rock_id in ROCK_IDS
        ]
        transforms.append(_bucket_transform(bridge_state))
        velocities = []
        for rock_id in ROCK_IDS:
            rock = rocks[rock_id]
            velocities.append(wp.spatial_vector(
                *[float(value) for value in rock["linear_velocity_world_m_s"]],
                *[float(value) for value in rock["angular_velocity_world_rad_s"]]))
        velocities.append(wp.spatial_vector(
            *[float(value) for value in
              bridge_state["bucket_linear_velocity_world_m_s"]],
            *[float(value) for value in
              bridge_state["bucket_angular_velocity_world_rad_s"]]))
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
        return (
            np.sum(selected, axis=0, dtype=np.float64) / dt,
            np.sum(np.cross(selected_positions - origin, selected),
                   axis=0, dtype=np.float64) / dt,
            int(np.count_nonzero(mask)),
        )

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
        selected = {
            rock_id: self.rock_id_to_collider_ids[rock_id]
            for rock_id in ROCK_IDS
        }
        selected["bucket"] = self.bucket_collider_ids
        results = {
            body_name: self._collect(
                impulse, position, ids, collider_ids_for_body,
                origins[body_name], dt)
            for body_name, collider_ids_for_body in selected.items()
        }
        owned_ids = self.static_collider_ids.copy()
        for collider_ids_for_body in selected.values():
            owned_ids = np.concatenate((owned_ids, collider_ids_for_body))
        sets = [set(values.tolist()) for values in selected.values()]
        sets.append(set(self.static_collider_ids.tolist()))
        ownership = {
            "sample_count": int(ids.size),
            "nonfinite_sample_count": int(np.count_nonzero(~finite)),
            "active_sample_count": int(np.count_nonzero(active)),
            "unmapped_active_sample_count": int(np.count_nonzero(
                active & ~np.isin(ids, owned_ids))),
            "mapping_disjoint": all(
                not sets[first].intersection(sets[second])
                for first in range(len(sets))
                for second in range(first + 1, len(sets))),
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


def _zeros_wrench(point) -> dict:
    return {"force": np.zeros(3), "torque": np.zeros(3),
            "point": np.asarray(point, dtype=np.float64)}


def _episode(config: dict, r3_c_config: dict, r3_b_config: dict,
             repository: Path, inventory: Path, binary: Path, model: Path,
             device: str, mode: str, frames: int, feedback: bool = True) -> dict:
    probe = config["probe"]
    dt = float(probe["output_dt_s"])
    bridge = SharedBridge(binary, model)
    fixture = None
    prior = bridge.initial
    prior_rocks = _rock_map(prior)
    wrenches = {
        "bucket": _zeros_wrench(prior["bucket_origin"]),
        **{rock_id: _zeros_wrench(prior_rocks[rock_id]["position"])
           for rock_id in ROCK_IDS},
    }
    accumulated = {
        "bucket_mpm_ns": 0.0,
        "bucket_rigid_ns": 0.0,
        "bucket_total_ns": 0.0,
        "rock_mpm_ns": {rock_id: 0.0 for rock_id in ROCK_IDS},
        "bucket_rigid_by_rock_ns": {rock_id: 0.0 for rock_id in ROCK_IDS},
    }
    contact_frames = {"bucket": 0, **{rock_id: 0 for rock_id in ROCK_IDS}}
    contact_samples = {"bucket": 0, **{rock_id: 0 for rock_id in ROCK_IDS}}
    maximum_echo_residual = 0.0
    maximum_reference_residual = 0.0
    maximum_rigid_force_residual = 0.0
    maximum_rigid_torque_residual = 0.0
    maximum_total_force_residual = 0.0
    maximum_total_torque_residual = 0.0
    maximum_forbidden = 0
    maximum_closure = 0.0
    ownership_totals = {
        "sample_count": 0, "nonfinite_sample_count": 0,
        "active_sample_count": 0, "unmapped_active_sample_count": 0,
    }
    mapping_disjoint = True
    trace = []
    try:
        fixture = SharedMatrixFixture(
            config, r3_c_config, r3_b_config, repository, inventory, device,
            bridge.initial, mode)
        for frame in range(frames):
            sent = {
                body: {
                    "force": (np.asarray(wrench["force"], dtype=np.float64).copy()
                              if feedback else np.zeros(3)),
                    "torque": (np.asarray(wrench["torque"], dtype=np.float64).copy()
                               if feedback else np.zeros(3)),
                    "point": np.asarray(
                        wrench["point"], dtype=np.float64).copy(),
                }
                for body, wrench in wrenches.items()
            }
            state = bridge.step(
                dt, int(probe["mujoco_internal_substeps"]),
                probe["command"], sent)
            applied_rocks = {
                item["rock_id"]: item for item in state["applied_rocks_mpm"]
            }
            maximum_echo_residual = max(
                maximum_echo_residual,
                float(np.linalg.norm(
                    np.asarray(state["applied_bucket_mpm_force_world_n"]) -
                    sent["bucket"]["force"])),
                float(np.linalg.norm(
                    np.asarray(state["applied_bucket_mpm_torque_world_nm"]) -
                    sent["bucket"]["torque"])))
            maximum_reference_residual = max(
                maximum_reference_residual,
                float(np.linalg.norm(
                    np.asarray(state["bucket_interval_reference_point_world_m"]) -
                    sent["bucket"]["point"])))
            for rock_id in ROCK_IDS:
                applied = applied_rocks[rock_id]
                maximum_echo_residual = max(
                    maximum_echo_residual,
                    float(np.linalg.norm(
                        np.asarray(applied["force_world_n"]) -
                        sent[rock_id]["force"])),
                    float(np.linalg.norm(
                        np.asarray(applied["torque_world_nm"]) -
                        sent[rock_id]["torque"])))
                maximum_reference_residual = max(
                    maximum_reference_residual,
                    float(np.linalg.norm(
                        np.asarray(applied["reference_point_world_m"]) -
                        sent[rock_id]["point"])))
            rigid_impulse = np.asarray(
                state["bucket_rock_impulse_world_ns"], dtype=np.float64)
            rigid_angular_impulse = np.asarray(
                state["bucket_rock_angular_impulse_world_nms"], dtype=np.float64)
            rigid_by_rock = {
                item["rock_id"]: item for item in state["bucket_rigid_by_rock"]
            }
            summed_rigid_impulse = np.sum([
                np.asarray(rigid_by_rock[rock_id]["bucket_impulse_world_ns"],
                           dtype=np.float64)
                for rock_id in ROCK_IDS], axis=0)
            summed_rigid_angular = np.sum([
                np.asarray(
                    rigid_by_rock[rock_id]
                    ["bucket_angular_impulse_about_interval_reference_world_nms"],
                    dtype=np.float64)
                for rock_id in ROCK_IDS], axis=0)
            maximum_rigid_force_residual = max(
                maximum_rigid_force_residual,
                float(np.linalg.norm(rigid_impulse - summed_rigid_impulse)) / dt)
            maximum_rigid_torque_residual = max(
                maximum_rigid_torque_residual,
                float(np.linalg.norm(
                    rigid_angular_impulse - summed_rigid_angular)) / dt)
            fixture.assign(state)
            current_rocks = _rock_map(state)
            origins = {
                "bucket": np.asarray(state["bucket_origin"], dtype=np.float64),
                **{rock_id: np.asarray(
                    current_rocks[rock_id]["position"], dtype=np.float64)
                   for rock_id in ROCK_IDS},
            }
            measured, ownership = fixture.step_and_collect(dt, origins)
            for body_name in ["bucket", *ROCK_IDS]:
                force, torque, samples = measured[body_name]
                wrenches[body_name] = {
                    "force": force,
                    "torque": torque,
                    "point": origins[body_name],
                }
                contact_frames[body_name] += int(samples > 0)
                contact_samples[body_name] += samples
            bucket_mpm_force = wrenches["bucket"]["force"]
            bucket_mpm_torque_current = wrenches["bucket"]["torque"]
            bucket_mpm_torque_interval = (
                bucket_mpm_torque_current + np.cross(
                    origins["bucket"] - sent["bucket"]["point"],
                    bucket_mpm_force))
            rigid_force = rigid_impulse / dt
            rigid_torque = rigid_angular_impulse / dt
            total_force = bucket_mpm_force + rigid_force
            total_torque = bucket_mpm_torque_interval + rigid_torque
            maximum_total_force_residual = max(
                maximum_total_force_residual,
                float(np.linalg.norm(
                    total_force - bucket_mpm_force - rigid_force)))
            maximum_total_torque_residual = max(
                maximum_total_torque_residual,
                float(np.linalg.norm(
                    total_torque - bucket_mpm_torque_interval - rigid_torque)))
            accumulated["bucket_mpm_ns"] += float(
                np.linalg.norm(bucket_mpm_force)) * dt
            accumulated["bucket_rigid_ns"] += float(np.linalg.norm(rigid_impulse))
            accumulated["bucket_total_ns"] += float(np.linalg.norm(total_force)) * dt
            for rock_id in ROCK_IDS:
                accumulated["rock_mpm_ns"][rock_id] += float(
                    np.linalg.norm(wrenches[rock_id]["force"])) * dt
                accumulated["bucket_rigid_by_rock_ns"][rock_id] += float(
                    np.linalg.norm(np.asarray(
                        rigid_by_rock[rock_id]["bucket_impulse_world_ns"])))
            maximum_forbidden = max(
                maximum_forbidden, int(state["forbidden_contact_count"]))
            maximum_closure = max(maximum_closure, float(state["closure_m"]))
            for key in ownership_totals:
                ownership_totals[key] += ownership[key]
            mapping_disjoint = mapping_disjoint and ownership["mapping_disjoint"]
            trace.append({
                "frame": frame + 1,
                "time_s": float(state["time_s"]),
                "bucket_mpm_force_world_n": bucket_mpm_force.tolist(),
                "bucket_rigid_impulse_world_ns": rigid_impulse.tolist(),
                "bucket_total_force_world_n": total_force.tolist(),
                "rock_mpm_force_world_n": {
                    rock_id: wrenches[rock_id]["force"].tolist()
                    for rock_id in ROCK_IDS
                },
                "bucket_rigid_impulse_by_rock_world_ns": {
                    rock_id: rigid_by_rock[rock_id]
                    ["bucket_impulse_world_ns"] for rock_id in ROCK_IDS
                },
                "ownership": ownership,
            })
    finally:
        bridge.close()
    if fixture is None:
        raise RuntimeError("r3_c_shared_fixture_not_created")
    trace_bytes = json.dumps(
        trace, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return {
        "qualification_mode": mode,
        "feedback_enabled": feedback,
        "frames": frames,
        "simulated_time_s": frames * dt,
        "final_mechanism_qpos": state["mechanism_qpos"],
        "final_rock_qpos": {
            rock_id: _rock_map(state)[rock_id]["qpos"] for rock_id in ROCK_IDS
        },
        "accumulated_impulse_magnitudes": accumulated,
        "mpm_contact_frames": contact_frames,
        "mpm_contact_samples": contact_samples,
        "maximum_applied_wrench_echo_residual": maximum_echo_residual,
        "maximum_reference_point_residual_m": maximum_reference_residual,
        "maximum_per_rock_rigid_force_partition_residual_n":
            maximum_rigid_force_residual,
        "maximum_per_rock_rigid_torque_partition_residual_nm":
            maximum_rigid_torque_residual,
        "maximum_total_force_partition_residual_n": maximum_total_force_residual,
        "maximum_total_torque_partition_residual_nm": maximum_total_torque_residual,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "maximum_mechanism_closure_m": maximum_closure,
        "collider_ownership": {
            **ownership_totals,
            "mapping_disjoint": mapping_disjoint,
            "rock_id_to_newton_body_index": fixture.rock_body_indices,
            "bucket_newton_body_index": fixture.bucket_body_index,
            "rock_id_to_collider_ids": {
                rock_id: fixture.rock_id_to_collider_ids[rock_id].tolist()
                for rock_id in ROCK_IDS
            },
            "bucket_collider_ids": fixture.bucket_collider_ids.tolist(),
            "static_collider_ids": fixture.static_collider_ids.tolist(),
            "collider_body_map": fixture.collider_body_map.tolist(),
        },
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
    artifact_dir = (Path(args.artifact_dir).resolve()
                    if args.artifact_dir is not None else None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    r3_c_config_path = repository / frozen["r3_c_config_path"]
    r3_c_config = json.loads(r3_c_config_path.read_text(encoding="utf-8"))
    r3_b_config = json.loads(
        (repository / frozen["r3_b_config_path"]).read_text(encoding="utf-8"))
    default_frames = int(round(
        float(config["probe"]["duration_s"]) /
        float(config["probe"]["output_dt_s"])))
    frames = default_frames if args.frames is None else int(args.frames)
    if frames < 1 or frames > default_frames:
        raise ValueError(f"frames_must_be_within_1_and_{default_frames}")
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
            prefix="mixed-material-r3-c-shared-") as temporary:
        work = Path(temporary)
        model = work / "model.xml"
        model_manifest = build_two_rock_candidate(
            r3_c_config_path, repository, inventory, model)
        rigid_disabled_model = work / "model-rigid-disabled.xml"
        _disable_bucket_rigid_contact(model, rigid_disabled_model)
        source = work / "bridge.cc"
        source_manifest = write_bridge(r3_c_config_path, repository, source)
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        episodes = {
            "clear": _episode(
                config, r3_c_config, r3_b_config, repository, inventory,
                binary, rigid_disabled_model, args.device, "clear", frames),
            "mpm_only": _episode(
                config, r3_c_config, r3_b_config, repository, inventory,
                binary, rigid_disabled_model, args.device, "mpm_only", frames),
            "rigid_only": _episode(
                config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "rigid_only", frames),
            "combined": _episode(
                config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", frames),
        }
        model_hashes = {
            "rigid_enabled": sha256(model),
            "bucket_rigid_disabled": sha256(rigid_disabled_model),
        }
        source_sha256 = sha256(source)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, artifact_dir / "generated-model.xml")
            shutil.copy2(
                rigid_disabled_model,
                artifact_dir / "generated-model-bucket-rigid-disabled.xml")
            shutil.copy2(source, artifact_dir / "generated-bridge.cc")
            _write_json(
                artifact_dir / "generated-model-manifest.json", model_manifest)
            _write_json(
                artifact_dir / "generated-bridge-manifest.json", source_manifest)
    probe = config["probe"]
    active = float(probe["minimum_active_component_impulse_ns"])
    inactive = float(probe["maximum_inactive_component_impulse_ns"])
    clear = episodes["clear"]["accumulated_impulse_magnitudes"]
    mpm_only = episodes["mpm_only"]["accumulated_impulse_magnitudes"]
    rigid_only = episodes["rigid_only"]["accumulated_impulse_magnitudes"]
    combined = episodes["combined"]["accumulated_impulse_magnitudes"]
    all_episodes = episodes.values()
    checks = {
        "clear_round_inactive": (
            clear["bucket_mpm_ns"] <= inactive and
            clear["bucket_rigid_ns"] <= inactive and
            all(value <= inactive for value in clear["rock_mpm_ns"].values())),
        "mpm_only_round_partitioned": (
            mpm_only["bucket_mpm_ns"] >= active and
            mpm_only["bucket_rigid_ns"] <= inactive and
            all(value >= active for value in mpm_only["rock_mpm_ns"].values())),
        "rigid_only_round_partitioned": (
            rigid_only["bucket_mpm_ns"] <= inactive and
            rigid_only["bucket_rigid_ns"] >= active and
            all(value <= inactive for value in rigid_only["rock_mpm_ns"].values())),
        "combined_round_components_active": (
            combined["bucket_mpm_ns"] >= active and
            combined["bucket_rigid_ns"] >= active and
            all(value >= active for value in combined["rock_mpm_ns"].values())),
        "per_rock_rigid_partition_closed": all(
            episode["maximum_per_rock_rigid_force_partition_residual_n"] <=
            float(probe["maximum_force_partition_residual_n"]) and
            episode["maximum_per_rock_rigid_torque_partition_residual_nm"] <=
            float(probe["maximum_torque_partition_residual_nm"])
            for episode in all_episodes),
        "bucket_total_partition_closed": all(
            episode["maximum_total_force_partition_residual_n"] <=
            float(probe["maximum_force_partition_residual_n"]) and
            episode["maximum_total_torque_partition_residual_nm"] <=
            float(probe["maximum_torque_partition_residual_nm"])
            for episode in episodes.values()),
        "applied_wrench_echo_exact": all(
            episode["maximum_applied_wrench_echo_residual"] <=
            float(probe["maximum_force_partition_residual_n"])
            for episode in episodes.values()),
        "all_reference_points_exact": all(
            episode["maximum_reference_point_residual_m"] <=
            float(probe["maximum_reference_point_residual_m"])
            for episode in episodes.values()),
        "stable_disjoint_collider_mapping": all(
            episode["collider_ownership"]["mapping_disjoint"] and
            episode["collider_ownership"]["rock_id_to_newton_body_index"] ==
            {"r1-rock-000": 0, "r1-rock-001": 1} and
            episode["collider_ownership"]["bucket_newton_body_index"] == 2
            for episode in episodes.values()),
        "no_unmapped_or_nonfinite_collider_samples": all(
            episode["collider_ownership"]["unmapped_active_sample_count"] == 0 and
            episode["collider_ownership"]["nonfinite_sample_count"] == 0
            for episode in episodes.values()),
        "matrix_finite_and_mass_matched": all(
            episode["matrix"]["finite"] and
            abs(episode["matrix"]["target_mass_error_kg"]) <=
            float(probe["maximum_mass_error_kg"])
            for episode in episodes.values()),
        "forbidden_contacts_zero": all(
            episode["maximum_forbidden_contact_count"] == 0
            for episode in episodes.values()),
        "closure_within_one_mm": all(
            episode["maximum_mechanism_closure_m"] <=
            float(probe["maximum_closure_m"])
            for episode in episodes.values()),
        "mpm_history_reset_complete": all(
            episode["mpm_history_reset"]["passed"]
            for episode in episodes.values()),
        "scope_not_overclaimed": (
            not config["scope"]["repeat_qualification"] and
            not config["scope"]["dt_refinement_qualification"] and
            not config["scope"]["complete_in_process_reset"] and
            not config["scope"]["r3_c_physics_qualified"] and
            not config["scope"]["official_envelope"] and
            not config["scope"]["training"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "r3_c_shared_matrix_four_round_passed" if all(checks.values())
        else "r3_c_shared_matrix_four_round_failed",
        "shared_four_round_qualified": bool(all(checks.values())),
        "r3_c_physics_qualified": False,
        "training_qualified": False,
        "checks": {key: bool(value) for key, value in checks.items()},
        "episodes": episodes,
        "backend": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device_requested": args.device,
            "device_resolved": str(wp.get_device()),
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "r3_c_config_sha256": sha256(r3_c_config_path),
            "derived_model_sha256": model_hashes,
            "model_manifest": model_manifest,
            "bridge_source_sha256": source_sha256,
            "bridge_source_manifest": source_manifest,
            "persisted_generated_artifacts": (
                [
                    "generated-model.xml",
                    "generated-model-bucket-rigid-disabled.xml",
                    "generated-model-manifest.json",
                    "generated-bridge.cc",
                    "generated-bridge-manifest.json",
                ] if artifact_dir is not None else []),
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "contact_ownership": {
            "mujoco": ["rigid-rock-bucket", "rigid-rock-rock",
                       "rigid-rock-support"],
            "newton_mpm": ["bucket-matrix", "each-rigid-rock-matrix",
                           "matrix-support"],
            "mpm_to_rock_not_added_to_bucket_total": True,
            "duplicate_contact_class_count": 0,
        },
        "scope_limit": (
            "exactly two rocks in one shared small MPM matrix and four contact-"
            "partition rounds; no reconstructed-repeat or dt-refinement gate, "
            "no complete in-process mixed reset, no official envelope, no "
            "calibrated material, no complete excavation, and no training"),
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir")
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
        "shared_four_round_qualified": report["shared_four_round_qualified"],
        "r3_c_physics_qualified": report["r3_c_physics_qualified"],
        "checks": report["checks"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
