#!/usr/bin/env python3
"""Run the R3-B bucket/rock/MPM force-partition qualification probe."""

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
from newton.solvers import SolverImplicitMPM

from build_mixed_coupled_bridge_r3_a import sha256
from build_mixed_coupled_bridge_r3_b import validate_config, write_bridge
from mixed_material_r1 import build_proxy
from mixed_material_r1_mechanism import build_candidate
from run_mixed_material_r3_a_probe import (
    Bridge,
    _compile_bridge,
    _make_transform,
    _read_obj,
    _write_json,
)


REPORT_SCHEMA = "excavation_mixed_material_r3_b_probe_report_v1"


def _quat_wxyz_to_xyzw(value):
    w, x, y, z = [float(item) for item in value]
    return wp.quat(x, y, z, w)


def _quat_xyzw_from_matrix(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


class R3BBridge(Bridge):
    def step_partitioned(self, dt: float, substeps: int, command,
                         bucket_force, bucket_torque, bucket_point,
                         rock_force, rock_torque, rock_point,
                         feedback: bool) -> dict:
        time_s = self.last_state["time_s"] if hasattr(self, "last_state") else self.initial["time_s"]
        fields = ["step", f"{float(time_s):.17g}", f"{dt:.17g}", str(substeps)]
        fields += [f"{float(value):.17g}" for value in command]
        fields.append("2" if feedback else "0")
        if feedback:
            for name, force, torque, point in (
                ("bucket", bucket_force, bucket_torque, bucket_point),
                ("r1-rock-000", rock_force, rock_torque, rock_point),
            ):
                fields.append(name)
                fields += [f"{float(value):.17g}" for value in force]
                fields += [f"{float(value):.17g}" for value in torque]
                fields += [f"{float(value):.17g}" for value in point]
        self._write(" ".join(fields) + "\n")
        self.last_state = self._read_state("step")
        return self.last_state


def _bucket_transform(state: dict):
    quaternion = _quat_xyzw_from_matrix(state["bucket_rotation"])
    return wp.transform(
        wp.vec3(*[float(value) for value in state["bucket_origin"]]),
        wp.quat(*[float(value) for value in quaternion]),
    )


def _assign_bodies(state, bridge_state: dict, rock_linear, rock_angular,
                   device: str) -> None:
    state.body_q.assign(wp.array([
        _make_transform(bridge_state["rock_qpos"][:3],
                        bridge_state["rock_qpos"][3:7]),
        _bucket_transform(bridge_state),
    ], dtype=wp.transform, device=device))
    bucket_linear = bridge_state["bucket_linear_velocity_world_m_s"]
    bucket_angular = bridge_state["bucket_angular_velocity_world_rad_s"]
    state.body_qd.assign(wp.array([
        wp.spatial_vector(*[float(value) for value in rock_linear],
                          *[float(value) for value in rock_angular]),
        wp.spatial_vector(*[float(value) for value in bucket_linear],
                          *[float(value) for value in bucket_angular]),
    ], dtype=wp.spatial_vector, device=device))


class PartitionFixture:
    def __init__(self, config: dict, repository: Path, inventory: Path,
                 device: str, initial_state: dict, mode: str):
        if mode not in {"clear", "mpm_only", "rigid_only", "combined"}:
            raise ValueError("r3_b_unknown_qualification_mode:" + mode)
        bucket_particle_collision = mode in {"mpm_only", "combined"}
        rock_particle_collision = mode == "combined"
        frozen = config["frozen_inputs"]
        patch = config["matrix_patch"]
        r1_config = json.loads(
            (repository / frozen["r1_config_path"]).read_text(encoding="utf-8"))
        if sha256(inventory) != r1_config["bucket"]["inventory_contract_sha256"]:
            raise ValueError("r3_b_inventory_sha256_mismatch")
        proxy = build_proxy(
            json.loads(inventory.read_text(encoding="utf-8")), r1_config)
        vertices, indices = _read_obj(repository / frozen["rock_asset_path"])
        inertia = np.asarray(frozen["rock_inertia_about_com_kg_m2"], dtype=np.float64)

        builder = newton.ModelBuilder()
        self.rock_body = builder.add_body(
            xform=_make_transform(initial_state["rock_qpos"][:3],
                                  initial_state["rock_qpos"][3:7]),
            com=wp.vec3(0.0), inertia=wp.mat33(inertia.astype(np.float32)),
            mass=float(frozen["rock_mass_kg"]), label="r1-rock-000",
            lock_inertia=True)
        builder.add_shape_mesh(
            self.rock_body,
            mesh=newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True),
            scale=wp.vec3(float(frozen["rock_equivalent_diameter_m"])),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(patch["material"]["friction"]),
                has_particle_collision=rock_particle_collision),
            label="r1-rock-000-mpm-collider")

        self.bucket_body = builder.add_body(
            xform=_bucket_transform(initial_state), mass=100.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)),
            label="bucket", lock_inertia=True, is_kinematic=True)
        shape_config = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(patch["material"]["friction"]),
            has_particle_collision=bucket_particle_collision)
        for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
            builder.add_shape_box(
                self.bucket_body,
                xform=wp.transform(
                    wp.vec3(*[float(value) for value in box["position_bucket_m"]]),
                    _quat_wxyz_to_xyzw(box["quaternion_wxyz"])),
                hx=float(box["half_size_m"][0]),
                hy=float(box["half_size_m"][1]),
                hz=float(box["half_size_m"][2]),
                cfg=shape_config, label="r3b_" + box["name"])
        for side in proxy["side_plate_meshes"]:
            side_vertices = np.asarray(side["vertices_bucket_m"], dtype=np.float32)
            side_indices = np.asarray(side["faces"], dtype=np.int32).reshape(-1)
            builder.add_shape_mesh(
                self.bucket_body,
                mesh=newton.Mesh(side_vertices, side_indices,
                                 compute_inertia=False, is_solid=True),
                scale=wp.vec3(1.0), cfg=shape_config,
                label="r3b_" + side["name"])
        support = patch["support_fixture"]
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*support["position_world_m"]),
                               wp.quat_identity()),
            hx=float(support["half_size_m"][0]),
            hy=float(support["half_size_m"][1]),
            hz=float(support["half_size_m"][2]),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(support["friction"])),
            label="r1_ground_push_shelf")
        self.collider_model = builder.finalize(device=device)

        lower = np.asarray(patch["lower_world_m"], dtype=np.float64)
        size = np.asarray(patch["size_xyz_m"], dtype=np.float64)
        spacing = float(patch["particle_spacing_m"])
        dimensions = np.rint(size / spacing).astype(int) + 1
        if not np.allclose((dimensions - 1) * spacing, size, atol=1.0e-12):
            raise ValueError("r3_b_patch_spacing_contract")
        particle_count = int(np.prod(dimensions, dtype=np.int64))
        self.target_mass_kg = float(np.prod(size) * float(patch["density_kg_m3"]))
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        sand_builder.add_particle_grid(
            pos=wp.vec3(lower), rot=wp.quat_identity(), vel=wp.vec3(0.0),
            dim_x=int(dimensions[0]), dim_y=int(dimensions[1]),
            dim_z=int(dimensions[2]), cell_x=spacing, cell_y=spacing,
            cell_z=spacing, mass=self.target_mass_kg / particle_count,
            jitter=0.0, radius_mean=0.45 * spacing,
            custom_attributes={"mpm:friction": float(patch["material"]["friction"])})
        self.model = sand_builder.finalize(device=device)
        self.model.set_gravity(tuple(float(value) for value in patch["gravity_m_s2"]))
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
        options.collider_velocity_mode = config["probe"]["collider_velocity_mode"]
        self.solver = SolverImplicitMPM(self.model, config=options)
        self.solver.setup_collider(model=self.collider_model)
        self.state = self.model.state()
        collider_state = self.collider_model.state()
        self.state.body_q = wp.clone(collider_state.body_q)
        self.state.body_qd = wp.clone(collider_state.body_qd)
        self.state.body_f = wp.zeros_like(collider_state.body_f)
        self.device = device
        _assign_bodies(self.state, initial_state, np.zeros(3), np.zeros(3), device)
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy()
        self.collider_body_map = body_map.copy()
        self.static_collider_ids = np.flatnonzero(body_map == -1)
        self.rock_collider_ids = np.flatnonzero(body_map == self.rock_body)
        self.bucket_collider_ids = np.flatnonzero(body_map == self.bucket_body)
        # Newton packs all shapes attached to one body into one stable collider.
        if (self.rock_collider_ids.size != int(rock_particle_collision) or
                self.bucket_collider_ids.size != int(bucket_particle_collision)):
            raise RuntimeError("r3_b_collider_map:" + json.dumps(body_map.tolist()))
        self.initial_q = self.state.particle_q.numpy().copy()
        self.initial_qd = self.state.particle_qd.numpy().copy()

    def assign(self, bridge_state: dict, rock_linear, rock_angular) -> None:
        _assign_bodies(self.state, bridge_state, rock_linear, rock_angular, self.device)

    def _collect(self, impulses, positions, ids, selected_ids, origin, dt):
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

    def step_and_collect(self, dt: float, rock_origin, bucket_origin):
        self.solver.step(self.state, self.state, None, None, dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        finite = np.isfinite(impulse).all(axis=1) & np.isfinite(position).all(axis=1)
        active = finite & (np.linalg.norm(impulse, axis=1) > 1.0e-12)
        bucket_mask = active & np.isin(ids, self.bucket_collider_ids)
        rock_mask = active & np.isin(ids, self.rock_collider_ids)
        static_mask = active & np.isin(ids, self.static_collider_ids)
        mapped_mask = bucket_mask | rock_mask | static_mask
        ownership = {
            "sample_count": int(ids.size),
            "nonfinite_sample_count": int(np.count_nonzero(~finite)),
            "active_sample_count": int(np.count_nonzero(active)),
            "bucket_active_sample_count": int(np.count_nonzero(bucket_mask)),
            "rock_active_sample_count": int(np.count_nonzero(rock_mask)),
            "static_active_sample_count": int(np.count_nonzero(static_mask)),
            "unmapped_active_sample_count": int(np.count_nonzero(active & ~mapped_mask)),
            "mapping_disjoint": bool(
                not np.intersect1d(self.bucket_collider_ids, self.rock_collider_ids).size and
                not np.intersect1d(self.bucket_collider_ids, self.static_collider_ids).size and
                not np.intersect1d(self.rock_collider_ids, self.static_collider_ids).size),
        }
        rock = self._collect(
            impulse, position, ids, self.rock_collider_ids,
            np.asarray(rock_origin, dtype=np.float64), dt)
        bucket = self._collect(
            impulse, position, ids, self.bucket_collider_ids,
            np.asarray(bucket_origin, dtype=np.float64), dt)
        return rock, bucket, ownership

    def metrics(self):
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        mass = float(np.sum(self.model.particle_mass.numpy(), dtype=np.float64))
        return {
            "particle_count": int(q.shape[0]),
            "target_mass_kg": self.target_mass_kg,
            "total_mass_kg": mass,
            "target_mass_error_kg": mass - self.target_mass_kg,
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "minimum_particle_z_m": float(np.min(q[:, 2])),
            "maximum_particle_speed_m_s": float(np.max(np.linalg.norm(qd, axis=1))),
            "particle_q_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
            "particle_qd_sha256": hashlib.sha256(qd.tobytes()).hexdigest(),
        }

    def reset_audit(self):
        self.state.particle_q.assign(wp.array(self.initial_q, dtype=wp.vec3, device=self.device))
        self.state.particle_qd.assign(wp.array(self.initial_qd, dtype=wp.vec3, device=self.device))
        self.solver.reset(self.state)
        wp.synchronize()
        mpm = self.state.mpm
        identity = np.eye(3, dtype=np.float32)
        checks = {
            "particle_q_exact": np.array_equal(self.state.particle_q.numpy(), self.initial_q),
            "particle_qd_exact": np.array_equal(self.state.particle_qd.numpy(), self.initial_qd),
            "particle_elastic_strain_identity": np.array_equal(
                mpm.particle_elastic_strain.numpy(),
                np.broadcast_to(identity, mpm.particle_elastic_strain.numpy().shape)),
            "particle_transform_identity": np.array_equal(
                mpm.particle_transform.numpy(),
                np.broadcast_to(identity, mpm.particle_transform.numpy().shape)),
            "particle_qd_grad_zero": not np.any(mpm.particle_qd_grad.numpy()),
            "particle_stress_zero": not np.any(mpm.particle_stress.numpy()),
            "particle_Jp_one": np.all(mpm.particle_Jp.numpy() == 1.0),
        }
        return {"checks": {key: bool(value) for key, value in checks.items()},
                "passed": bool(all(checks.values()))}


def _episode(config, repository, inventory, binary, model, device, frames,
             feedback, mode):
    probe = config["probe"]
    frozen = config["frozen_inputs"]
    dt = float(probe["output_dt_s"])
    substeps = int(probe["mujoco_internal_substeps"])
    bridge = R3BBridge(binary, model, config["bridge"]["ready_schema"])
    fixture = None
    bucket_force = np.zeros(3)
    bucket_torque = np.zeros(3)
    rock_force = np.zeros(3)
    rock_torque = np.zeros(3)
    maximum_force_residual = 0.0
    maximum_torque_residual = 0.0
    maximum_applied_force_residual = 0.0
    maximum_applied_torque_residual = 0.0
    maximum_reference_point_residual = 0.0
    bucket_mpm_contact_frames = 0
    bucket_mpm_samples = 0
    rock_mpm_contact_frames = 0
    rigid_contact_substeps = 0
    maximum_closure = 0.0
    maximum_forbidden = 0
    ownership_totals = {
        "sample_count": 0, "nonfinite_sample_count": 0,
        "active_sample_count": 0, "bucket_active_sample_count": 0,
        "rock_active_sample_count": 0, "static_active_sample_count": 0,
        "unmapped_active_sample_count": 0,
    }
    mapping_disjoint = True
    accumulated_impulse = {
        "bucket_mpm_ns": 0.0, "bucket_rigid_ns": 0.0,
        "bucket_measured_total_ns": 0.0, "bucket_applied_total_ns": 0.0,
    }
    peak = {"bucket_mpm_force_n": 0.0, "bucket_rigid_force_n": 0.0,
            "bucket_measured_total_force_n": 0.0,
            "bucket_applied_total_force_n": 0.0,
            "bucket_measured_total_torque_nm": 0.0,
            "bucket_applied_total_torque_nm": 0.0}
    trace = []
    final_mechanism_qpos = None
    try:
        fixture = PartitionFixture(
            config, repository, inventory, device, bridge.initial, mode)
        for frame in range(frames):
            prior = bridge.last_state if hasattr(bridge, "last_state") else bridge.initial
            sent_bucket_force = bucket_force.copy() if feedback else np.zeros(3)
            sent_bucket_torque = bucket_torque.copy() if feedback else np.zeros(3)
            interval_reference = np.asarray(prior["bucket_origin"], dtype=np.float64)
            state = bridge.step_partitioned(
                dt, substeps, probe["command"],
                bucket_force, bucket_torque, interval_reference,
                rock_force, rock_torque, prior["rock_position"], feedback)
            final_mechanism_qpos = list(state["mechanism_qpos"])
            applied_bucket_force = np.asarray(
                state["applied_bucket_mpm_force_world_n"], dtype=np.float64)
            applied_bucket_torque = np.asarray(
                state["applied_bucket_mpm_torque_world_nm"], dtype=np.float64)
            reported_reference = np.asarray(
                state["bucket_interval_reference_point_world_m"], dtype=np.float64)
            rigid_force = np.asarray(state["bucket_rock_impulse_world_ns"], dtype=np.float64) / dt
            rigid_torque = np.asarray(
                state["bucket_rock_angular_impulse_world_nms"], dtype=np.float64) / dt
            applied_total_force = applied_bucket_force + rigid_force
            applied_total_torque = applied_bucket_torque + rigid_torque
            maximum_applied_force_residual = max(
                maximum_applied_force_residual,
                float(np.linalg.norm(applied_bucket_force - sent_bucket_force)))
            maximum_applied_torque_residual = max(
                maximum_applied_torque_residual,
                float(np.linalg.norm(applied_bucket_torque - sent_bucket_torque)))
            maximum_reference_point_residual = max(
                maximum_reference_point_residual,
                float(np.linalg.norm(reported_reference - interval_reference)))
            peak["bucket_rigid_force_n"] = max(
                peak["bucket_rigid_force_n"], float(np.linalg.norm(rigid_force)))
            peak["bucket_applied_total_force_n"] = max(
                peak["bucket_applied_total_force_n"],
                float(np.linalg.norm(applied_total_force)))
            peak["bucket_applied_total_torque_nm"] = max(
                peak["bucket_applied_total_torque_nm"],
                float(np.linalg.norm(applied_total_torque)))
            accumulated_impulse["bucket_rigid_ns"] += float(np.linalg.norm(rigid_force)) * dt
            accumulated_impulse["bucket_applied_total_ns"] += (
                float(np.linalg.norm(applied_total_force)) * dt)
            rigid_contact_substeps += int(state["contact_substeps"]["rock_bucket"])
            maximum_closure = max(maximum_closure, float(state["closure_m"]))
            maximum_forbidden = max(
                maximum_forbidden, int(state["forbidden_contact_count"]))
            # Backward collider velocities use the prior/current poses, so no
            # free-body velocity subtraction is valid or required here.
            fixture.assign(
                state,
                np.asarray(state["rock_linear_velocity_world_m_s"], dtype=np.float64),
                np.asarray(state["rock_angular_velocity_world_rad_s"], dtype=np.float64))
            next_reference = np.asarray(state["bucket_origin"], dtype=np.float64)
            rock, bucket, ownership = fixture.step_and_collect(
                dt, state["rock_position"], next_reference)
            rock_force, rock_torque, rock_samples = rock
            bucket_force, bucket_torque, bucket_samples = bucket
            measured_bucket_torque_at_interval_reference = (
                bucket_torque + np.cross(
                    next_reference - interval_reference, bucket_force))
            measured_total_force = bucket_force + rigid_force
            measured_total_torque = (
                measured_bucket_torque_at_interval_reference + rigid_torque)
            maximum_force_residual = max(
                maximum_force_residual,
                float(np.linalg.norm(
                    measured_total_force - bucket_force - rigid_force)))
            maximum_torque_residual = max(
                maximum_torque_residual,
                float(np.linalg.norm(
                    measured_total_torque -
                    measured_bucket_torque_at_interval_reference - rigid_torque)))
            peak["bucket_mpm_force_n"] = max(
                peak["bucket_mpm_force_n"], float(np.linalg.norm(bucket_force)))
            peak["bucket_measured_total_force_n"] = max(
                peak["bucket_measured_total_force_n"],
                float(np.linalg.norm(measured_total_force)))
            peak["bucket_measured_total_torque_nm"] = max(
                peak["bucket_measured_total_torque_nm"],
                float(np.linalg.norm(measured_total_torque)))
            accumulated_impulse["bucket_mpm_ns"] += float(np.linalg.norm(bucket_force)) * dt
            accumulated_impulse["bucket_measured_total_ns"] += (
                float(np.linalg.norm(measured_total_force)) * dt)
            for key in ownership_totals:
                ownership_totals[key] += ownership[key]
            mapping_disjoint = mapping_disjoint and ownership["mapping_disjoint"]
            bucket_mpm_contact_frames += int(bucket_samples > 0)
            bucket_mpm_samples += bucket_samples
            rock_mpm_contact_frames += int(rock_samples > 0)
            trace.append({
                "frame": frame + 1,
                "time_s": float(state["time_s"]),
                "mechanism_qpos": [float(value) for value in state["mechanism_qpos"]],
                "interval_reference_world_m": interval_reference.tolist(),
                "next_reference_world_m": next_reference.tolist(),
                "applied_mpm_force_world_n": applied_bucket_force.tolist(),
                "new_measured_mpm_force_world_n": bucket_force.tolist(),
                "rigid_force_world_n": rigid_force.tolist(),
                "applied_total_force_world_n": applied_total_force.tolist(),
                "measured_total_force_world_n": measured_total_force.tolist(),
                "applied_mpm_torque_about_interval_reference_world_nm":
                    applied_bucket_torque.tolist(),
                "new_measured_mpm_torque_about_next_reference_world_nm":
                    bucket_torque.tolist(),
                "new_measured_mpm_torque_about_interval_reference_world_nm":
                    measured_bucket_torque_at_interval_reference.tolist(),
                "rigid_torque_about_interval_reference_world_nm": rigid_torque.tolist(),
                "applied_total_torque_about_interval_reference_world_nm":
                    applied_total_torque.tolist(),
                "measured_total_torque_about_interval_reference_world_nm":
                    measured_total_torque.tolist(),
                "bucket_mpm_samples": bucket_samples,
                "rock_mpm_samples": rock_samples,
                "ownership": ownership,
            })
    finally:
        bridge.close()
    trace_bytes = json.dumps(trace, separators=(",", ":"), allow_nan=False).encode()
    if fixture is None:
        raise RuntimeError("r3_b_fixture_not_created")
    return {
        "qualification_mode": mode,
        "feedback_enabled": feedback,
        "frames": frames,
        "simulated_time_s": frames * dt,
        "final_mechanism_qpos": final_mechanism_qpos,
        "bucket_mpm_contact_frames": bucket_mpm_contact_frames,
        "bucket_mpm_contact_samples": bucket_mpm_samples,
        "rock_mpm_contact_frames": rock_mpm_contact_frames,
        "bucket_rigid_contact_substeps": rigid_contact_substeps,
        "maximum_mechanism_closure_m": maximum_closure,
        "maximum_forbidden_contact_count": maximum_forbidden,
        "maximum_force_partition_residual_n": maximum_force_residual,
        "maximum_torque_partition_residual_nm": maximum_torque_residual,
        "maximum_applied_bucket_force_echo_residual_n":
            maximum_applied_force_residual,
        "maximum_applied_bucket_torque_echo_residual_nm":
            maximum_applied_torque_residual,
        "maximum_interval_reference_point_residual_m":
            maximum_reference_point_residual,
        "accumulated_impulse_magnitudes": accumulated_impulse,
        "collider_ownership": {
            **ownership_totals,
            "mapping_disjoint": mapping_disjoint,
            "collider_body_map": fixture.collider_body_map.tolist(),
            "static_collider_ids": fixture.static_collider_ids.tolist(),
            "rock_collider_ids": fixture.rock_collider_ids.tolist(),
            "bucket_collider_ids": fixture.bucket_collider_ids.tolist(),
        },
        "peaks": peak,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "trace": trace,
        "matrix": fixture.metrics(),
        "reset_audit": fixture.reset_audit(),
    }


def run(args):
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    inventory = Path(args.inventory).resolve()
    output = Path(args.output).resolve()
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
    with tempfile.TemporaryDirectory(prefix="mixed-material-r3-b-") as temporary:
        work = Path(temporary)
        source = work / "bridge.cc"
        source_manifest = write_bridge(config_path, repository, source)
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source, binary,
            [args.mechanism_header_dir, args.controller_header_dir, args.mujoco_include],
            Path(args.mujoco_library_dir))
        models = {}
        model_manifests = {}
        model_hashes = {}
        for model_key, scenario in (("push", probe["r1_scenario"]),
                                    ("clear", "bucket_push_clear")):
            model = work / f"model-{model_key}.xml"
            model_manifests[model_key] = build_candidate(
                repository / config["frozen_inputs"]["mechanism_model_path"],
                model, repository, inventory,
                repository / config["frozen_inputs"]["r1_config_path"], scenario)
            model_hashes[model_key] = sha256(model)
            models[model_key] = model
        clear = _episode(
            config, repository, inventory, binary, models["clear"], args.device,
            frames, True, "clear")
        mpm_only = _episode(
            config, repository, inventory, binary, models["clear"], args.device,
            frames, True, "mpm_only")
        rigid_only = _episode(
            config, repository, inventory, binary, models["push"], args.device,
            frames, True, "rigid_only")
        coupled_first = _episode(
            config, repository, inventory, binary, models["push"], args.device,
            frames, True, "combined")
        coupled_repeat = _episode(
            config, repository, inventory, binary, models["push"], args.device,
            frames, True, "combined")
        no_mpm_feedback = _episode(
            config, repository, inventory, binary, models["push"], args.device,
            frames, False, "combined")
        fine_config = json.loads(json.dumps(config))
        fine_config["probe"]["output_dt_s"] = float(probe["output_dt_s"]) / 2.0
        # The 40/20 pair keeps the MuJoCo internal step fixed at 1/4800 s,
        # isolating refinement of the outer staggered coupling step.
        fine_config["probe"]["mujoco_internal_substeps"] = (
            int(probe["mujoco_internal_substeps"]) // 2)
        if fine_config["probe"]["mujoco_internal_substeps"] < 20:
            raise ValueError("r3_b_dt_refinement_substeps")
        refined_frames = frames * 2
        dt_refined = _episode(
            fine_config, repository, inventory, binary, models["push"],
            args.device, refined_frames, True, "combined")
        dt_refined_repeat = _episode(
            fine_config, repository, inventory, binary, models["push"],
            args.device, refined_frames, True, "combined")
    qpos_delta = float(np.max(np.abs(
        np.asarray(coupled_first["final_mechanism_qpos"]) -
        np.asarray(no_mpm_feedback["final_mechanism_qpos"]))))
    deterministic = (
        coupled_first["trace_sha256"] == coupled_repeat["trace_sha256"] and
        coupled_first["matrix"]["particle_q_sha256"] ==
        coupled_repeat["matrix"]["particle_q_sha256"] and
        coupled_first["matrix"]["particle_qd_sha256"] ==
        coupled_repeat["matrix"]["particle_qd_sha256"])
    repeat_qpos_delta = float(np.max(np.abs(
        np.asarray(coupled_first["final_mechanism_qpos"]) -
        np.asarray(coupled_repeat["final_mechanism_qpos"]))))

    def relative_difference(first, second):
        return abs(first - second) / max(abs(first), abs(second), 1.0e-30)

    repeat_impulse_relative_differences = {
        key: relative_difference(
            coupled_first["accumulated_impulse_magnitudes"][key],
            coupled_repeat["accumulated_impulse_magnitudes"][key])
        for key in ("bucket_mpm_ns", "bucket_rigid_ns", "bucket_measured_total_ns")
    }
    refined_repeat_qpos_delta = float(np.max(np.abs(
        np.asarray(dt_refined["final_mechanism_qpos"]) -
        np.asarray(dt_refined_repeat["final_mechanism_qpos"]))))
    refined_repeat_impulse_relative_differences = {
        key: relative_difference(
            dt_refined["accumulated_impulse_magnitudes"][key],
            dt_refined_repeat["accumulated_impulse_magnitudes"][key])
        for key in ("bucket_mpm_ns", "bucket_rigid_ns", "bucket_measured_total_ns")
    }
    coarse_mean_qpos = 0.5 * (
        np.asarray(coupled_first["final_mechanism_qpos"]) +
        np.asarray(coupled_repeat["final_mechanism_qpos"]))
    refined_mean_qpos = 0.5 * (
        np.asarray(dt_refined["final_mechanism_qpos"]) +
        np.asarray(dt_refined_repeat["final_mechanism_qpos"]))
    dt_refinement_qpos_delta = float(np.max(np.abs(
        coarse_mean_qpos - refined_mean_qpos)))
    dt_refinement_impulse_relative_differences = {
        key: relative_difference(
            0.5 * (
                coupled_first["accumulated_impulse_magnitudes"][key] +
                coupled_repeat["accumulated_impulse_magnitudes"][key]),
            0.5 * (
                dt_refined["accumulated_impulse_magnitudes"][key] +
                dt_refined_repeat["accumulated_impulse_magnitudes"][key]))
        for key in ("bucket_mpm_ns", "bucket_rigid_ns", "bucket_measured_total_ns")
    }
    active_impulse = float(probe["minimum_active_component_impulse_ns"])
    inactive_impulse = float(probe["maximum_inactive_component_impulse_ns"])
    checks = {
        "clear_round_inactive": (
            clear["accumulated_impulse_magnitudes"]["bucket_mpm_ns"] <=
            inactive_impulse and
            clear["accumulated_impulse_magnitudes"]["bucket_rigid_ns"] <=
            inactive_impulse),
        "mpm_only_round_partitioned": (
            mpm_only["accumulated_impulse_magnitudes"]["bucket_mpm_ns"] >=
            active_impulse and
            mpm_only["accumulated_impulse_magnitudes"]["bucket_rigid_ns"] <=
            inactive_impulse),
        "rigid_only_round_partitioned": (
            rigid_only["accumulated_impulse_magnitudes"]["bucket_mpm_ns"] <=
            inactive_impulse and
            rigid_only["accumulated_impulse_magnitudes"]["bucket_rigid_ns"] >=
            active_impulse),
        "combined_round_components_active": (
            coupled_first["accumulated_impulse_magnitudes"]["bucket_mpm_ns"] >=
            active_impulse and
            coupled_first["accumulated_impulse_magnitudes"]["bucket_rigid_ns"] >=
            active_impulse),
        "coupled_finite": coupled_first["matrix"]["finite"],
        "repeat_finite": coupled_repeat["matrix"]["finite"],
        "control_finite": no_mpm_feedback["matrix"]["finite"],
        "bucket_mpm_contact_nonzero": (
            coupled_first["bucket_mpm_contact_frames"] >=
            int(probe["minimum_bucket_mpm_contact_frames"])),
        "bucket_rigid_contact_nonzero": (
            coupled_first["bucket_rigid_contact_substeps"] >=
            int(probe["minimum_bucket_rigid_contact_substeps"])),
        "bucket_mpm_wrench_nonzero": coupled_first["peaks"]["bucket_mpm_force_n"] > 0.0,
        "bucket_rigid_wrench_nonzero": coupled_first["peaks"]["bucket_rigid_force_n"] > 0.0,
        "bucket_total_wrench_nonzero": (
            coupled_first["peaks"]["bucket_measured_total_force_n"] > 0.0),
        "force_partition_closed": (
            coupled_first["maximum_force_partition_residual_n"] <=
            float(probe["maximum_force_partition_residual_n"])),
        "torque_partition_closed": (
            coupled_first["maximum_torque_partition_residual_nm"] <=
            float(probe["maximum_torque_partition_residual_nm"])),
        "applied_wrench_echo_exact": (
            coupled_first["maximum_applied_bucket_force_echo_residual_n"] <=
            float(probe["maximum_force_partition_residual_n"]) and
            coupled_first["maximum_applied_bucket_torque_echo_residual_nm"] <=
            float(probe["maximum_torque_partition_residual_nm"])),
        "interval_reference_exact": (
            coupled_first["maximum_interval_reference_point_residual_m"] <= 1.0e-12),
        "collider_mapping_disjoint": (
            coupled_first["collider_ownership"]["mapping_disjoint"]),
        "no_unmapped_active_collider_samples": (
            coupled_first["collider_ownership"]["unmapped_active_sample_count"] == 0),
        "finite_collider_impulse_samples": (
            coupled_first["collider_ownership"]["nonfinite_sample_count"] == 0),
        "feedback_changes_mechanism": (
            float(probe["minimum_feedback_mechanism_qpos_difference_m"]) <= qpos_delta <=
            float(probe["maximum_feedback_mechanism_qpos_difference_m"])),
        "bounded_repeat": (
            max(repeat_qpos_delta, refined_repeat_qpos_delta) <=
            float(probe["maximum_repeat_mechanism_qpos_difference"]) and
            max(
                max(repeat_impulse_relative_differences.values()),
                max(refined_repeat_impulse_relative_differences.values())) <=
            float(probe["maximum_repeat_component_impulse_relative_difference"])),
        "dt_refinement_bounded": (
            dt_refinement_qpos_delta <=
            float(probe["maximum_dt_refinement_mechanism_qpos_difference_m"]) and
            max(
                dt_refinement_impulse_relative_differences[key]
                for key in probe["dt_refinement_acceptance_impulses"]) <=
            float(probe[
                "maximum_dt_refinement_component_impulse_relative_difference"])),
        "forbidden_contacts_zero": max(
            episode["maximum_forbidden_contact_count"]
            for episode in (clear, mpm_only, rigid_only, coupled_first,
                            coupled_repeat, no_mpm_feedback, dt_refined,
                            dt_refined_repeat)) == 0,
        "closure_within_one_mm": max(
            episode["maximum_mechanism_closure_m"]
            for episode in (clear, mpm_only, rigid_only, coupled_first,
                            coupled_repeat, no_mpm_feedback, dt_refined,
                            dt_refined_repeat)) <=
            float(probe["maximum_closure_m"]),
        "matrix_target_mass_matched": all(
            abs(episode["matrix"]["target_mass_error_kg"]) <= 1.0e-5
            for episode in (clear, mpm_only, rigid_only, coupled_first,
                            coupled_repeat, no_mpm_feedback, dt_refined,
                            dt_refined_repeat)),
        "mpm_history_reset_complete": all(
            episode["reset_audit"]["passed"]
            for episode in (clear, mpm_only, rigid_only, coupled_first,
                            coupled_repeat, no_mpm_feedback, dt_refined,
                            dt_refined_repeat)),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "r3_b_bucket_force_partition_probe_passed" if all(checks.values())
        else "r3_b_bucket_force_partition_probe_failed",
        "physics_qualified": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "comparison": {
            "maximum_final_mechanism_qpos_feedback_difference_m": qpos_delta,
            "byte_exact_repeat": deterministic,
            "maximum_final_mechanism_qpos_repeat_difference": repeat_qpos_delta,
            "component_impulse_repeat_relative_differences":
                repeat_impulse_relative_differences,
            "refined_repeat": {
                "byte_exact": (
                    dt_refined["trace_sha256"] ==
                    dt_refined_repeat["trace_sha256"]),
                "maximum_final_mechanism_qpos_difference_m":
                    refined_repeat_qpos_delta,
                "component_impulse_relative_differences":
                    refined_repeat_impulse_relative_differences,
            },
            "dt_refinement": {
                "coarse_output_dt_s": float(probe["output_dt_s"]),
                "refined_output_dt_s": fine_config["probe"]["output_dt_s"],
                "mujoco_internal_dt_preserved": True,
                "comparison_basis": "mean_of_two_reconstructed_episodes_per_dt",
                "maximum_final_mechanism_qpos_difference_m":
                    dt_refinement_qpos_delta,
                "component_impulse_relative_differences":
                    dt_refinement_impulse_relative_differences,
                "acceptance_impulses":
                    probe["dt_refinement_acceptance_impulses"],
                "rigid_component_is_diagnostic_not_converged": (
                    dt_refinement_impulse_relative_differences["bucket_rigid_ns"] >
                    float(probe[
                        "maximum_dt_refinement_component_impulse_relative_difference"])),
            },
        },
        "episodes": {
            "clear": clear,
            "mpm_only": mpm_only,
            "rigid_only": rigid_only,
            "coupled_first": coupled_first,
            "coupled_repeat": coupled_repeat,
            "no_mpm_feedback": no_mpm_feedback,
            "dt_refined_combined": dt_refined,
            "dt_refined_combined_repeat": dt_refined_repeat,
        },
        "backend": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device_requested": args.device, "device_resolved": str(wp.get_device()),
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "derived_model_sha256": model_hashes,
            "model_manifest": model_manifests,
            "bridge_source_manifest": source_manifest,
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "contact_ownership": {
            "mujoco": ["bucket-rigid-rock", "rigid-rock-support-fixture"],
            "newton_mpm": ["bucket-matrix", "rigid-rock-matrix", "matrix-support-fixture"],
            "duplicate_contact_class_count": 0,
        },
        "scope_limit": (
            "one rock and a small elevated MPM patch in the frozen R1 bucket-push fixture; "
            "not multi-rock, not the official envelope, not a calibrated material, not "
            "complete excavation, and not training-qualified"),
    }
    _write_json(output, report)
    return report


def main():
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
    print(json.dumps({"status": report["status"],
                      "physics_qualified": report["physics_qualified"],
                      "checks": report["checks"],
                      "comparison": report["comparison"]},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
