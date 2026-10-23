#!/usr/bin/env python3
"""Bind the audited 47-rock checkpoint to the stationary excavation scene.

This is a read-only, pre-contact gate.  It intentionally adds the qualified
open-bucket Newton collider to a fresh fixture, transfers only the saved grid
warm-start fields, places the mechanism at the approved non-contact operator
start pose, and executes a zero-command hold.  It does not dig, write a new
checkpoint, or qualify the truck receiver as a physics collider.
"""

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

from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    checkpoint_identity,
    load_checkpoint_readonly,
    restore_fixture_prefix,
    transfer_solver_warmstart,
)
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    restore_runtime,
)
from mpm_rigid_free_velocity import predict_free_linear_velocity
from mixed_material_r1 import outward_side_plate_faces
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_layered_mound_three_rock_adaptive_press import _zero
from run_layered_mound_three_rock_cover_release import _stable_particle
from run_mixed_material_r3_a_probe import _make_transform, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_47_excavation_preflight_config_v1"
REPORT_SCHEMA = "excavation_layered_47_excavation_preflight_report_v1"
LAYERS = ("lower", "lower_middle", "middle", "upper")
DEFAULT_PROFILE = {
    "layers": LAYERS,
    "expected_ids": (
        [f"lower-{index:03d}" for index in range(9)]
        + [f"lower_middle-{index:03d}" for index in range(11)]
        + [f"middle-{index:03d}" for index in range(13)]
        + [f"upper-{index:03d}" for index in range(14)]),
    "temporary_directory_prefix": "layered-47-excavation-preflight-",
    "restored_runtime_label": "source-restored-47",
    "progress_label": "PROGRESS_47_PREFLIGHT",
    "report_schema": REPORT_SCHEMA,
    "passed_status": "layered_47_excavation_preflight_passed",
    "failed_status": "layered_47_excavation_preflight_failed",
    "passed_next_gate":
        "scripted_approach_to_program_selected_noncontact_prepoint",
    "failed_next_gate": "fix_precontact_scene_binding",
}
FORMAL_INPUT_CHECKPOINT_CONTRACT = {
    "default_directory_from_home": "a30-layered-33-to-47-v5/checkpoints",
    "filename_pattern": "four-layer-47-checkpoint-cuda-0-*.npz",
    "checkpoint_schema": "excavation_layered_mound_continuation_checkpoint_v2",
    "checkpoint_status": "four_layer_continuation_checkpoint_saved",
    "sha256": "40b98e83bcc1f7b91366612755eee57bc5df21db2a5c53fe2accf4408c987ea5",
    "source_config_sha256": "ac0d18752937230337f332b0fc08bfcfe639fc893f31f6fc2c840c1534dd3548",
    "source_runner_sha256": "bf9987f3cc5563f6cf5fdb61b17d062046ca94512f80e46d59a003809135c3c5",
    "particle_count": 256441,
    "rock_count": 47,
    "newton_version": "1.5.1",
    "warp_version": "1.17.0",
    "read_only": True,
    "metadata_required": True,
}

FORMAL_MPM_SOLVER_PROFILE = {
    "grid_type": "fixed",
    "grid_padding": 12,
    "max_active_cell_count": 1 << 18,
    "max_iterations": 50,
    "tolerance": 1.0e-4,
    "transfer_scheme": "pic",
    "integration_scheme": "pic",
    "strain_basis": "P0",
    "critical_fraction": 0.0,
    "collider_basis": "S2",
    "velocity_basis": "Q1",
    "collider_velocity_mode": "forward",
}

# This is part of the current collision baseline, not an optional visual or
# operator safety switch.  The post-solve correction closes endpoint misses
# that can occur when a particle travels beyond the normal query band in one
# outer MPM step.  It is intentionally recorded separately from the solver's
# collider impulses: Newton's public projection API does not expose the hit
# collider id for the velocity correction, so routing it into MuJoCo would
# fabricate ownership and risk double counting a bucket/rock reaction.
POSTSOLVE_PROJECTION_POLICY = {
    "enabled": True,
    "method": "SolverImplicitMPM.project_outside",
    "base_query_gap": "voxel_size_m * sqrt(3)",
    "speed_headroom": 1.25,
    "reaction_routing": "unattributed_diagnostic_not_applied_to_mujoco",
    "continuous_collision_proven": False,
}


def mpm_solver_profile(config: dict, override: dict | None = None) -> dict:
    """Return a complete, validated solver profile.

    The formal preflight path passes no override and therefore retains the
    original fixed-grid settings exactly.  Non-formal benchmark runners may
    supply a complete override, but cannot silently change material, particle
    spacing, transfer/integration bases, or collider velocity semantics.
    """
    profile = {
        "voxel_size_m": float(config["mpm"]["voxel_size_m"]),
        **FORMAL_MPM_SOLVER_PROFILE,
    }
    if override is None:
        return profile
    if set(override) != set(profile):
        raise ValueError("excavation_solver_profile_fields")
    candidate = dict(override)
    if candidate["grid_type"] not in {"fixed", "sparse"}:
        raise ValueError("excavation_solver_profile_grid_type")
    for key in (
            "transfer_scheme", "integration_scheme", "strain_basis",
            "critical_fraction", "collider_basis", "velocity_basis",
            "collider_velocity_mode"):
        if candidate[key] != profile[key]:
            raise ValueError(f"excavation_solver_profile_frozen_{key}")
    if (float(candidate["voxel_size_m"]) <= 0.0
            or int(candidate["grid_padding"]) < 0
            or int(candidate["max_active_cell_count"]) <= 0
            or int(candidate["max_iterations"]) <= 0
            or float(candidate["tolerance"]) <= 0.0):
        raise ValueError("excavation_solver_profile_numeric")
    return candidate


def postsolve_projection_diagnostics(
        before_q: np.ndarray, before_qd: np.ndarray,
        after_q: np.ndarray, after_qd: np.ndarray, *, particle_mass_kg: float,
        base_query_gap_m: float, dt_s: float, applied: bool) -> dict:
    """Return a JSON-safe audit record for the endpoint correction.

    `momentum_delta` is an aggregate particle diagnostic only.  It must not be
    fed to the rigid bridge until Newton exposes per-collider attribution for
    the correction.
    """
    before_q = np.asarray(before_q, dtype=np.float64)
    before_qd = np.asarray(before_qd, dtype=np.float64)
    after_q = np.asarray(after_q, dtype=np.float64)
    after_qd = np.asarray(after_qd, dtype=np.float64)
    same_shape = (before_q.shape == after_q.shape == before_qd.shape
                  == after_qd.shape and before_q.ndim == 2
                  and before_q.shape[1] == 3)
    if not same_shape:
        raise ValueError("postsolve_projection_particle_shape")
    before_finite = bool(
        np.isfinite(before_q).all() and np.isfinite(before_qd).all())
    after_finite = bool(
        np.isfinite(after_q).all() and np.isfinite(after_qd).all())
    pre_projection_max_speed_m_s = (
        float(np.linalg.norm(before_qd, axis=1).max()) if before_finite
        and len(before_qd) else float("nan"))
    query_gap_m = max(
        float(base_query_gap_m),
        (POSTSOLVE_PROJECTION_POLICY["speed_headroom"]
         * pre_projection_max_speed_m_s * float(dt_s))
        if np.isfinite(pre_projection_max_speed_m_s) else float(base_query_gap_m))
    if before_finite and after_finite:
        position_delta = after_q - before_q
        velocity_delta = after_qd - before_qd
        position_norm = np.linalg.norm(position_delta, axis=1)
        velocity_norm = np.linalg.norm(velocity_delta, axis=1)
        changed = (position_norm > 1.0e-12) | (velocity_norm > 1.0e-12)
        momentum_delta = float(particle_mass_kg) * np.sum(
            velocity_delta, axis=0, dtype=np.float64)
        kinetic_energy_delta = 0.5 * float(particle_mass_kg) * (
            np.sum(after_qd * after_qd, dtype=np.float64)
            - np.sum(before_qd * before_qd, dtype=np.float64))
        max_position_delta_m = float(position_norm.max()) if len(position_norm) else 0.0
        max_velocity_delta_m_s = float(velocity_norm.max()) if len(velocity_norm) else 0.0
        corrected_count = int(np.count_nonzero(changed))
    else:
        momentum_delta = np.full(3, np.nan, dtype=np.float64)
        kinetic_energy_delta = float("nan")
        max_position_delta_m = float("nan")
        max_velocity_delta_m_s = float("nan")
        corrected_count = 0
    return {
        **POSTSOLVE_PROJECTION_POLICY,
        "applied": bool(applied),
        "input_finite": before_finite,
        "output_finite": after_finite,
        "base_query_gap_m": float(base_query_gap_m),
        "query_gap_m": float(query_gap_m),
        "pre_projection_max_speed_m_s": pre_projection_max_speed_m_s,
        "corrected_particle_count": corrected_count,
        "maximum_position_delta_m": max_position_delta_m,
        "maximum_velocity_delta_m_s": max_velocity_delta_m_s,
        "particle_momentum_delta_kg_m_s": momentum_delta.tolist(),
        "particle_momentum_delta_norm_kg_m_s": float(
            np.linalg.norm(momentum_delta)),
        "particle_kinetic_energy_delta_j": float(kinetic_energy_delta),
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_bound_json(root: Path, binding: dict, label: str) -> tuple[Path, dict]:
    path = _resolve(root, binding.get("path", ""))
    if not path.is_file() or sha256(path) != binding.get("sha256"):
        raise ValueError(f"excavation_preflight_{label}_binding")
    return path, json.loads(path.read_text(encoding="utf-8"))


def load_config(
        path: str | Path,
        repository_root: str | Path,
        *,
        expected_input_checkpoint: dict | None = None) -> dict:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "bind_audited_47_to_stationary_open_bucket_scene"):
        raise ValueError("excavation_preflight_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("runtime_dependencies", ("checkpoint", "runtime", "predictor"))):
        binding = config.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"excavation_preflight_{group}_{field}_binding")
    _, operator = _load_bound_json(
        root, config.get("operator_start", {}), "operator_start")
    _, proxy = _load_bound_json(
        root, config.get("open_bucket_proxy", {}), "open_bucket_proxy")
    _, receiver = _load_bound_json(
        root, config.get("truck_receiver_observer", {}), "truck_receiver")
    source = config.get("input_checkpoint", {})
    required_source = (FORMAL_INPUT_CHECKPOINT_CONTRACT
                       if expected_input_checkpoint is None
                       else expected_input_checkpoint)
    if source != required_source:
        raise ValueError("excavation_preflight_input_checkpoint_contract")
    qpos = operator.get("qpos")
    if (not isinstance(qpos, list) or len(qpos) != 19
            or not np.isfinite(np.asarray(qpos, dtype=np.float64)).all()
            or operator.get("physics_steps") != 0):
        raise ValueError("excavation_preflight_operator_start_contract")
    if (proxy.get("schema") != "excavation_open_bucket_collision_proxy_v1"
            or proxy.get("geom_count") != 18
            or proxy.get("has_mouth_cap") is not False
            or receiver.get("schema") != "truck_receiver_triangles_v1"):
        raise ValueError("excavation_preflight_scene_geometry_contract")
    expected = {
        "hold_frames": 120,
        "minimum_initial_vertical_clearance_m": 0.02,
        "maximum_bucket_origin_drift_m": 0.002,
        "maximum_bucket_rotation_drift_rad": 0.002,
        "writes_checkpoint": False,
        "requires_zero_bucket_contact_during_hold": True,
    }
    if config.get("preflight") != expected:
        raise ValueError("excavation_preflight_gate_contract")
    expected_scope = {
        "source_checkpoint_mutated": False,
        "new_particles_added": False,
        "new_rocks_added": False,
        "new_checkpoint_written": False,
        "bucket_collision_active": True,
        "bucket_motion_commanded": False,
        "truck_visual_and_receiver_observer_reused": True,
        "truck_physics_collider_qualified": False,
        "digging_executed": False,
        "training": False,
    }
    if config.get("scope") != expected_scope:
        raise ValueError("excavation_preflight_scope_contract")
    return config


def _quat_angle(first: np.ndarray, second: np.ndarray) -> float:
    a = first / np.linalg.norm(first)
    b = second / np.linalg.norm(second)
    return 2.0 * math.acos(min(1.0, max(-1.0, abs(float(np.dot(a, b))))))


def _quat_xyzw_from_matrix(rotation: np.ndarray) -> list[float]:
    """Convert one proper rotation matrix to Newton's xyzw convention."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("excavation_preflight_bucket_rotation")
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (r[2, 1] - r[1, 2]) / scale
        y = (r[0, 2] - r[2, 0]) / scale
        z = (r[1, 0] - r[0, 1]) / scale
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / scale
        x = 0.25 * scale
        y = (r[0, 1] + r[1, 0]) / scale
        z = (r[0, 2] + r[2, 0]) / scale
    elif r[1, 1] > r[2, 2]:
        scale = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / scale
        x = (r[0, 1] + r[1, 0]) / scale
        y = 0.25 * scale
        z = (r[1, 2] + r[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / scale
        x = (r[0, 2] + r[2, 0]) / scale
        y = (r[1, 2] + r[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _bucket_transform(state: dict):
    rotation = np.asarray(state["bucket_rotation"], dtype=np.float64).reshape(3, 3)
    quaternion = _quat_xyzw_from_matrix(rotation)
    return wp.transform(
        wp.vec3(*[float(value) for value in state["bucket_origin"]]),
        wp.quat(*[float(value) for value in quaternion]),
    )


def _bucket_world_vertices(proxy: dict, state: dict) -> np.ndarray:
    values = []
    for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
        half = np.asarray(box["half_size_m"], dtype=np.float64)
        corners = np.asarray([
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)], dtype=np.float64)
        w, x, y, z = map(float, box["quaternion_wxyz"])
        rotation = np.asarray([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float64)
        values.append(corners @ rotation.T + np.asarray(
            box["position_bucket_m"], dtype=np.float64))
    values.extend(np.asarray(side["vertices_bucket_m"], dtype=np.float64)
                  for side in proxy["side_plate_meshes"])
    local = np.concatenate(values, axis=0)
    rotation = np.asarray(state["bucket_rotation"], dtype=np.float64).reshape(3, 3)
    return local @ rotation.T + np.asarray(
        state["bucket_origin"], dtype=np.float64)


class ExcavationFixture:
    """The checkpoint fixture expanded by one kinematic open-bucket body."""

    def __init__(self, config: dict, repository: Path, records: list[dict],
                 particle_q: np.ndarray, proxy: dict, initial_state: dict,
                 device: str, solver_profile: dict | None = None,
                 static_fixtures: list[dict] | None = None):
        self.records = records
        self.ids = [f"r1-rock-{index:03d}" for index in range(len(records))]
        self.device = device
        self.grid_reference_particle_q = np.asarray(particle_q).copy()
        material = config["mpm"]["material"]
        spacing = float(config["mpm"]["particle_spacing_m"])
        self.particle_spacing_m = spacing
        radius = 0.45 * spacing
        collider = newton.ModelBuilder()
        self.body_indices = {}
        for index, record in enumerate(records):
            vertices, indices = _read_obj(
                repository / "assets/v4/mixed-material-r0" /
                f"{record['prototype_id']}.obj")
            inertia = np.asarray(
                record["inertia_about_com_body_kg_m2"], dtype=np.float32)
            body = int(collider.add_body(
                xform=_make_transform(
                    record["position_world_m"],
                    record["orientation_quaternion_wxyz"]),
                com=wp.vec3(0.0), inertia=wp.mat33(inertia),
                mass=float(record["mass_kg"]), label=self.ids[index],
                lock_inertia=True))
            self.body_indices[self.ids[index]] = body
            collider.add_shape_mesh(
                body, mesh=newton.Mesh(
                    vertices, indices, compute_inertia=False, is_solid=True),
                scale=wp.vec3(float(record["equivalent_diameter_m"])),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(material["friction"]),
                    has_particle_collision=True),
                label=self.ids[index] + "-mpm-collider")
        self.bucket_body_index = int(collider.add_body(
            xform=_bucket_transform(initial_state), mass=100.0,
            inertia=wp.mat33(np.eye(3, dtype=np.float32)), label="bucket",
            lock_inertia=True, is_kinematic=True))
        self.bucket_proxy_geom_count = int(proxy["geom_count"])
        bucket_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(material["friction"]),
            has_particle_collision=True)
        for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
            quaternion = box["quaternion_wxyz"]
            collider.add_shape_box(
                self.bucket_body_index,
                xform=wp.transform(
                    wp.vec3(*[float(value) for value in box["position_bucket_m"]]),
                    wp.quat(float(quaternion[1]), float(quaternion[2]),
                            float(quaternion[3]), float(quaternion[0]))),
                hx=float(box["half_size_m"][0]),
                hy=float(box["half_size_m"][1]),
                hz=float(box["half_size_m"][2]), cfg=bucket_cfg,
                label="excavation_" + box["name"])
        self.bucket_side_winding_audit = []
        for side in proxy["side_plate_meshes"]:
            faces, winding_audit = outward_side_plate_faces(side)
            self.bucket_side_winding_audit.append(winding_audit)
            collider.add_shape_mesh(
                self.bucket_body_index,
                mesh=newton.Mesh(
                    np.asarray(side["vertices_bucket_m"], dtype=np.float32),
                    faces.reshape(-1),
                    compute_inertia=False, is_solid=True),
                scale=wp.vec3(1.0), cfg=bucket_cfg,
                label="excavation_" + side["name"])
        self.truck_shape_indices = {}
        for fixture in static_fixtures or []:
            if (fixture.get("contact_role") != "truck_receiver"
                    or not str(fixture.get("name", "")).startswith("r1_truck_")):
                raise ValueError("excavation_static_fixture_contract")
            quaternion = fixture["quaternion_wxyz"]
            shape = collider.add_shape_box(
                -1,
                xform=wp.transform(
                    wp.vec3(*[float(value) for value in
                              fixture["position_world_m"]]),
                    wp.quat(float(quaternion[1]), float(quaternion[2]),
                            float(quaternion[3]), float(quaternion[0]))),
                hx=float(fixture["half_size_m"][0]),
                hy=float(fixture["half_size_m"][1]),
                hz=float(fixture["half_size_m"][2]),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=float(fixture["newton_friction"]),
                    has_particle_collision=True),
                label=fixture["name"])
            self.truck_shape_indices[fixture["name"]] = int(shape)
        self.ground_shape_index = int(collider.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(material["friction"]))))
        self.collider_model = collider.finalize(device=device)

        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        self.particle_mass = (
            float(config["mpm"]["particle_density_kg_m3"]) * spacing ** 3)
        sand.add_particles(
            pos=particle_q.tolist(), vel=np.zeros_like(particle_q).tolist(),
            mass=[self.particle_mass] * len(particle_q),
            radius=[radius] * len(particle_q),
            custom_attributes={
                "mpm:friction": [float(material["friction"])] * len(particle_q)})
        self.model = sand.finalize(device=device)
        self.model.set_gravity(tuple(config["mpm"]["gravity_m_s2"]))
        for name, value in material.items():
            if hasattr(self.model.mpm, name):
                getattr(self.model.mpm, name).fill_(float(value))
        selected = mpm_solver_profile(config, solver_profile)
        self.solver_profile = dict(selected)
        options = SolverImplicitMPM.Config()
        options.voxel_size = float(selected["voxel_size_m"])
        options.grid_type = selected["grid_type"]
        options.grid_padding = int(selected["grid_padding"])
        options.max_active_cell_count = int(selected["max_active_cell_count"])
        options.max_iterations = int(selected["max_iterations"])
        options.tolerance = float(selected["tolerance"])
        options.transfer_scheme = selected["transfer_scheme"]
        options.integration_scheme = selected["integration_scheme"]
        options.strain_basis = selected["strain_basis"]
        options.critical_fraction = float(selected["critical_fraction"])
        options.collider_basis = selected["collider_basis"]
        options.velocity_basis = selected["velocity_basis"]
        options.collider_velocity_mode = selected["collider_velocity_mode"]
        self.solver = SolverImplicitMPM(self.model, config=options)
        self.projection_base_query_gap_m = (
            float(options.voxel_size) * math.sqrt(3.0))
        self.solver.setup_collider(model=self.collider_model)
        self.state = self.model.state()
        collider_state = self.collider_model.state()
        self.state.body_q = wp.clone(collider_state.body_q)
        self.state.body_qd = wp.clone(collider_state.body_qd)
        self.state.body_f = wp.zeros_like(collider_state.body_f)
        self.assign(initial_state)
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy().copy()
        self.static_collider_ids = np.flatnonzero(body_map == -1)
        self.truck_collider_ids = {
            name: np.asarray([index], dtype=np.int64)
            for name, index in self.truck_shape_indices.items()}
        self.ground_collider_ids = np.setdiff1d(
            self.static_collider_ids,
            np.asarray(list(self.truck_shape_indices.values()), dtype=np.int64),
            assume_unique=True)
        self.collider_ids = {
            rock_id: np.flatnonzero(body_map == body)
            for rock_id, body in self.body_indices.items()}
        self.bucket_collider_ids = np.flatnonzero(
            body_map == self.bucket_body_index)
        sets = [set(value.tolist()) for value in self.collider_ids.values()]
        sets += [set(self.bucket_collider_ids.tolist()),
                 set(self.ground_collider_ids.tolist())]
        sets += [set(value.tolist()) for value in
                 self.truck_collider_ids.values()]
        self.mapping_disjoint = all(
            not sets[first].intersection(sets[second])
            for first in range(len(sets)) for second in range(first + 1, len(sets)))
        if (any(value.size != 1 for value in self.collider_ids.values())
                or self.bucket_collider_ids.size != 1
                or self.ground_collider_ids.size != 1
                or any(value.size != 1 for value in
                       self.truck_collider_ids.values())
                or not self.mapping_disjoint):
            raise RuntimeError("excavation_preflight_collider_mapping")

    def assign(self, bridge_state: dict) -> None:
        transforms = [
            _make_transform(rock["qpos"][:3], rock["qpos"][3:7])
            for rock in bridge_state["rocks"]]
        transforms.append(_bucket_transform(bridge_state))
        velocities = [wp.spatial_vector(
            *[float(value) for value in rock["linear_velocity_world_m_s"]],
            *[float(value) for value in rock["angular_velocity_world_rad_s"]])
            for rock in bridge_state["rocks"]]
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
    def _collect(impulse, position, ids, selected_ids, origin, dt):
        mask = np.isin(ids, selected_ids)
        mask &= np.isfinite(impulse).all(axis=1)
        mask &= np.linalg.norm(impulse, axis=1) > 1.0e-12
        selected = impulse[mask]
        selected_position = position[mask]
        if not selected.size:
            return _zero(origin), 0
        force = np.sum(selected, axis=0, dtype=np.float64) / dt
        torque = np.sum(np.cross(
            selected_position - origin, selected), axis=0,
            dtype=np.float64) / dt
        return {"force": force, "torque": torque,
                "point": np.asarray(origin, dtype=np.float64)}, int(mask.sum())

    def step(self, dt: float, origins: dict[str, np.ndarray]) -> tuple[dict, dict]:
        self.solver.step(self.state, self.state, None, None, dt)
        wp.synchronize()
        before_projection_q = self.state.particle_q.numpy()
        before_projection_qd = self.state.particle_qd.numpy()
        projection_input_finite = bool(
            np.isfinite(before_projection_q).all()
            and np.isfinite(before_projection_qd).all())
        if projection_input_finite:
            pre_projection_max_speed_m_s = float(np.linalg.norm(
                before_projection_qd, axis=1).max())
            query_gap_m = max(
                self.projection_base_query_gap_m,
                (POSTSOLVE_PROJECTION_POLICY["speed_headroom"]
                 * pre_projection_max_speed_m_s * dt))
            self.solver.project_outside(
                self.state, self.state, dt, gap=query_gap_m)
            wp.synchronize()
            projection_applied = True
        else:
            # An invalid solver output is a hard failure recorded by the caller;
            # passing it through a geometric correction would make the fault less
            # interpretable.  This is not an operator/configuration toggle.
            projection_applied = False
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        postsolve_projection = postsolve_projection_diagnostics(
            before_projection_q, before_projection_qd, q, qd,
            particle_mass_kg=self.particle_mass,
            base_query_gap_m=self.projection_base_query_gap_m,
            dt_s=dt, applied=projection_applied)
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(
            self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        finite_samples = (np.isfinite(impulse).all(axis=1)
                          & np.isfinite(position).all(axis=1))
        active = finite_samples & (np.linalg.norm(impulse, axis=1) > 1.0e-12)
        wrenches, counts = {}, {}
        for body_name, selected_ids in {
                **self.collider_ids, "bucket": self.bucket_collider_ids}.items():
            wrenches[body_name], counts[body_name] = self._collect(
                impulse, position, ids, selected_ids,
                origins[body_name], dt)
        owned = self.static_collider_ids.copy()
        for selected_ids in self.collider_ids.values():
            owned = np.concatenate((owned, selected_ids))
        owned = np.concatenate((owned, self.bucket_collider_ids))
        truck_counts = {
            name: int(np.count_nonzero(active & np.isin(ids, selected_ids)))
            for name, selected_ids in self.truck_collider_ids.items()}
        speed = np.linalg.norm(qd, axis=1)
        return wrenches, {
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
            "particle_max_speed_m_s": float(speed.max()),
            "particle_p99_speed_m_s": float(np.quantile(speed, 0.99)),
            "particle_minimum_z_m": float(q[:, 2].min()),
            "contact_samples_by_body": counts,
            "contact_samples_by_truck_fixture": truck_counts,
            "truck_mpm_contact_samples": int(sum(truck_counts.values())),
            "total_contact_samples": int(sum(counts.values())),
            "nonfinite_collider_sample_count": int(np.count_nonzero(~finite_samples)),
            "unmapped_active_collider_sample_count": int(np.count_nonzero(
                active & ~np.isin(ids, owned))),
            "postsolve_projection": postsolve_projection,
        }


def _rock_world_max_z(records: list[dict], qpos: np.ndarray,
                      repository: Path) -> float:
    maximum = -math.inf
    for record, pose in zip(records, qpos, strict=True):
        vertices, _ = _read_obj(
            repository / "assets/v4/mixed-material-r0" /
            f"{record['prototype_id']}.obj")
        w, x, y, z = map(float, pose[3:7])
        rotation = np.asarray([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float64)
        world = (np.asarray(vertices, dtype=np.float64)
                 * float(record["equivalent_diameter_m"])) @ rotation.T + pose[:3]
        maximum = max(maximum, float(world[:, 2].max()))
    return maximum


def run(args, *, config_loader=load_config, profile: dict | None = None) -> dict:
    profile = DEFAULT_PROFILE if profile is None else profile
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = Path(args.checkpoint_metadata).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = config_loader(config_path, root)
    arrays, source_metadata, before = load_checkpoint_readonly(
        checkpoint, metadata_path, config["input_checkpoint"])
    manifest_path = _resolve(
        root, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in profile["layers"]]
    expected_ids = profile["expected_ids"]
    if (len(records) != len(expected_ids)
            or [rock["rock_id"] for rock in records] != expected_ids):
        raise ValueError("excavation_preflight_manifest_order")
    _, operator = _load_bound_json(
        root, config["operator_start"], "operator_start")
    proxy_path, proxy = _load_bound_json(
        root, config["open_bucket_proxy"], "open_bucket_proxy")
    receiver_path, _ = _load_bound_json(
        root, config["truck_receiver_observer"], "truck_receiver")
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    dt = float(config["probe"]["output_dt_s"])
    target_bridge = None
    source_bridge = None
    with tempfile.TemporaryDirectory(
            prefix=profile["temporary_directory_prefix"]) as raw:
        work = Path(raw)
        restored_runtime_label = profile["restored_runtime_label"]
        source_fixture, source_bridge, restored = restore_runtime(
            args, config, root, records, arrays, work,
            restored_runtime_label)
        source_model = Path(restored["model_manifest"]["model_path"]) if (
            "model_path" in restored["model_manifest"]) else (
                work / f"{restored_runtime_label}.xml")
        binary = Path(restored["binary"])
        source_bridge.close()
        source_bridge = None
        target_bridge = RestoredRuntimeBridge(
            binary, source_model, len(records),
            [float(value) for value in operator["qpos"]], [0.0] * 18,
            arrays["rigid_qpos"], arrays["rigid_qvel"])
        fixture = ExcavationFixture(
            config, root, records, arrays["grid_reference_particle_q"],
            proxy, target_bridge.state, args.device)
        particle_restore = restore_fixture_prefix(fixture, arrays)
        fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        warmstart_transfer = transfer_solver_warmstart(source_fixture, fixture)
        fixture.assign(target_bridge.state)

        particle_radius = 0.45 * float(config["mpm"]["particle_spacing_m"])
        bucket_vertices = _bucket_world_vertices(proxy, target_bridge.state)
        particle_top = float(arrays["particle_q"][:, 2].max()) + particle_radius
        rock_top = _rock_world_max_z(records, arrays["rigid_qpos"], root)
        vertical_clearance = float(bucket_vertices[:, 2].min()) - max(
            particle_top, rock_top)
        initial_bucket_origin = np.asarray(
            target_bridge.state["bucket_origin"], dtype=np.float64)
        initial_bucket_quat = _quat_xyzw_from_matrix(np.asarray(
            target_bridge.state["bucket_rotation"], dtype=np.float64).reshape(3, 3))
        rows = []
        gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
        hold_frames = int(config["preflight"]["hold_frames"])
        for frame in range(hold_frames):
            prior = target_bridge.state
            fixture.assign(prior)
            rock_origins = {
                rock["rock_id"]: np.asarray(rock["position"], dtype=np.float64)
                for rock in prior["rocks"]}
            origins = {**rock_origins, "bucket": np.asarray(
                prior["bucket_origin"], dtype=np.float64)}
            # Newton's forward collider velocity uses assigned body_qd.  Rigid
            # gravity is predicted exactly once before the MPM impulse sample.
            transforms = [
                _make_transform(rock["qpos"][:3], rock["qpos"][3:7])
                for rock in prior["rocks"]] + [_bucket_transform(prior)]
            velocities = [wp.spatial_vector(
                *[float(value) for value in predict_free_linear_velocity(
                    np.asarray(rock["qvel"][:3], dtype=np.float64), gravity, dt)],
                *[float(value) for value in rock["angular_velocity_world_rad_s"]])
                for rock in prior["rocks"]]
            velocities.append(wp.spatial_vector(
                *[float(value) for value in prior[
                    "bucket_linear_velocity_world_m_s"]],
                *[float(value) for value in prior[
                    "bucket_angular_velocity_world_rad_s"]]))
            fixture.state.body_q.assign(wp.array(
                transforms, dtype=wp.transform, device=args.device))
            fixture.state.body_qd.assign(wp.array(
                velocities, dtype=wp.spatial_vector, device=args.device))
            measured, particles = fixture.step(dt, origins)
            after = target_bridge.step(
                dt, int(config["probe"]["mujoco_internal_substeps"]), measured)
            linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                         for rock in after["rocks"])
            angular = max(float(np.linalg.norm(rock["qvel"][3:]))
                          for rock in after["rocks"])
            step_delta = max(float(np.linalg.norm(
                np.asarray(current["qpos"][:3]) - np.asarray(previous["qpos"][:3])))
                for previous, current in zip(
                    prior["rocks"], after["rocks"], strict=True))
            bucket_origin = np.asarray(after["bucket_origin"], dtype=np.float64)
            bucket_quat = _quat_xyzw_from_matrix(np.asarray(
                after["bucket_rotation"], dtype=np.float64).reshape(3, 3))
            contacts = after.get("contact_substeps", {})
            stable = bool(
                linear <= float(config["probe"][
                    "maximum_stable_rock_linear_speed_m_s"])
                and angular <= float(config["probe"][
                    "maximum_stable_rock_angular_speed_rad_s"])
                and _stable_particle(particles, config["probe"]))
            row = {
                "frame": frame,
                "finite": bool(particles["finite"] and _finite_tree(after)),
                "stable": stable,
                "maximum_rock_linear_speed_m_s": linear,
                "maximum_rock_angular_speed_rad_s": angular,
                "maximum_rock_position_step_m": step_delta,
                "bucket_origin_drift_m": float(np.linalg.norm(
                    bucket_origin - initial_bucket_origin)),
                "bucket_rotation_drift_rad": _quat_angle(
                    bucket_quat, initial_bucket_quat),
                "bucket_mpm_contact_samples": int(
                    particles["contact_samples_by_body"]["bucket"]),
                "rock_bucket_contact_substeps": int(
                    contacts.get("rock_bucket", 0)),
                "external_rigid_support_substeps": sum(int(
                    contacts.get(name, 0)) for name in (
                        "rock_ground", "rock_ramp", "rock_truck")),
                "forbidden_contact_count": int(
                    after.get("forbidden_contact_count", 0)),
                "closure_m": float(after.get("closure_m", math.inf)),
                **particles,
            }
            rows.append(row)
            if frame == 0 or (frame + 1) % 30 == 0:
                print(
                    f"{profile['progress_label']} "
                    f"frame={frame + 1}/{hold_frames} stable={stable} "
                    f"bucket_mpm={row['bucket_mpm_contact_samples']} "
                    f"rock_bucket={row['rock_bucket_contact_substeps']} "
                    f"rock_linear={linear:.9g} particle_max="
                    f"{particles['particle_max_speed_m_s']:.9g}",
                    flush=True)
            if not row["finite"]:
                break
        target_bridge.close(kill=not rows or not rows[-1]["finite"])
        target_bridge = None
    after_identity = checkpoint_identity(checkpoint)
    executed = len(rows)
    first_position_limit = (
        float(config["probe"]["maximum_stable_rock_linear_speed_m_s"])
        * dt * 1.01)
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] ==
            config["input_checkpoint"]["sha256"],
        "source_checkpoint_metadata_verified": source_metadata[
            "checkpoint_sha256"] == before["sha256"],
        "source_checkpoint_unchanged": after_identity == before,
        "source_restore_exact": (
            all(restored["restore"]["field_readback_exact"].values())
            and restored["restore"]["rigid_qpos_readback_within_tolerance"]
            and restored["restore"]["rigid_qvel_readback_within_tolerance"]
            and all(restored.get("solver_restore", restored["restore"].get(
                "solver_continuation"))["grid_readback_exact"].values())
            and all(restored.get("solver_restore", restored["restore"].get(
                "solver_continuation"))[
                "solver_field_readback_exact"].values())),
        "expanded_particle_fields_restored_exactly": all(
            particle_restore["field_readback_exact"].values()),
        "grid_warmstart_transferred_for_bucket_expansion": (
            warmstart_transfer["body_q_prev_transferred"] is False
            and warmstart_transfer["collider_velocity_mode"] == "forward"),
        "open_bucket_proxy_active_and_uncapped": (
            fixture.bucket_collider_ids.size == 1
            and fixture.bucket_proxy_geom_count == proxy["geom_count"]
            and proxy["has_mouth_cap"] is False),
        "collider_ownership_disjoint": fixture.mapping_disjoint,
        "initial_vertical_clearance_passed": vertical_clearance >= float(
            config["preflight"]["minimum_initial_vertical_clearance_m"]),
        "all_requested_hold_frames_executed": executed == hold_frames,
        "all_states_finite": bool(rows) and all(row["finite"] for row in rows),
        "first_step_continuous_and_stable": bool(rows) and rows[0]["stable"]
            and rows[0]["maximum_rock_position_step_m"] <= first_position_limit,
        "short_hold_all_frames_stable": bool(rows) and all(
            row["stable"] for row in rows),
        "zero_bucket_soil_contact_before_approach": bool(rows) and all(
            row["bucket_mpm_contact_samples"] == 0 for row in rows),
        "zero_bucket_rock_contact_before_approach": bool(rows) and all(
            row["rock_bucket_contact_substeps"] == 0 for row in rows),
        "no_external_rigid_support": bool(rows) and all(
            row["external_rigid_support_substeps"] == 0 for row in rows),
        "no_unmapped_or_nonfinite_collider_samples": bool(rows) and all(
            row["unmapped_active_collider_sample_count"] == 0
            and row["nonfinite_collider_sample_count"] == 0 for row in rows),
        "forbidden_contacts_zero": bool(rows) and all(
            row["forbidden_contact_count"] == 0 for row in rows),
        "mechanism_closure_within_one_mm": bool(rows) and all(
            row["closure_m"] <= 0.001 for row in rows),
        "bucket_stationary_within_tolerance": bool(rows) and max(
            row["bucket_origin_drift_m"] for row in rows) <= float(
                config["preflight"]["maximum_bucket_origin_drift_m"])
            and max(row["bucket_rotation_drift_rad"] for row in rows) <= float(
                config["preflight"]["maximum_bucket_rotation_drift_rad"]),
        "no_checkpoint_written": True,
    }
    report = {
        "schema": profile["report_schema"],
        "status": profile["passed_status"] if all(
            checks.values()) else profile["failed_status"],
        "checks": checks,
        "source_checkpoint": before,
        "source_checkpoint_after": after_identity,
        "source_metadata": source_metadata,
        "operator_start": {
            "source_pose_label": operator["source_pose_label"],
            "qpos": operator["qpos"],
            "physics_steps_before_preflight": operator["physics_steps"],
        },
        "initial_clearance": {
            "bucket_proxy_min_z_m": float(bucket_vertices[:, 2].min()),
            "particle_envelope_top_z_m": particle_top,
            "rock_envelope_top_z_m": rock_top,
            "vertical_clearance_m": vertical_clearance,
            "minimum_required_m": config["preflight"][
                "minimum_initial_vertical_clearance_m"],
            "method": "conservative_vertical_envelope_before_zero_command_hold",
        },
        "restore": {
            "source": restored,
            "expanded_particle_restore": particle_restore,
            "expanded_warmstart": warmstart_transfer,
        },
        "collider_ownership": {
            "rock_count": len(fixture.collider_ids),
            "bucket_body_index": fixture.bucket_body_index,
            "bucket_proxy_geom_count": fixture.bucket_proxy_geom_count,
            "bucket_side_winding_audit": fixture.bucket_side_winding_audit,
            "bucket_solver_collider_id_count": int(
                fixture.bucket_collider_ids.size),
            "static_collider_count": int(fixture.static_collider_ids.size),
            "mapping_disjoint": fixture.mapping_disjoint,
        },
        "hold": {
            "requested_frames": hold_frames,
            "executed_frames": executed,
            "trace": rows,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
            "open_bucket_proxy_sha256": sha256(proxy_path),
            "truck_receiver_observer_sha256": sha256(receiver_path),
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "newton_solver_step_calls": executed,
            "mujoco_step_calls": executed,
            "wall_seconds": time.perf_counter() - started,
        },
        "scope": config["scope"],
        "next_gate": (profile["passed_next_gate"] if all(checks.values())
                      else profile["failed_next_gate"]),
    }
    if not _finite_tree(report):
        raise RuntimeError("excavation_preflight_nonfinite_report")
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "source_checkpoint": report["source_checkpoint"],
        "initial_clearance": report["initial_clearance"],
        "hold": {key: value for key, value in report["hold"].items()
                 if key != "trace"},
        "execution": report["execution"],
        "next_gate": report["next_gate"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == (
        "layered_47_excavation_preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
