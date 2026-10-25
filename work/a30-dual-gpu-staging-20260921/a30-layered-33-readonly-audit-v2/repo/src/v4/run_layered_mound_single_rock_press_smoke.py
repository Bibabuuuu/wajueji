#!/usr/bin/env python3
"""Press one lower-layer rock into an unexcavated foundation soil stage.

This is a deliberately narrow construction-primitive smoke.  The admitted
65-rock geometry and its actual time-zero report are hash-bound, but only the
foundation soil and ``lower-000`` participate in this run.  The rock begins a
few millimetres above the measured particle surface and follows a prescribed,
low-speed vertical path; it is never dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from build_mixed_material_capacity_preflight import sha256
from preview_layered_mound_existing_viewer import _mound_mask
from run_mixed_material_r3_a_probe import _make_transform, _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_single_rock_press_smoke_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_single_rock_press_smoke_report_v1"


def _resolve(repository: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repository / path


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "one_lower_rock_controlled_press_short_smoke_only"):
        raise ValueError("single_press_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("config", "report", "manifest")),
            ("time_zero", ("config", "runner", "report")),
            ("unexcavated_soil", ("visual_config", "source_bank"))):
        binding = value.get(group, {})
        for field in fields:
            target = _resolve(repository, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"single_press_{group}_{field}_binding")

    geometry_report = json.loads(_resolve(
        repository, value["physical_geometry"]["report_path"]
    ).read_text(encoding="utf-8"))
    manifest = json.loads(_resolve(
        repository, value["physical_geometry"]["manifest_path"]
    ).read_text(encoding="utf-8"))
    time_zero = json.loads(_resolve(
        repository, value["time_zero"]["report_path"]
    ).read_text(encoding="utf-8"))
    if (geometry_report.get("status") != "nonintersecting_geometry_preflight_passed"
            or not all(geometry_report.get("checks", {}).values())
            or manifest.get("rock_count") != 65
            or manifest.get("status") !=
            "nonintersecting_geometry_admitted_for_time_zero"
            or time_zero.get("status") !=
            "actual_65_rock_and_mpm_time_zero_passed"
            or not all(time_zero.get("checks", {}).values())):
        raise ValueError("single_press_upstream_gate")

    target = value.get("target", {})
    records = [rock for rock in manifest["rocks"]
               if rock["rock_id"] == target.get("manifest_rock_id")]
    if (len(records) != 1 or records[0]["layer_id"] != "lower"
            or target.get("runtime_rock_id") != "r1-rock-000"
            or target.get("free_fall") is not False
            or target.get("orientation_locked") is not True
            or not 0.0 < float(target.get("start_surface_clearance_m", -1)) <= 0.01
            or not 0.0 < float(target.get("controlled_press_distance_m", -1)) <= 0.03):
        raise ValueError("single_press_target_contract")

    soil = value.get("unexcavated_soil", {})
    if (soil.get("activation_stage") != 0
            or float(soil.get("maximum_center_height_m", -1)) != 0.32
            or soil.get("expected_particle_count") != 85620
            or soil.get("precarved") is not False):
        raise ValueError("single_press_soil_contract")
    probe = value.get("probe", {})
    if probe != {
            "output_dt_s": 1.0 / 120.0,
            "foundation_presettle_frames": 24,
            "controlled_press_frames": 36,
            "maximum_particle_speed_m_s": 5.0,
            "maximum_particle_drop_below_initial_m": 0.05,
            "minimum_nonzero_contact_samples": 1,
            "position_tolerance_m": 1e-6}:
        raise ValueError("single_press_probe_contract")
    if value.get("scope") != {
            "full_65_rock_identity_bound_by_time_zero": True,
            "active_soil": "unexcavated_foundation_stage_only",
            "active_rock_count": 1, "other_64_rocks_active": False,
            "kinematic_controlled_press": True, "free_fall": False,
            "mj_step_calls": 0, "newton_solver_step_calls": 60,
            "settling_qualified": False, "stable_scene_claimed": False,
            "fullscale_qualified": False, "training": False}:
        raise ValueError("single_press_scope_contract")
    return value


def _rotation_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _foundation_particles(config: dict, repository: Path) -> np.ndarray:
    soil = config["unexcavated_soil"]
    visual_path = _resolve(repository, soil["visual_config_path"])
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    source_path = _resolve(repository, soil["source_bank_path"])
    with np.load(source_path, allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    selected = (_mound_mask(source_q, visual)
                & (source_q[:, 2] <= float(soil["maximum_center_height_m"]) + 1e-9))
    q = source_q[selected]
    if q.shape != (int(soil["expected_particle_count"]), 3):
        raise ValueError("single_press_foundation_particle_count")
    return q


class PressFixture:
    def __init__(self, config: dict, repository: Path, record: dict,
                 particle_q: np.ndarray, device: str):
        material = config["mpm"]["material"]
        vertices, indices = _read_obj(
            repository / "assets/v4/mixed-material-r0" /
            f"{record['prototype_id']}.obj")
        rotation = _rotation_matrix(record["orientation_quaternion_wxyz"])
        scaled_rotated = ((np.asarray(vertices, dtype=np.float64)
                           * float(record["equivalent_diameter_m"])) @ rotation.T)
        self.local_min_z = float(scaled_rotated[:, 2].min())
        spacing = float(config["mpm"]["particle_spacing_m"])
        particle_radius = 0.45 * spacing
        surface_center_z = float(particle_q[:, 2].max())
        surface_extent_z = surface_center_z + particle_radius
        clearance = float(config["target"]["start_surface_clearance_m"])
        self.start_position = np.asarray(record["position_world_m"], dtype=np.float64)
        self.start_position[2] = surface_extent_z + clearance - self.local_min_z
        self.orientation = list(record["orientation_quaternion_wxyz"])

        collider = newton.ModelBuilder()
        inertia = np.asarray(record["inertia_about_com_body_kg_m2"], dtype=np.float32)
        self.rock_body = int(collider.add_body(
            xform=_make_transform(self.start_position, self.orientation),
            com=wp.vec3(0.0), inertia=wp.mat33(inertia),
            mass=float(record["mass_kg"]), label="r1-rock-000",
            lock_inertia=True))
        mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True)
        collider.add_shape_mesh(
            self.rock_body, mesh=mesh,
            scale=wp.vec3(float(record["equivalent_diameter_m"])),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, mu=float(material["friction"]),
                has_particle_collision=True),
            label="r1-rock-000-mpm-collider")
        collider.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0, mu=float(material["friction"])))
        self.collider_model = collider.finalize(device=device)

        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        particle_mass = float(config["mpm"]["particle_density_kg_m3"]) * spacing ** 3
        sand.add_particles(
            pos=particle_q.tolist(), vel=np.zeros_like(particle_q).tolist(),
            mass=[particle_mass] * len(particle_q),
            radius=[particle_radius] * len(particle_q),
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
        self.assign_rock(self.start_position, np.zeros(3))
        self.solver.reset(self.state)
        body_map = self.solver.collider_body_index.numpy().copy()
        self.rock_collider_ids = np.flatnonzero(body_map == self.rock_body)
        if self.rock_collider_ids.size != 1:
            raise RuntimeError("single_press_collider_mapping")
        self.initial_particle_q = self.state.particle_q.numpy().copy()
        self.particle_mass = particle_mass

    def assign_rock(self, position: np.ndarray, linear_velocity: np.ndarray) -> None:
        self.state.body_q.assign(wp.array([
            _make_transform(position, self.orientation)
        ], dtype=wp.transform, device=self.device))
        self.state.body_qd.assign(wp.array([
            wp.spatial_vector(
                float(linear_velocity[0]), float(linear_velocity[1]),
                float(linear_velocity[2]), 0.0, 0.0, 0.0)
        ], dtype=wp.spatial_vector, device=self.device))

    def step(self, dt: float, rock_position: np.ndarray) -> dict:
        self.solver.step(self.state, self.state, contacts=None, control=None, dt=dt)
        wp.synchronize()
        impulses, positions, collider_ids = self.solver.collect_collider_impulses(self.state)
        impulse = impulses.numpy()
        position = positions.numpy()
        ids = collider_ids.numpy()
        mask = np.isin(ids, self.rock_collider_ids)
        mask &= np.isfinite(impulse).all(axis=1)
        mask &= np.linalg.norm(impulse, axis=1) > 1e-12
        selected = impulse[mask]
        selected_position = position[mask]
        if selected.size:
            force = np.sum(selected, axis=0, dtype=np.float64) / dt
            torque = np.sum(np.cross(
                selected_position - rock_position, selected), axis=0,
                dtype=np.float64) / dt
        else:
            force, torque = np.zeros(3), np.zeros(3)
        q = self.state.particle_q.numpy()
        qd = self.state.particle_qd.numpy()
        return {
            "finite": bool(np.isfinite(q).all() and np.isfinite(qd).all()
                           and np.isfinite(force).all() and np.isfinite(torque).all()),
            "contact_samples": int(np.count_nonzero(mask)),
            "force_world_n": force.tolist(),
            "torque_world_nm": torque.tolist(),
            "maximum_particle_speed_m_s": float(np.linalg.norm(qd, axis=1).max()),
            "minimum_particle_z_m": float(q[:, 2].min()),
            "maximum_particle_drop_from_initial_m": float(np.max(
                self.initial_particle_q[:, 2] - q[:, 2])),
        }


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("single_press_output_exists")
    config = load_config(config_path, repository)
    manifest_path = _resolve(repository, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(rock for rock in manifest["rocks"]
                  if rock["rock_id"] == config["target"]["manifest_rock_id"])
    particle_q = _foundation_particles(config, repository)
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    fixture = PressFixture(config, repository, record, particle_q, args.device)
    probe = config["probe"]
    dt = float(probe["output_dt_s"])
    presettle_frames = int(probe["foundation_presettle_frames"])
    press_frames = int(probe["controlled_press_frames"])
    press_distance = float(config["target"]["controlled_press_distance_m"])
    press_velocity = press_distance / (press_frames * dt)
    frames = []
    peak_speed = 0.0
    peak_drop = 0.0
    peak_force = 0.0
    contact_samples = 0
    presettle_contact_samples = 0
    press_contact_samples = 0
    peak_presettle_force = 0.0
    peak_press_force = 0.0
    first_contact_frame = None
    first_press_contact_frame = None
    all_finite = True
    current = fixture.start_position.copy()
    for frame in range(presettle_frames + press_frames):
        if frame < presettle_frames:
            phase = "foundation_presettle"
            velocity = np.zeros(3)
        else:
            phase = "controlled_press"
            press_index = frame - presettle_frames + 1
            current = fixture.start_position.copy()
            current[2] -= press_distance * press_index / press_frames
            velocity = np.asarray([0.0, 0.0, -press_velocity])
        fixture.assign_rock(current, velocity)
        metrics = fixture.step(dt, current)
        samples = int(metrics["contact_samples"])
        force_norm = float(np.linalg.norm(metrics["force_world_n"]))
        contact_samples += samples
        if phase == "foundation_presettle":
            presettle_contact_samples += samples
            peak_presettle_force = max(peak_presettle_force, force_norm)
        else:
            press_contact_samples += samples
            peak_press_force = max(peak_press_force, force_norm)
            if samples and first_press_contact_frame is None:
                first_press_contact_frame = frame
        peak_speed = max(peak_speed, float(metrics["maximum_particle_speed_m_s"]))
        peak_drop = max(peak_drop, float(metrics["maximum_particle_drop_from_initial_m"]))
        peak_force = max(peak_force, force_norm)
        all_finite &= bool(metrics["finite"])
        if samples and first_contact_frame is None:
            first_contact_frame = frame
        if (frame == presettle_frames - 1 or frame == presettle_frames
                or frame == presettle_frames + press_frames - 1
                or samples and (not frames or frames[-1].get("contact_samples", 0) == 0)):
            frames.append({
                "frame": frame, "phase": phase,
                "rock_position_world_m": current.tolist(), **metrics})

    body_position = fixture.state.body_q.numpy()[0, :3].astype(np.float64)
    commanded_end = fixture.start_position.copy()
    commanded_end[2] -= press_distance
    actual_displacement = float(fixture.start_position[2] - body_position[2])
    target_final = np.asarray(record["position_world_m"], dtype=np.float64)
    checks = {
        "upstream_65_rock_time_zero_bound_and_passed": True,
        "unexcavated_foundation_particle_count": len(particle_q) ==
            int(config["unexcavated_soil"]["expected_particle_count"]),
        "single_target_rock_identity": bool(
            record["rock_id"] == "lower-000"
            and fixture.rock_collider_ids.size == 1),
        "controlled_path_no_free_fall": (
            config["target"]["free_fall"] is False
            and abs(actual_displacement - press_distance) <=
            float(probe["position_tolerance_m"])),
        "all_particle_and_contact_values_finite": all_finite,
        "nonzero_rock_soil_contact_during_press": press_contact_samples >=
            int(probe["minimum_nonzero_contact_samples"]),
        "press_reaction_strengthened_over_clearance_state":
            peak_press_force > peak_presettle_force + 1.0,
        "particle_speed_bounded": peak_speed <=
            float(probe["maximum_particle_speed_m_s"]),
        "particle_drop_bounded": peak_drop <=
            float(probe["maximum_particle_drop_below_initial_m"]),
        "exact_newton_step_scope": presettle_frames + press_frames ==
            int(config["scope"]["newton_solver_step_calls"]),
        "smoke_stops_above_final_construction_pose": bool(
            commanded_end[2] > target_final[2]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("single_lower_rock_controlled_press_smoke_passed"
                   if all(checks.values()) else
                   "single_lower_rock_controlled_press_smoke_failed"),
        "checks": checks,
        "target": {
            "manifest_rock_id": record["rock_id"],
            "runtime_rock_id": config["target"]["runtime_rock_id"],
            "prototype_id": record["prototype_id"],
            "diameter_m": record["equivalent_diameter_m"],
            "mass_kg": record["mass_kg"],
            "start_position_world_m": fixture.start_position.tolist(),
            "commanded_end_position_world_m": commanded_end.tolist(),
            "actual_end_position_world_m": body_position.tolist(),
            "full_construction_target_position_world_m": target_final.tolist(),
            "controlled_displacement_m": actual_displacement,
            "remaining_to_full_construction_target_z_m":
                float(commanded_end[2] - target_final[2]),
            "prescribed_press_speed_m_s": press_velocity,
            "local_mesh_min_z_m": fixture.local_min_z,
        },
        "metrics": {
            "foundation_particle_count": len(particle_q),
            "foundation_mass_kg": len(particle_q) * fixture.particle_mass,
            "total_nonzero_contact_samples": contact_samples,
            "presettle_nonzero_contact_samples": presettle_contact_samples,
            "press_nonzero_contact_samples": press_contact_samples,
            "first_contact_frame": first_contact_frame,
            "first_press_contact_frame": first_press_contact_frame,
            "peak_reaction_force_n": peak_force,
            "peak_presettle_reaction_force_n": peak_presettle_force,
            "peak_press_reaction_force_n": peak_press_force,
            "peak_particle_speed_m_s": peak_speed,
            "peak_particle_drop_from_initial_m": peak_drop,
            "trace": frames,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "physical_manifest_sha256": sha256(manifest_path),
            "source_bank_sha256": sha256(_resolve(
                repository, config["unexcavated_soil"]["source_bank_path"])),
            "foundation_q_sha256": hashlib.sha256(particle_q.tobytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "foundation_presettle_frames": presettle_frames,
            "controlled_press_frames": press_frames,
            "newton_solver_step_calls": presettle_frames + press_frames,
            "mj_step_calls": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "scope": config["scope"],
        "stable_scene_claimed": False,
        "settling_qualified": False,
        "next_gate": "review_smoke_before_any_multirock_or_full_depth_press",
    }
    if not _finite_tree(report):
        raise RuntimeError("single_press_nonfinite_report")
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "target": report["target"], "metrics": {
            key: value for key, value in report["metrics"].items() if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
