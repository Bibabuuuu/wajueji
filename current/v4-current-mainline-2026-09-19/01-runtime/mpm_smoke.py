"""Short independent MPM wiring smoke for the four-axis runtime.

This is deliberately not a training loop or a task-success evaluation.  It
uses a tiny reversible four-axis command around the qualified fast-tier full
initial state to prove clocks, coupling cadence, terrain age, causal wrench
history, actor-observation construction and fault-preserving logs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from contracts import (ACTION_SPEED_LIMIT_SI, ACTION_SCHEMA, AXES,
                       OBSERVATION_SCHEMA, validate_observation)
from runtime_adapter import (POLICY_DT_S, RESET_SCHEMA, FourAxisRuntime,
                             ResetRequest, step_request)


TIP_LOCAL_M = np.array([0.214, 0.0, -1.023])
MOUTH_LOCAL_M = np.array([0.117, 0.0, -0.590])
CYLINDER_LIMITS_M = np.array(((0.06, 0.56), (0.28, 1.10), (0.37, 0.86)))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _portable_source_hashes(values):
    """Normalize frozen source maps across an absolute runtime-root move.

    Historical qualification reports were produced below a ``codex-work``
    directory and stored absolute path keys.  The official platform necessarily
    uses a different home/workspace prefix.  Keep the source identity strict by
    comparing the stable path below ``codex-work`` (or ``site-packages`` for the
    installed Newton source) together with the original digest.
    """
    normalized = {}
    for raw_path, digest in values.items():
        text = str(raw_path).replace("\\", "/")
        if "/codex-work/" in text:
            key = "codex-work/" + text.split("/codex-work/", 1)[1]
        elif "/site-packages/" in text:
            key = "site-packages/" + text.split("/site-packages/", 1)[1]
        else:
            raise ValueError("source_path_has_no_portable_anchor:" + text)
        if key in normalized:
            raise ValueError("duplicate_portable_source_path:" + key)
        normalized[key] = digest
    return normalized


def _quaternion_wxyz(rotation):
    """Convert a finite proper 3x3 rotation without adding a dependency."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("invalid_bucket_rotation")
    values = np.array([
        1.0 + r[0, 0] + r[1, 1] + r[2, 2],
        1.0 + r[0, 0] - r[1, 1] - r[2, 2],
        1.0 - r[0, 0] + r[1, 1] - r[2, 2],
        1.0 - r[0, 0] - r[1, 1] + r[2, 2],
    ])
    index = int(np.argmax(values))
    if values[index] <= 0.0:
        raise ValueError("invalid_bucket_rotation")
    if index == 0:
        w = math.sqrt(values[0]) / 2.0
        q = np.array([w, (r[2, 1] - r[1, 2]) / (4.0 * w),
                      (r[0, 2] - r[2, 0]) / (4.0 * w),
                      (r[1, 0] - r[0, 1]) / (4.0 * w)])
    else:
        xyz = np.zeros(3)
        xyz[index - 1] = math.sqrt(values[index]) / 2.0
        x, y, z = xyz
        denominator = 4.0 * xyz[index - 1]
        if index == 1:
            w = (r[2, 1] - r[1, 2]) / denominator
            y = (r[0, 1] + r[1, 0]) / denominator
            z = (r[0, 2] + r[2, 0]) / denominator
        elif index == 2:
            w = (r[0, 2] - r[2, 0]) / denominator
            x = (r[0, 1] + r[1, 0]) / denominator
            z = (r[1, 2] + r[2, 1]) / denominator
        else:
            w = (r[1, 0] - r[0, 1]) / denominator
            x = (r[0, 2] + r[2, 0]) / denominator
            y = (r[1, 2] + r[2, 1]) / denominator
        q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def _smoke_action(decision_step, decisions):
    """Tiny simultaneous motion, exact reversal, then a zero-speed hold."""
    third = decisions // 3
    if decision_step < third:
        return [0.05, -0.05, 0.05, -0.05]
    if decision_step < 2 * third:
        return [-0.05, 0.05, -0.05, 0.05]
    return [0.0, 0.0, 0.0, 0.0]


class LiveFastTierBackend:
    def __init__(self, reset_request, *, backend_root, output, geometry_report,
                 empty_report, fast_reference_report, fill_root=None,
                 fill_calibration_report=None, fill_held_out_report=None):
        self.root = Path(backend_root).resolve()
        self.output = Path(output).resolve()
        self.geometry_report_path = Path(geometry_report).resolve()
        self.empty_report_path = Path(empty_report).resolve()
        self.fast_reference_report_path = Path(fast_reference_report).resolve()
        truth_values = (fill_root, fill_calibration_report, fill_held_out_report)
        if any(value is not None for value in truth_values) and not all(
                value is not None for value in truth_values):
            raise ValueError("incomplete_online_truth_configuration")
        self.fill_root = Path(fill_root).resolve() if fill_root is not None else None
        self.fill_calibration_report = (
            Path(fill_calibration_report).resolve()
            if fill_calibration_report is not None else None)
        self.fill_held_out_report = (
            Path(fill_held_out_report).resolve()
            if fill_held_out_report is not None else None)
        self.output.mkdir(parents=True, exist_ok=False)
        self.mechanism_log = (self.output / "mechanism-steps.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        self.soil_event_log = (self.output / "soil-steps.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        self.mech = None
        self.soil = None
        self.collider_model = None
        self.closed = False
        self.mechanism_steps = 0
        self.logged_mechanism_steps = 0
        self.soil_steps = 0
        self.logged_soil_steps = 0
        self.capture_calls = 0
        self.maximum_scene_outliers = 0
        self.digging_wrench_observed = False
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.point = np.zeros(3)
        self._common_sources = None
        self.source_before = None
        self.source_after = None
        self.qualification = {}
        self.separation_detector = None
        self.online_fill = None
        self.separation_cache = None
        self.separation_cache_soil_step = None
        self.latest_fill = None
        self.latest_fill_soil_step = None
        self.separation_evaluations = 0
        self.fill_evaluations = 0
        self.truth_sources = {}
        self.truth_sources_before = None
        self.truth_sources_after = None

    def reset(self, reset_request):
        if reset_request.episode_id == "":
            raise ValueError("empty_episode_id")
        if str(self.root) not in sys.path:
            sys.path.append(str(self.root))
        from common import A, B, M, R, T, Sensor, read, setup, sources
        from mechanism import Mechanism
        from particle_sampling import build_particle_bank, particle_sha256
        from training_scope import numerical_profile
        from work_envelope import WorkEnvelope

        self._common_sources = sources
        self.source_before = sources()
        geometry = _read(self.geometry_report_path)
        empty = _read(self.empty_report_path)
        empty_audit = _read(self.empty_report_path.parent / "analysis.json")
        reference = _read(self.fast_reference_report_path)
        source_before_portable = _portable_source_hashes(self.source_before)
        expected_configuration = (
            reference.get("status") == "passed"
            and reference.get("particle_subdivision_per_bank_cell") == 2
            and abs(float(reference.get("soil_voxel_m")) - 0.1) <= 1e-12
            and reference.get("soil_hz") == 60
            and reference.get("numerical_profile") == "train-fast"
            and reference.get("solver_requested") == "auto"
            and reference.get("training_tier_accepted") is True
            and reference.get("sources_unchanged") is True
            and _portable_source_hashes(reference.get("sources_before", {}))
            == source_before_portable
        )
        if not (geometry.get("status") == "passed"
                and geometry.get("particle_subdivision_per_bank_cell") == 2
                and abs(float(geometry.get("soil_voxel_m")) - 0.1) <= 1e-12
                and geometry.get("sources_unchanged") is True):
            raise ValueError("fast_geometry_qualification_mismatch")
        if not (empty.get("status") == "passed" and empty.get("mode") == "empty"
                and empty.get("particle_subdivision_per_bank_cell") == 2
                and abs(float(empty.get("soil_voxel_m")) - 0.1) <= 1e-12
                and empty.get("sources_unchanged") is True and empty_audit.get("passed") is True
                and _portable_source_hashes(empty.get("sources_before", {}))
                == source_before_portable):
            raise ValueError("fast_empty_source_qualification_mismatch")
        if not expected_configuration:
            raise ValueError("fast_contact_reference_mismatch")

        import newton
        import warp as wp
        import bank_runtime
        from bank_soil import Soil
        from coupled import velocity
        from moving_scene import to_pose
        from objective_strain import install as install_objective

        self.wp = wp
        self.velocity = velocity
        self.to_pose = to_pose
        wp.init()
        wp.set_device("cuda:0")
        install_objective()
        newton.use_coord_layout_targets = True
        (self.asm, self.chain, self.initial, self.geom, self.poses,
         self.scene, self.bounds, self.parts) = setup()
        self.envelope = WorkEnvelope(self.scene.tri)
        requested_bank, initial_particles, _ = build_particle_bank(2)
        model, bucket, initial_poses, _, self.names = bank_runtime.build(
            T / "scene", R / "manual-comparison-20260908-v1", particle_subdivision=2)
        # Rendering-only handle for interactive acquisition.  The physics state
        # remains owned by ``self.soil`` and the runtime timing is unchanged.
        self.collider_model = model
        if not np.allclose(initial_poses["bucket"], self.poses["bucket"], rtol=0, atol=1e-10):
            raise ValueError("initial_bucket_pose_mismatch")
        cfg = read(B / "manual-20260911-090834/soil-config.json")
        fields = ("friction", "young_modulus", "poisson_ratio", "yield_pressure",
                  "yield_stress", "tensile_yield_ratio", "hardening", "hardening_rate",
                  "softening_rate", "dilatancy", "viscosity", "damping")
        numerics = numerical_profile("train-fast")
        settings = {
            "material": {key: cfg["material"][key] for key in fields},
            "voxel": 0.1,
            "numerics": {"max_iterations": numerics["max_iterations"],
                         "tolerance": numerics["tolerance"], "solver": "auto"},
            "static_sweep": True,
        }
        inventory_contract_path = B / "inventory-contract.json"
        inventory_contract = read(inventory_contract_path)
        self.soil = Soil(model, bucket, inventory_contract, self.output,
                         spacing=bank_runtime.bank.spacing, settings=settings,
                         speed_limit_enforced=False)
        initial_q, initial_v = self.soil.host_snapshot()
        exact_particles = (np.array_equal(initial_particles, initial_q)
                           and not np.any(initial_v))
        particle_hash = particle_sha256(initial_q)
        if not exact_particles or particle_hash != geometry["initial_particle_sha256"]:
            raise ValueError("initial_particle_state_mismatch")
        self.soil.state.body_q.assign(wp.array(
            [to_pose(self.poses[name]) for name in self.names], dtype=wp.transform))
        self.soil.state.body_qd.zero_()
        self.soil.sample(self.poses["bucket"])
        self.soil_poses = {name: self.poses[name].copy() for name in self.names}
        self.sensor = Sensor(self.geom, self.parts)
        self.mech = Mechanism(M, self.initial, self.output / "mechanism.stderr")
        self.state = self.mech.state
        self.point = np.asarray(self.state["bucket_origin"], dtype=float)
        self._update_geometry()
        if self.fill_root is not None:
            from online_truth import OnlineBucketFill, SoilBodySeparationDetector, sha256
            front_path = A / "retention-front-rim.npz"
            front = np.load(front_path)["front"]
            self.separation_detector = SoilBodySeparationDetector(
                initial_q, inventory_contract, front, particle_spacing_m=0.05)
            self.online_fill = OnlineBucketFill.from_paths(
                fill_module=self.fill_root / "fill_volume.py",
                calibration_report=self.fill_calibration_report,
                held_out_report=self.fill_held_out_report,
                contract=inventory_contract, front_rim_xz=front)
            source_paths = (
                Path(__file__).resolve().with_name("online_truth.py"),
                self.fill_root / "fill_volume.py", self.fill_calibration_report,
                self.fill_held_out_report, inventory_contract_path, front_path,
            )
            self.truth_sources = {str(path): sha256(path) for path in source_paths}
            self.truth_sources_before = dict(self.truth_sources)
        self.qualification = {
            "geometry_report_sha256": _sha256(self.geometry_report_path),
            "empty_report_sha256": _sha256(self.empty_report_path),
            "empty_audit_sha256": _sha256(self.empty_report_path.parent / "analysis.json"),
            "fast_reference_report_sha256": _sha256(self.fast_reference_report_path),
            "source_set_matches_empty_qualification": True,
            "initial_particle_state_exact": exact_particles,
            "initial_particle_sha256": particle_hash,
            "configuration": {
                "particles": len(initial_q), "subdivision": 2, "soil_voxel_m": 0.1,
                "soil_hz": 60, "mechanism_hz": 120, "numerical_profile": "train-fast",
                "max_iterations": 80, "tolerance": 5e-4, "solver": "auto",
            },
            "scope": "independent short wiring smoke; no task-success or material qualification",
        }
        if self.online_fill is not None:
            self.qualification["online_truth"] = {
                "separation_schema": "soil_body_separation_truth_v1",
                "fill_schema": "online_bucket_fill_truth_v1",
                "fill_method": "bucket-local-density-v2-weighted",
                "fill_profile": "fast-5cm",
                "source_hashes": dict(self.truth_sources_before),
                "scope": "read-only fast-tier truth; no success qualification by installation alone",
            }

    def _update_geometry(self):
        self.poses = self.asm.assemble(np.asarray(self.state["qpos"]),
                                       include_meshes=False)[1]
        clearance = self.scene.check(self.poses, "SMOKE")
        if np.linalg.norm(np.asarray(self.state["bucket_origin"])
                          - self.poses["bucket"][:3, 3]) > 1e-8:
            raise ValueError("dynamic_geometry_alignment")
        target = np.asarray(self.state["target"])
        if np.any(target < self.bounds[:, 0] - 1e-10) or np.any(
                target > self.bounds[:, 1] + 1e-10):
            raise ValueError("internal_target_outside_autonomous_8mm")
        self.clearance = clearance

    def observe(self):
        return {
            "sim_time_s": float(self.state["time_s"]),
            "qpos": self.state["qpos"],
            "qvel": self.state["qvel"],
            "axis_qpos_address": self.state["axis_qpos_address"],
            "axis_dof_address": self.state["axis_dof_address"],
            "bucket_origin": self.state["bucket_origin"],
            "bucket_rotation": self.state["bucket_rotation"],
            "closure_m": self.state["closure_m"],
            "contact_count": self.state["contact_count"],
            "bucket_min_z_m": self.clearance["bucket_min_z_m"],
        }

    def capture_terrain(self):
        terrain, _, metadata = self.sensor.capture(
            self.soil.host_positions(), self.soil_poses, self.capture_calls)
        self.capture_calls += 1
        heights = np.asarray(terrain["height_m"], dtype=float)
        known = np.isfinite(heights)
        return {
            "height_m": [[float(value) if np.isfinite(value) else None for value in row]
                         for row in heights],
            "known_mask": known.tolist(),
            "valid": bool(terrain["valid"]),
            "origin_xy_m": terrain["origin_xy_m"],
            "resolution_m": float(terrain["resolution_m"]),
            "shape_yx": list(heights.shape),
            "source": metadata["source"],
        }

    def mechanism_step(self, command):
        applied_force = self.force.copy()
        applied_torque = self.torque.copy()
        applied_point = self.point.copy()
        self.state = self.mech.step(command.physical_velocity_si, self.force,
                                    self.torque, self.point, dt=command.dt_s, substeps=20)
        self.mechanism_steps += 1
        self._update_geometry()
        row = {
            "mechanism_step": self.mechanism_steps,
            "time_s": float(self.state["time_s"]),
            "command": command.as_dict(),
            "applied_force_n": applied_force.tolist(),
            "applied_torque_nm": applied_torque.tolist(),
            "applied_point_world_m": applied_point.tolist(),
            "qpos": self.state["qpos"],
            "qvel": self.state["qvel"],
            "closure_m": self.state["closure_m"],
        }
        self.mechanism_log.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.logged_mechanism_steps += 1

    def soil_step(self):
        newposes = self.poses
        dt = 1.0 / 60.0
        self.soil.state.body_q.assign(self.wp.array(
            [self.to_pose(self.soil_poses[name]) for name in self.names],
            dtype=self.wp.transform))
        self.soil.state.body_qd.assign(self.wp.array(
            [self.velocity(self.soil_poses[name], newposes[name], dt)
             for name in self.names], dtype=self.wp.spatial_vector))
        self.soil.state.body_f.zero_()
        self.soil.step(dt, newposes["bucket"], record=(self.soil.steps + 1) % 6 == 0)
        self.soil_steps += 1
        self.point = self.soil.last_contact_point.copy()
        self.force = self.soil.last_force.copy()
        self.torque = self.soil.last_torque.copy()
        self.digging_wrench_observed |= bool(np.linalg.norm(self.force) > 1e-6
                                             or np.linalg.norm(self.torque) > 1e-6)
        self.soil.state.body_q.assign(self.wp.array(
            [self.to_pose(newposes[name]) for name in self.names], dtype=self.wp.transform))
        self.soil_poses = {name: newposes[name].copy() for name in self.names}
        if self.soil_steps % 6 == 0:
            outside = self.envelope.outside(self.soil.host_positions())
            count = int(outside.sum())
            self.maximum_scene_outliers = max(self.maximum_scene_outliers, count)
            if count:
                raise ValueError("particle_outside_full_work_envelope")
        row = {
            "soil_step": self.soil_steps,
            "sim_time_s": float(self.soil.time),
            "force_n": self.force.tolist(),
            "torque_nm": self.torque.tolist(),
            "point_world_m": self.point.tolist(),
        }
        self.soil_event_log.write(json.dumps(
            row, separators=(",", ":"), allow_nan=False) + "\n")
        self.logged_soil_steps += 1

    def current_wrench(self):
        return tuple(np.concatenate((self.force, self.torque)))

    def _mouth_angle_deg(self):
        pose = self.poses["bucket"]
        tip = pose[:3, :3] @ TIP_LOCAL_M + pose[:3, 3]
        mouth = pose[:3, :3] @ MOUTH_LOCAL_M + pose[:3, 3]
        line = mouth - tip
        return float(math.degrees(math.atan2(line[2], np.linalg.norm(line[:2]))))

    def _truth_snapshot(self, *, require_fill=False):
        if self.separation_detector is None or self.online_fill is None:
            return None, None, False
        positions = None
        if self.separation_cache_soil_step != self.soil_steps:
            positions = self.soil.host_positions()
            self.separation_cache = self.separation_detector.analyze(
                positions, self.poses["bucket"])
            self.separation_cache_soil_step = self.soil_steps
            self.separation_evaluations += 1
        if require_fill:
            if positions is None:
                positions = self.soil.host_positions()
            self.latest_fill = self.online_fill.measure(positions, self.poses["bucket"])
            self.latest_fill_soil_step = self.soil_steps
            self.fill_evaluations += 1
        fill_fresh = (self.latest_fill is not None
                      and self.latest_fill_soil_step == self.soil_steps)
        return self.separation_cache, self.latest_fill if fill_fresh else None, fill_fresh

    def termination_inputs(self, *, require_fill=False):
        separation, fill, fill_fresh = self._truth_snapshot(require_fill=require_fill)
        if separation is not None:
            current_force_contact = bool(np.linalg.norm(self.force) >= 1.0)
            valid_fill = bool(fill_fresh and fill["valid"])
            return {
                "mouth_angle_deg": self._mouth_angle_deg(),
                "digging_interaction_observed_from_mpm_wrench": self.digging_wrench_observed,
                "digging_contact_current": bool(
                    current_force_contact and separation["terrain_contact"] is True),
                "terrain_contact": separation["terrain_contact"],
                "soil_body_separation_valid": separation["valid"],
                "soil_body_separation": separation,
                "fill_fraction": fill["fill_fraction"] if valid_fill else None,
                "fill_measurement_valid": valid_fill,
                "fill_measurement_fresh": fill_fresh,
                "fill_measurement": fill,
                "termination_machine_advanced": False,
                "reason": None if separation["valid"] else separation["reason"],
            }
        return {
            "mouth_angle_deg": self._mouth_angle_deg(),
            "digging_interaction_observed_from_mpm_wrench": self.digging_wrench_observed,
            "terrain_contact": None,
            "soil_body_separation_valid": False,
            "fill_fraction": None,
            "fill_measurement_valid": False,
            "termination_machine_advanced": False,
            "reason": "validated soil-body separation and online 2 cm fill are not implemented",
        }

    def close(self):
        if self.closed:
            return
        errors = []
        if self.mech is not None:
            try:
                self.mech.close()
            except BaseException as error:
                errors.append(repr(error))
        if self.soil is not None:
            try:
                self.soil.close()
            except BaseException as error:
                errors.append(repr(error))
        self.mechanism_log.close()
        self.soil_event_log.close()
        if self._common_sources is not None:
            self.source_after = self._common_sources()
        if self.truth_sources_before is not None:
            from online_truth import sha256
            self.truth_sources_after = {
                path: sha256(path) for path in self.truth_sources_before}
        self.closed = True
        if errors:
            raise RuntimeError("cleanup_failed:" + ";".join(errors))


def actor_observation(frame, target_world_m):
    state = frame["backend_state"]
    qpos = np.asarray(state["qpos"], dtype=float)
    qvel = np.asarray(state["qvel"], dtype=float)
    qpos_addresses = np.asarray(state["axis_qpos_address"], dtype=int)
    dof_addresses = np.asarray(state["axis_dof_address"], dtype=int)
    rotation = np.asarray(state["bucket_rotation"], dtype=float).reshape(3, 3)
    bucket_position = np.asarray(state["bucket_origin"], dtype=float)
    target = np.asarray(target_world_m, dtype=float)
    cylinders = qpos[qpos_addresses[1:]]
    cylinder_normalized = (2.0 * (cylinders - CYLINDER_LIMITS_M[:, 0])
                           / (CYLINDER_LIMITS_M[:, 1] - CYLINDER_LIMITS_M[:, 0]) - 1.0)
    axis_velocity = qvel[dof_addresses] / np.asarray(ACTION_SPEED_LIMIT_SI)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = bucket_position
    tip = rotation @ TIP_LOCAL_M + bucket_position
    mouth = rotation @ MOUTH_LOCAL_M + bucket_position
    mouth_line = mouth - tip
    mouth_angle = math.degrees(math.atan2(mouth_line[2], np.linalg.norm(mouth_line[:2])))
    history = frame["wrench_history_world_si"]
    values = list(history) + [None] * (4 - len(history))
    valid = [True] * len(history) + [False] * (4 - len(history))
    terrain = frame["terrain"]
    value = {
        "schema": OBSERVATION_SCHEMA,
        "decision_step": frame["decision_step"],
        "sim_time_s": frame["sim_time_s"],
        "terrain": {
            "height_m": terrain["height_m"],
            "known_mask": terrain["known_mask"],
            "valid": terrain["valid"],
            "capture_sim_s": frame["terrain_capture_sim_s"],
            "age_s": frame["terrain_age_s"],
            "sequence": frame["terrain_sequence"],
        },
        "target_world_m": list(target_world_m),
        "target_valid": True,
        "proprioception": {
            "swing_sin_cos": [math.sin(qpos[qpos_addresses[0]]),
                              math.cos(qpos[qpos_addresses[0]])],
            "cylinder_position_normalized": cylinder_normalized.tolist(),
            "axis_velocity_normalized": axis_velocity.tolist(),
            "axis_valid_mask": [True, True, True, True],
        },
        "bucket": {
            "position_world_m": bucket_position.tolist(),
            "orientation_quaternion_wxyz": _quaternion_wxyz(rotation).tolist(),
            "relative_to_target_m": (bucket_position - target).tolist(),
            "mouth_angle_deg": float(mouth_angle),
            "valid": True,
        },
        "wrench_history": {
            "values_world_si": values,
            "age_s": [index * POLICY_DT_S for index in range(4)],
            "valid_mask": valid,
        },
        "previous_action": frame["previous_action"],
        "elapsed_fraction": frame["sim_time_s"] / 60.0,
        "digging_interaction_observed": bool(
            frame["termination_inputs"]["digging_interaction_observed_from_mpm_wrench"]),
    }
    validate_observation(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--empty-report", type=Path, required=True)
    parser.add_argument("--fast-reference-report", type=Path, required=True)
    parser.add_argument("--episode-id", default="four-axis-mpm-smoke-v1")
    parser.add_argument("--decisions", type=int, default=60)
    parser.add_argument("--max-wall", type=float, default=120.0)
    args = parser.parse_args()
    if args.decisions != 60:
        raise ValueError("smoke_requires_exactly_60_decisions")
    if args.max_wall <= 0 or not math.isfinite(args.max_wall):
        raise ValueError("invalid_max_wall")
    geometry = _read(args.geometry_report)
    target = geometry["selection"]["entry_xy_m"] + [geometry["selection"]["surface_z_m"]]
    reset_value = {"schema": RESET_SCHEMA, "episode_id": args.episode_id,
                   "target_world_m": target}
    ResetRequest.parse(reset_value)
    if args.output.exists():
        raise FileExistsError("output_directory_exists:" + str(args.output))

    made = []

    def factory(request):
        backend = LiveFastTierBackend(
            request, backend_root=args.backend_root, output=args.output,
            geometry_report=args.geometry_report, empty_report=args.empty_report,
            fast_reference_report=args.fast_reference_report)
        made.append(backend)
        return backend

    runtime = FourAxisRuntime(factory)
    started = time.perf_counter()
    report = {
        "schema": "excavation_four_axis_mpm_smoke_v1",
        "status": "initializing",
        "episode_id": args.episode_id,
        "requested_decisions": args.decisions,
        "requested_sim_time_s": args.decisions * POLICY_DT_S,
        "max_wall_s": args.max_wall,
        "training_started": False,
        "task_success_evaluated": False,
        "M2_loose_v1_qualified": False,
    }
    policy_log = None
    terrain_log = None
    try:
        frame = runtime.reset(reset_value)
        backend = made[0]
        policy_log = (args.output / "policy-steps.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        terrain_log = (args.output / "terrain-frames.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        last_terrain_sequence = -1
        for decision in range(args.decisions):
            if time.perf_counter() - started > args.max_wall:
                raise TimeoutError("smoke_wall_budget")
            if (args.output / "STOP").exists():
                raise InterruptedError("cooperative_stop")
            observation = actor_observation(frame, target)
            action = _smoke_action(decision, args.decisions)
            result = runtime.step(step_request(frame, action))
            next_frame = result["observation"]
            next_observation = actor_observation(next_frame, target)
            policy_log.write(json.dumps({
                "decision_step": decision,
                "observation": observation,
                "action": {"schema": ACTION_SCHEMA, "axis_order": list(AXES),
                           "normalized_velocity": action},
                "plan": result["plan"],
                "next_observation": next_observation,
                "termination_inputs": next_frame["termination_inputs"],
            }, separators=(",", ":"), allow_nan=False) + "\n")
            if next_frame["terrain_sequence"] != last_terrain_sequence:
                terrain_log.write(json.dumps({
                    "terrain_sequence": next_frame["terrain_sequence"],
                    "capture_sim_s": next_frame["terrain_capture_sim_s"],
                    "terrain": next_frame["terrain"],
                }, separators=(",", ":"), allow_nan=False) + "\n")
                last_terrain_sequence = next_frame["terrain_sequence"]
            frame = next_frame
        policy_log.flush()
        terrain_log.flush()
        report["status"] = "passed"
        report["final_observation"] = actor_observation(frame, target)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
    finally:
        cleanup_error = None
        try:
            runtime.close()
        except BaseException as error:
            cleanup_error = repr(error)
            report["status"] = "failed"
            report["cleanup_error"] = cleanup_error
        if policy_log is not None:
            policy_log.close()
        if terrain_log is not None:
            terrain_log.close()
        report["wall_s"] = time.perf_counter() - started
        report["runtime_lifecycle"] = runtime.lifecycle
        if made:
            backend = made[0]
            report["runtime_counters"] = runtime.counters
            report["backend_counters"] = {
                "mechanism_steps": backend.mechanism_steps,
                "logged_mechanism_steps": backend.logged_mechanism_steps,
                "soil_steps": backend.soil_steps,
                "logged_soil_steps": backend.logged_soil_steps,
                "terrain_captures": backend.capture_calls,
                "maximum_scene_outliers": backend.maximum_scene_outliers,
            }
            report["qualification"] = backend.qualification
            report["sources_unchanged"] = (
                backend.source_before is not None
                and backend.source_before == backend.source_after)
            if report["status"] == "passed":
                expected = {
                    "policy_steps": 60, "mechanism_steps": 360,
                    "soil_steps": 180, "terrain_captures": 16,
                }
                counts_ok = (runtime.counters == expected
                             and backend.mechanism_steps == backend.logged_mechanism_steps == 360
                             and backend.soil_steps == backend.logged_soil_steps == 180
                             and backend.capture_calls == 16)
                report["checks"] = {
                    "exact_20_120_60_cadence": runtime.counters == expected,
                    "physical_and_log_counts_match": counts_ok,
                    "five_hz_terrain_schedule": backend.capture_calls == 16,
                    "source_files_unchanged": report["sources_unchanged"],
                    "termination_gap_not_silently_filled": (
                        report["final_observation"] is not None
                        and not frame["termination_inputs"]["soil_body_separation_valid"]
                        and not frame["termination_inputs"]["fill_measurement_valid"]
                        and not frame["termination_inputs"]["termination_machine_advanced"]),
                    "training_not_started": True,
                }
                if not all(report["checks"].values()):
                    report["status"] = "failed"
                    report["error"] = "post_run_check_failed"
        if args.output.exists():
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: report.get(key) for key in (
            "status", "error", "wall_s", "runtime_counters", "backend_counters",
            "sources_unchanged", "checks")}, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
