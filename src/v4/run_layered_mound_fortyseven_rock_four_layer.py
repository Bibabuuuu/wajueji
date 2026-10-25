#!/usr/bin/env python3
"""Restore the admitted 33-rock state, add the fourth layer, and checkpoint 47 rocks."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_fullscale_graded_gravel_slope import _inside_convex, sat_axis_gap
from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_arrays,
    checkpoint_identity,
    load_checkpoint_readonly,
    restore_fixture_prefix,
    transfer_solver_warmstart,
    write_checkpoint_atomic,
)
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    build_model_and_bridge,
    dynamic_step,
    restore_runtime,
    run_continuity_hold,
)
from mixed_material_r0 import normalized_prototypes, quaternion_matrix_wxyz
from mpm_rigid_free_velocity import predict_free_linear_velocity
from preview_layered_mound_existing_viewer import _mound_mask
from run_layered_mound_single_rock_press_smoke import _finite_tree, _rotation_matrix
from run_layered_mound_three_rock_adaptive_press import MultiRockFixture, _zero
from run_layered_mound_three_rock_cover_release import ParticleActivation, _stable_particle
from run_mixed_material_r3_a_probe import _read_obj, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_fortyseven_rock_four_layer_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_fortyseven_rock_four_layer_report_v1"
RECOVERY_SCHEMA = (
    "excavation_layered_mound_fortyseven_rock_four_layer_recovery_v1")
RECOVERY_STATUS = "four_layer_post_physics_recovery_saved_not_admitted"
SOURCE_LAYERS = ("lower", "lower_middle", "middle")
NEW_LAYER = "upper"
FORMAL_INPUT_CHECKPOINT_CONTRACT = {
    "default_path": (
        "/home/user01/a30-layered-checkpoint-v2/checkpoints/"
        "three-layer-checkpoint-cuda-0-20260921-170946-35334.npz"
    ),
    "sha256": "d30db1bd42c064934badc21716d0ac6f1db80e24d009c71d27a4751b77f3f545",
    "checkpoint_schema": CHECKPOINT_SCHEMA,
    "checkpoint_status": "three_layer_continuation_checkpoint_saved",
    "source_config_sha256": "bbe3c6b3a1893093d87ee6700169fcb77960d4c3428078a533f8edde087508b1",
    "source_runner_sha256": "71b0dbca886d476462ab6e043963393b4cc6ee3276fb44054113ba1151d71a86",
    "particle_count": 221882,
    "rock_count": 33,
    "newton_version": "1.5.1",
    "warp_version": "1.17.0",
    "read_only": True,
    "metadata_required": True,
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _recovery_paths(output: Path, checkpoint_out: Path) -> tuple[Path, Path, Path]:
    recovery_checkpoint = checkpoint_out.with_name(
        checkpoint_out.stem + ".recovery" + checkpoint_out.suffix)
    recovery_metadata = recovery_checkpoint.with_suffix(
        recovery_checkpoint.suffix + ".json")
    recovery_evidence = output.with_name(
        output.stem + ".recovery" + output.suffix)
    return recovery_checkpoint, recovery_metadata, recovery_evidence


def _solver_restore_report(restored: dict) -> dict:
    """Accept both historical and current restore report layouts."""
    report = restored.get("solver_restore")
    if report is None:
        report = restored.get("restore", {}).get("solver_continuation")
    if report is None:
        raise KeyError("solver_restore_or_solver_continuation")
    return report


def _final_checks(report: dict, config: dict, source_after: dict) -> dict:
    restored = report["restore"]
    solver_restore = _solver_restore_report(restored)
    geometry = report["geometry"]
    particles = report["particles"]
    construction = report["construction"]
    release = report["release"]
    hold = report["additional_uncontrolled_hold"]
    return {
        "source_checkpoint_sha256_verified":
            report["source_checkpoint"]["sha256"] ==
            config["input_checkpoint"]["sha256"],
        "source_checkpoint_unchanged":
            source_after == report["source_checkpoint"],
        "source_particle_history_restored_exactly": all(
            restored["restore"]["field_readback_exact"].values()),
        "source_rigid_state_restored": (
            restored["restore"]["rigid_qpos_readback_within_tolerance"]
            and restored["restore"]["rigid_qvel_readback_within_tolerance"]),
        "source_solver_state_restored_exactly": (
            all(solver_restore["grid_readback_exact"].values())
            and all(solver_restore[
                "solver_field_readback_exact"].values())),
        "checkpoint_one_step_and_short_window_continuity": all(
            report["checkpoint_self_check"]["checks"].values()),
        "expanded_fixture_one_step_continuity":
            report["expanded_fixture_continuity"]["passed"],
        "upper_runtime_sat_gap_passed": (
            geometry["post_cover_runtime_sat_passed"]
            and min(
                geometry["minimum_new_vs_existing_sat_gap_m"],
                geometry["minimum_new_vs_new_sat_gap_m"],
                geometry["minimum_pre_cover_runtime_sat_gap_m"],
                geometry["minimum_post_cover_runtime_sat_gap_m"])
            >= config["construction"]["required_new_pair_sat_gap_m"]),
        "upper_cover_particles_added":
            particles["upper_cover_active_count"] > 0,
        "upper_cover_history_restored": particles[
            "activation_report"]["dormant_history_restored"],
        "upper_construction_finite": construction["finite"],
        "parked_new_rocks_without_mpm_force":
            construction["parked_force_max_n"] == 0.0,
        "upper_cover_stable_window_reached":
            construction["settle"]["passed"],
        "release_executed": release["executed"],
        "release_state_finite": release["all_finite"],
        "release_continuous_stable_window_reached":
            release["stable_window_frames"] >= int(
                config["probe"]["release_stable_window_frames"]),
        "additional_uncontrolled_hold_all_frames_stable":
            hold["all_frames_stable"],
        "no_external_rigid_support": (
            construction["external_rigid_support_substeps"] == 0
            and release["external_rigid_support_substeps"] == 0),
        "forbidden_contacts_zero": (
            construction["maximum_forbidden_contacts"] == 0
            and release["maximum_forbidden_contacts"] == 0),
        "gravity_and_mpm_global_impulse_accounted_once":
            release["maximum_global_momentum_residual_ns"] < 1e-4,
        "release_speed_safety_bounds": (
            release["maximum_rock_linear_speed_m_s"] <=
                config["probe"]["maximum_rock_linear_speed_m_s"]
            and release["maximum_rock_angular_speed_rad_s"] <=
                config["probe"]["maximum_rock_angular_speed_rad_s"]
            and release["maximum_particle_speed_m_s"] <=
                config["probe"]["maximum_particle_speed_m_s"]),
    }


def _finalize_recovery(
        *, output: Path, checkpoint_out: Path, metadata_out: Path,
        recovery_checkpoint: Path, recovery_metadata_path: Path,
        recovery_evidence_path: Path, config: dict, config_path: Path,
        source_checkpoint: Path, source_metadata_path: Path,
        data_domain: str | None) -> dict:
    finalization_started = time.perf_counter()
    recovery = json.loads(
        recovery_metadata_path.read_text(encoding="utf-8"))
    report = json.loads(recovery_evidence_path.read_text(encoding="utf-8"))
    if (recovery.get("schema") != RECOVERY_SCHEMA
            or recovery.get("status") != RECOVERY_STATUS
            or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != RECOVERY_STATUS):
        raise ValueError("fortyseven_recovery_schema_or_status")
    if (recovery.get("intended_checkpoint_file") != checkpoint_out.name
            or recovery.get("intended_report_file") != output.name
            or recovery.get("recovery_checkpoint_file") !=
                recovery_checkpoint.name
            or recovery.get("recovery_evidence_file") !=
                recovery_evidence_path.name):
        raise ValueError("fortyseven_recovery_file_binding")
    if (checkpoint_identity(recovery_checkpoint)["sha256"] !=
            recovery.get("recovery_checkpoint_sha256")
            or sha256(recovery_evidence_path) !=
            recovery.get("recovery_evidence_sha256")):
        raise ValueError("fortyseven_recovery_content_binding")
    if (sha256(config_path) != recovery.get("source_config_sha256")
            or sha256(source_metadata_path) !=
                recovery.get("source_metadata_sha256")
            or str(source_checkpoint) != recovery.get("source_checkpoint")
            or data_domain != recovery.get("data_domain")):
        raise ValueError("fortyseven_recovery_input_binding")

    source_after = checkpoint_identity(source_checkpoint)
    checks = _final_checks(report, config, source_after)
    passed = all(checks.values())
    report["checks"] = checks
    report["source_checkpoint_after"] = source_after
    report["status"] = ("four_layer_checkpoint_saved" if passed
                        else "fortyseven_rock_four_layer_failed")
    report["next_gate"] = (
        "restore_47_checkpoint_then_add_top_18" if passed
        else "inspect_four_layer_recovery_evidence")
    report["execution"]["finalized_from_recovery"] = True
    report["execution"]["finalization_wall_seconds"] = (
        time.perf_counter() - finalization_started)

    if passed:
        if checkpoint_out.exists() or metadata_out.exists():
            raise FileExistsError(checkpoint_out if checkpoint_out.exists()
                                  else metadata_out)
        formal_metadata = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "four_layer_continuation_checkpoint_saved",
            "checkpoint_file": checkpoint_out.name,
            "checkpoint_sha256": recovery["recovery_checkpoint_sha256"],
            "source_checkpoint_sha256": recovery[
                "source_checkpoint_sha256"],
            "source_config_sha256": recovery["source_config_sha256"],
            "source_runner_sha256": recovery["physics_runner_sha256"],
            "continuation_point": recovery["continuation_point"],
            "gravity_predictor_required_on_next_free_step": True,
            "same_solver_material_and_dt_required": True,
            "fixed_grid_reference_saved": True,
            "newton_solver_warmstart_saved": True,
            "newton_version": recovery["newton_version"],
            "warp_version": recovery["warp_version"],
            "solver_grid_type": recovery["solver_grid_type"],
            "solver_warmstart_mode": recovery["solver_warmstart_mode"],
            "collider_velocity_mode": recovery["collider_velocity_mode"],
            "array_count": recovery["array_count"],
            "fields": recovery["fields"],
            "particle_count": recovery["particle_count"],
            "rock_count": 47,
            "simulation_time_s": recovery["simulation_time_s"],
        }
        if data_domain is not None:
            formal_metadata["data_domain"] = data_domain
            formal_metadata["formal_a30_equivalent"] = False
        metadata_staging = metadata_out.with_suffix(
            metadata_out.suffix + ".finalizing")
        _write_json(metadata_staging, formal_metadata)
        report["checkpoint"] = {
            **formal_metadata,
            "metadata_file": metadata_out.name,
            "metadata_sha256": sha256(metadata_staging),
            "size_bytes": recovery["size_bytes"],
        }
        report["execution"]["wall_seconds_with_checkpoint"] = report[
            "execution"]["recovery_saved_wall_seconds"]
        if not _finite_tree(report):
            raise RuntimeError("fortyseven_nonfinite_final_report")
        _write_json(output, report)
        recovery_checkpoint.replace(checkpoint_out)
        metadata_staging.replace(metadata_out)
        recovery["promoted_checkpoint_file"] = checkpoint_out.name
        recovery["promoted_report_file"] = output.name
        recovery["requires_finalization"] = False
        recovery["status"] = "four_layer_recovery_promoted_to_formal"
        _write_json(recovery_metadata_path, recovery)
    else:
        report["checkpoint"] = None
        report["execution"]["wall_seconds_with_checkpoint"] = report[
            "execution"]["recovery_saved_wall_seconds"]
        if not _finite_tree(report):
            raise RuntimeError("fortyseven_nonfinite_failed_report")
        _write_json(output, report)
    return report


def load_config(
        path: str | Path,
        repository_root: str | Path,
        *,
        expected_input_checkpoint: dict | None = None) -> dict:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "restore_33_verify_continuity_add_upper_14_then_checkpoint_47"):
        raise ValueError("fortyseven_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("source", ("visual_config", "source_bank")),
            ("runtime_dependencies", ("checkpoint", "runtime", "predictor"))):
        binding = config.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if not target.is_file() or sha256(target) != binding.get(field + "_sha256"):
                raise ValueError(f"fortyseven_{group}_{field}_binding")
    expected_source = (FORMAL_INPUT_CHECKPOINT_CONTRACT
                       if expected_input_checkpoint is None
                       else expected_input_checkpoint)
    source = config.get("input_checkpoint", {})
    if source != expected_source:
        raise ValueError("fortyseven_input_checkpoint_contract")
    targets = config.get("targets", [])
    if [item.get("manifest_rock_id") for item in targets] != [
            f"upper-{index:03d}" for index in range(14)]:
        raise ValueError("fortyseven_target_ids")
    schedule = [(float(item["press_distance_m"]), int(item["press_frames"]))
                for item in targets]
    if schedule != [
            (0.006, 9), (0.008, 12), (0.005, 8), (0.009, 14),
            (0.007, 11), (0.006, 9), (0.008, 12), (0.005, 8),
            (0.007, 11), (0.006, 9), (0.008, 12), (0.005, 8),
            (0.007, 11), (0.006, 9)]:
        raise ValueError("fortyseven_press_schedule")
    construction = config.get("construction", {})
    expected_construction = {
        "start_surface_clearance_m": 0.005,
        "park_new_rocks_position_world_m": [0.0, 0.0, 5.0],
        "source_lower_height_exclusive_m": 1.12,
        "upper_cover_height_inclusive_m": 1.40,
        "raw_upper_cover_particle_count": 39056,
        "runtime_vertical_clearance_step_m": 0.002,
        "runtime_vertical_clearance_maximum_m": 0.16,
        "required_new_pair_sat_gap_m": 0.001,
        "checkpoint_self_check_frames": 12,
        "expanded_fixture_continuity_frames": 1,
        "cover_maximum_settle_frames": 240,
        "cover_stable_window_frames": 60,
        "temporary_mpm_damping": 0.0,
        "temporary_rigid_damping": 0.0,
        "free_fall_from_height": False,
    }
    if construction != expected_construction:
        raise ValueError("fortyseven_construction_contract")
    probe = config.get("probe", {})
    expected_probe = {
        "output_dt_s": 1.0 / 120.0,
        "mujoco_internal_substeps": 40,
        "maximum_release_frames": 360,
        "release_stable_window_frames": 120,
        "additional_uncontrolled_hold_frames": 240,
        "maximum_stable_rock_linear_speed_m_s": 0.01,
        "maximum_stable_rock_angular_speed_rad_s": 0.15,
        "maximum_stable_particle_speed_m_s": 0.05,
        "maximum_stable_particle_p99_speed_m_s": 0.01,
        "maximum_particle_speed_m_s": 5.0,
        "maximum_rock_linear_speed_m_s": 5.0,
        "maximum_rock_angular_speed_rad_s": 25.0,
        "minimum_world_z_m": -0.05,
    }
    if probe != expected_probe:
        raise ValueError("fortyseven_probe_contract")
    if config.get("scope") != {
            "source_checkpoint_mutated": False,
            "source_rock_count": 33,
            "new_layer": "upper",
            "new_rock_count": 14,
            "output_rock_count": 47,
            "completed_layers": 4,
            "full_scene": False,
            "stable_scene_claimed": False,
            "fullscale_qualified": False,
            "training": False}:
        raise ValueError("fortyseven_scope_contract")
    return config


def _geometry(record: dict, position: np.ndarray,
              quaternion: np.ndarray, prototypes: dict) -> dict:
    prototype = prototypes[record["prototype_id"]]
    rotation = quaternion_matrix_wxyz(quaternion)
    vertices = (prototype["vertices"] * float(record["equivalent_diameter_m"]))
    return {
        "id": record["rock_id"],
        "vertices": vertices @ rotation.T + position,
        "faces": prototype["faces"],
        "record": record,
        "position": position,
        "quaternion": quaternion,
    }


def _minimum_cross_gap(first: list[dict], second: list[dict]) -> tuple[float, list[list[str]]]:
    gaps, intersections = [], []
    for a in first:
        for b in second:
            intersects, gap = sat_axis_gap(
                a["vertices"], a["faces"], b["vertices"], b["faces"])
            gaps.append(float(gap))
            if intersects:
                intersections.append([a["id"], b["id"]])
    return min(gaps, default=math.inf), intersections


def _prepare_upper(config: dict, root: Path, source_records: list[dict],
                   upper_records: list[dict], source_qpos: np.ndarray) -> dict:
    visual = json.loads(_resolve(
        root, config["source"]["visual_config_path"]).read_text(encoding="utf-8"))
    with np.load(_resolve(root, config["source"]["source_bank_path"]),
                 allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    mound = _mound_mask(source_q, visual)
    construction = config["construction"]
    lower_height = float(construction["source_lower_height_exclusive_m"])
    upper_height = float(construction["upper_cover_height_inclusive_m"])
    raw_mask = (mound & (source_q[:, 2] > lower_height + 1e-9)
                & (source_q[:, 2] <= upper_height + 1e-9))
    if int(np.count_nonzero(raw_mask)) != int(
            construction["raw_upper_cover_particle_count"]):
        raise ValueError("fortyseven_raw_upper_cover_count")
    lower_mask = mound & (source_q[:, 2] <= lower_height + 1e-9)
    spacing = float(config["mpm"]["particle_spacing_m"])
    surface_extent = float(source_q[lower_mask, 2].max()) + 0.45 * spacing
    prototype_ids = sorted({record["prototype_id"]
                            for record in source_records + upper_records})
    prototypes = normalized_prototypes(prototype_ids)
    existing = [
        _geometry(record, pose[:3], pose[3:7], prototypes)
        for record, pose in zip(source_records, source_qpos, strict=True)
    ]
    required = float(construction["required_new_pair_sat_gap_m"])
    step = float(construction["runtime_vertical_clearance_step_m"])
    maximum = float(construction["runtime_vertical_clearance_maximum_m"])
    accepted: list[dict] = []
    starts, pressed, lifts = [], [], []
    for target, record in zip(config["targets"], upper_records, strict=True):
        vertices, _ = _read_obj(
            root / "assets/v4/mixed-material-r0" / f"{record['prototype_id']}.obj")
        rotation = _rotation_matrix(record["orientation_quaternion_wxyz"])
        local = (np.asarray(vertices, dtype=np.float64)
                 * float(record["equivalent_diameter_m"])) @ rotation.T
        base = np.asarray(record["position_world_m"], dtype=np.float64)
        base[2] = (surface_extent
                   + float(construction["start_surface_clearance_m"])
                   - float(local[:, 2].min()))
        chosen = None
        for lift in np.arange(0.0, maximum + 0.5 * step, step):
            start = base.copy()
            start[2] += float(lift)
            end = start.copy()
            end[2] -= float(target["press_distance_m"])
            candidate = _geometry(
                record, end,
                np.asarray(record["orientation_quaternion_wxyz"], dtype=np.float64),
                prototypes)
            gap, intersections = _minimum_cross_gap(
                [candidate], existing + accepted)
            if not intersections and gap >= required - 1e-12:
                chosen = (start, end, float(lift), candidate)
                break
        if chosen is None:
            raise ValueError(f"fortyseven_no_upper_placement:{record['rock_id']}")
        start, end, lift, candidate = chosen
        starts.append(start)
        pressed.append(end)
        lifts.append(lift)
        accepted.append(candidate)
    new_gap, new_intersections = _minimum_cross_gap(accepted, existing)
    within_gap, within_intersections = _minimum_cross_gap(
        accepted, [])
    if len(accepted) > 1:
        internal_gaps = []
        internal_intersections = []
        for first, second in combinations(accepted, 2):
            intersects, gap = sat_axis_gap(
                first["vertices"], first["faces"],
                second["vertices"], second["faces"])
            internal_gaps.append(float(gap))
            if intersects:
                internal_intersections.append([first["id"], second["id"]])
        within_gap = min(internal_gaps)
        within_intersections = internal_intersections
    if (new_intersections or within_intersections
            or min(new_gap, within_gap) < required - 1e-12):
        raise ValueError("fortyseven_prepared_upper_sat")
    remove = np.zeros(len(source_q), dtype=bool)
    removed_by_rock = {}
    for item in existing + accepted:
        record = item["record"]
        prototype = prototypes[record["prototype_id"]]
        transform = quaternion_matrix_wxyz(item["quaternion"])
        candidates = np.flatnonzero(
            raw_mask
            & np.all(source_q >= item["vertices"].min(axis=0) - 1e-10, axis=1)
            & np.all(source_q <= item["vertices"].max(axis=0) + 1e-10, axis=1))
        local_candidates = (
            (source_q[candidates].astype(np.float64) - item["position"])
            @ transform / float(record["equivalent_diameter_m"]))
        selected = candidates[_inside_convex(
            local_candidates, prototype["vertices"], prototype["faces"])]
        newly_removed = selected[~remove[selected]]
        remove[selected] = True
        removed_by_rock[item["id"]] = int(len(newly_removed))
    cover_q = source_q[raw_mask & ~remove]
    if len(cover_q) == 0:
        raise ValueError("fortyseven_empty_upper_cover")
    return {
        "cover_q": cover_q,
        "starts": starts,
        "pressed": pressed,
        "vertical_clearance_lifts_m": lifts,
        "minimum_new_vs_existing_sat_gap_m": new_gap,
        "minimum_new_vs_new_sat_gap_m": within_gap,
        "removed_by_rock": removed_by_rock,
        "raw_count": int(np.count_nonzero(raw_mask)),
        "active_count": int(len(cover_q)),
        "cover_q_sha256": hashlib.sha256(cover_q.tobytes()).hexdigest(),
        "visual": visual,
    }


def _current_geometry(records: list[dict], positions: list[np.ndarray],
                      quaternions: list[np.ndarray]) -> list[dict]:
    prototypes = normalized_prototypes(sorted({record["prototype_id"]
                                               for record in records}))
    return [_geometry(record, position, quaternion, prototypes)
            for record, position, quaternion in zip(
                records, positions, quaternions, strict=True)]


def run(
        args,
        *,
        expected_input_checkpoint: dict | None = None,
        data_domain: str | None = None) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    checkpoint_out = Path(args.output_checkpoint).resolve()
    metadata_out = checkpoint_out.with_suffix(checkpoint_out.suffix + ".json")
    source_checkpoint = Path(args.checkpoint).resolve()
    source_metadata_path = Path(args.checkpoint_metadata).resolve()
    recovery_checkpoint, recovery_metadata_path, recovery_evidence_path = (
        _recovery_paths(output, checkpoint_out))
    config = load_config(
        config_path, root,
        expected_input_checkpoint=expected_input_checkpoint)
    if checkpoint_out.exists() or metadata_out.exists():
        raise FileExistsError(checkpoint_out if checkpoint_out.exists()
                              else metadata_out)
    recovery_presence = tuple(path.is_file() for path in (
        recovery_checkpoint, recovery_metadata_path, recovery_evidence_path))
    if all(recovery_presence):
        return _finalize_recovery(
            output=output, checkpoint_out=checkpoint_out,
            metadata_out=metadata_out,
            recovery_checkpoint=recovery_checkpoint,
            recovery_metadata_path=recovery_metadata_path,
            recovery_evidence_path=recovery_evidence_path,
            config=config, config_path=config_path,
            source_checkpoint=source_checkpoint,
            source_metadata_path=source_metadata_path,
            data_domain=data_domain)
    if any(recovery_presence):
        raise RuntimeError(
            "fortyseven_incomplete_recovery_artifacts:"
            + ":".join(str(path) for path, exists in zip(
                (recovery_checkpoint, recovery_metadata_path,
                 recovery_evidence_path), recovery_presence, strict=True)
                if exists))
    if output.exists():
        raise FileExistsError(output)
    arrays, source_metadata, source_before = load_checkpoint_readonly(
        source_checkpoint, source_metadata_path, config["input_checkpoint"])
    manifest_path = _resolve(root, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_records = [rock for rock in manifest["rocks"]
                      if rock["layer_id"] in SOURCE_LAYERS]
    upper_records = [rock for rock in manifest["rocks"]
                     if rock["layer_id"] == NEW_LAYER]
    if len(source_records) != 33 or len(upper_records) != 14:
        raise ValueError("fortyseven_manifest_layer_counts")
    all_records = source_records + upper_records
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    dt = float(config["probe"]["output_dt_s"])
    total_incremental_steps = 0
    with tempfile.TemporaryDirectory(prefix="fortyseven-rock-four-layer-") as raw:
        work = Path(raw)
        source_fixture, source_bridge, restore = restore_runtime(
            args, config, root, source_records, arrays, work, "restored-33")
        source_bridge_open = True
        release_bridge = None
        try:
            checkpoint_self_check = run_continuity_hold(
                source_fixture, source_bridge, source_records, config,
                int(config["construction"]["checkpoint_self_check_frames"]))
            total_incremental_steps += checkpoint_self_check["executed_frames"]
            if not all(checkpoint_self_check["checks"].values()):
                raise RuntimeError("fortyseven_checkpoint_self_check_failed")
            evolved_arrays = checkpoint_arrays(
                source_fixture, source_bridge.state,
                float(arrays["simulation_time_s"][0])
                + checkpoint_self_check["executed_frames"] * dt)
            evolved_qpos = evolved_arrays["rigid_qpos"]
            prepared = _prepare_upper(
                config, root, source_records, upper_records, evolved_qpos)
            combined_q = np.concatenate((
                evolved_arrays["particle_q"], prepared["cover_q"]), axis=0)
            fixture = MultiRockFixture(
                config, root, all_records, combined_q, args.device)
            expanded_restore = restore_fixture_prefix(fixture, evolved_arrays)
            fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            expanded_warmstart = transfer_solver_warmstart(
                source_fixture, fixture)
            activation = ParticleActivation(fixture)
            new_positions = [np.asarray(
                config["construction"]["park_new_rocks_position_world_m"],
                dtype=np.float64).copy() for _ in upper_records]
            new_velocities = [np.zeros(3) for _ in upper_records]
            new_quaternions = [np.asarray(
                record["orientation_quaternion_wxyz"], dtype=np.float64)
                for record in upper_records]
            parked_new_indices = set(range(14))
            kinematic_steps = 0
            kinematic_finite = True
            parked_force_max_n = 0.0
            existing_safety_speed_max = 0.0
            construction_external_support = 0
            construction_forbidden_contacts = 0

            def held_step() -> tuple[dict, dict, dict]:
                nonlocal kinematic_steps, kinematic_finite, parked_force_max_n
                nonlocal existing_safety_speed_max, construction_external_support
                nonlocal construction_forbidden_contacts, total_incremental_steps
                before = source_bridge.state
                gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
                positions = [np.asarray(rock["position"], dtype=np.float64)
                             for rock in before["rocks"]] + new_positions
                velocities = [predict_free_linear_velocity(
                    np.asarray(rock["qvel"][:3], dtype=np.float64), gravity, dt)
                    for rock in before["rocks"]] + new_velocities
                angular = [np.asarray(
                    rock["angular_velocity_world_rad_s"], dtype=np.float64)
                    for rock in before["rocks"]] + [np.zeros(3) for _ in upper_records]
                quaternions = [rock["qpos"][3:7] for rock in before["rocks"]] + [
                    value.tolist() for value in new_quaternions]
                fixture.assign(positions, velocities, angular, quaternions)
                measured, particles = fixture.step(dt, positions)
                wrenches = {"bucket": _zero(before["bucket_origin"])}
                for index in range(33):
                    wrenches[f"r1-rock-{index:03d}"] = measured[
                        f"r1-rock-{index:03d}"]
                after = source_bridge.step(
                    dt, int(config["probe"]["mujoco_internal_substeps"]), wrenches)
                contacts = after.get("contact_substeps", {})
                construction_external_support += sum(int(contacts.get(name, 0))
                    for name in ("rock_ground", "rock_bucket", "rock_ramp", "rock_truck"))
                construction_forbidden_contacts = max(
                    construction_forbidden_contacts,
                    int(after.get("forbidden_contact_count", 0)))
                existing_safety_speed_max = max(
                    existing_safety_speed_max,
                    max(float(np.linalg.norm(rock["qvel"][:3]))
                        for rock in after["rocks"]))
                if parked_new_indices:
                    parked_force_max_n = max(
                        parked_force_max_n,
                        max(float(np.linalg.norm(
                            measured[f"r1-rock-{33 + index:03d}"]["force"]))
                            for index in parked_new_indices))
                kinematic_steps += 1
                total_incremental_steps += 1
                kinematic_finite &= bool(particles["finite"] and _finite_tree(after))
                return measured, particles, after

            expanded_rows = []
            for _ in range(int(config["construction"][
                    "expanded_fixture_continuity_frames"])):
                _, particles, after = held_step()
                expanded_rows.append({
                    "finite": bool(particles["finite"] and _finite_tree(after)),
                    "particle_max_speed_m_s": particles["particle_max_speed_m_s"],
                    "particle_p99_speed_m_s": particles["particle_p99_speed_m_s"],
                    "maximum_existing_rock_speed_m_s": max(
                        float(np.linalg.norm(rock["qvel"][:3]))
                        for rock in after["rocks"]),
                })
            expanded_continuity_passed = all(
                row["finite"]
                and row["particle_max_speed_m_s"] <=
                    config["probe"]["maximum_stable_particle_speed_m_s"]
                and row["particle_p99_speed_m_s"] <=
                    config["probe"]["maximum_stable_particle_p99_speed_m_s"]
                and row["maximum_existing_rock_speed_m_s"] <=
                    config["probe"]["maximum_stable_rock_linear_speed_m_s"]
                for row in expanded_rows)
            if not expanded_continuity_passed:
                raise RuntimeError("fortyseven_expanded_fixture_continuity_failed")

            presses = []
            for local_index, target in enumerate(config["targets"]):
                start_position = prepared["starts"][local_index].copy()
                new_positions[local_index] = start_position.copy()
                parked_new_indices.remove(local_index)
                distance = float(target["press_distance_m"])
                frames = int(target["press_frames"])
                speed = distance / (frames * dt)
                contact_samples = 0
                for frame in range(frames):
                    new_positions[local_index] = start_position.copy()
                    new_positions[local_index][2] -= distance * (frame + 1) / frames
                    new_velocities = [np.zeros(3) for _ in upper_records]
                    new_velocities[local_index][2] = -speed
                    measured, particles, _ = held_step()
                    contact_samples += particles["contact_samples"]
                    if not particles["finite"]:
                        break
                new_velocities = [np.zeros(3) for _ in upper_records]
                presses.append({
                    "manifest_rock_id": target["manifest_rock_id"],
                    "press_distance_m": distance,
                    "press_frames": frames,
                    "press_speed_m_s": speed,
                    "runtime_vertical_clearance_lift_m": prepared[
                        "vertical_clearance_lifts_m"][local_index],
                    "contact_samples": contact_samples,
                    "held_position_world_m": new_positions[local_index].tolist(),
                })

            pre_cover_existing_positions = [
                np.asarray(rock["qpos"][:3], dtype=np.float64)
                for rock in source_bridge.state["rocks"]]
            pre_cover_existing_quaternions = [
                np.asarray(rock["qpos"][3:7], dtype=np.float64)
                for rock in source_bridge.state["rocks"]]
            existing_geometry = _current_geometry(
                source_records, pre_cover_existing_positions,
                pre_cover_existing_quaternions)
            new_geometry = _current_geometry(
                upper_records, new_positions, new_quaternions)
            pre_cover_gap, pre_cover_intersections = _minimum_cross_gap(
                new_geometry, existing_geometry)
            if (pre_cover_intersections or pre_cover_gap <
                    config["construction"]["required_new_pair_sat_gap_m"] - 1e-12):
                raise RuntimeError("fortyseven_pre_cover_sat_failed")

            active_indices = np.arange(len(combined_q), dtype=np.int64)
            activation_report = activation.set_active(active_indices)
            stable_count = 0
            settle_trace = []
            settle_contact_samples = 0
            for frame in range(int(config["construction"]["cover_maximum_settle_frames"])):
                _, particles, after = held_step()
                settle_contact_samples += particles["contact_samples"]
                existing_stable = max(
                    float(np.linalg.norm(rock["qvel"][:3]))
                    for rock in after["rocks"]) <= float(
                        config["probe"]["maximum_stable_rock_linear_speed_m_s"])
                stable = existing_stable and _stable_particle(particles, config["probe"])
                stable_count = stable_count + 1 if stable else 0
                if frame == 0 or (frame + 1) % 30 == 0:
                    settle_trace.append({
                        "frame": frame, "stable": stable,
                        "stable_window_frames": stable_count,
                        "maximum_existing_rock_speed_m_s": max(
                            float(np.linalg.norm(rock["qvel"][:3]))
                            for rock in after["rocks"]),
                        **particles,
                    })
                if not particles["finite"] or stable_count >= int(
                        config["construction"]["cover_stable_window_frames"]):
                    break
            cover_settled = (kinematic_finite and stable_count >= int(
                config["construction"]["cover_stable_window_frames"]))
            settle_report = {
                "passed": cover_settled,
                "steps": frame + 1,
                "stable_window_frames": stable_count,
                "contact_samples": settle_contact_samples,
                "trace": settle_trace,
            }

            post_cover_existing_geometry = _current_geometry(
                source_records,
                [np.asarray(rock["qpos"][:3], dtype=np.float64)
                 for rock in source_bridge.state["rocks"]],
                [np.asarray(rock["qpos"][3:7], dtype=np.float64)
                 for rock in source_bridge.state["rocks"]])
            post_cover_gap, post_cover_intersections = _minimum_cross_gap(
                new_geometry, post_cover_existing_geometry)
            post_cover_sat_passed = (
                not post_cover_intersections
                and post_cover_gap >= config["construction"][
                    "required_new_pair_sat_gap_m"] - 1e-12)

            release = {
                "executed": False, "steps": 0, "stable_window_frames": 0,
                "all_finite": True, "trace": [],
                "rock_rock_contact_substeps": 0,
                "external_rigid_support_substeps": 0,
                "maximum_forbidden_contacts": 0,
                "maximum_global_momentum_residual_ns": 0.0,
                "maximum_rock_linear_speed_m_s": 0.0,
                "maximum_rock_angular_speed_rad_s": 0.0,
                "maximum_particle_speed_m_s": 0.0,
            }
            hold = {
                "requested_frames": int(config["probe"][
                    "additional_uncontrolled_hold_frames"]),
                "executed_frames": 0, "all_frames_stable": False, "trace": [],
            }
            if cover_settled and post_cover_sat_passed:
                existing_qpos = np.asarray([
                    rock["qpos"] for rock in source_bridge.state["rocks"]],
                    dtype=np.float64)
                existing_qvel = np.asarray([
                    rock["qvel"] for rock in source_bridge.state["rocks"]],
                    dtype=np.float64)
                upper_qpos = np.asarray([
                    [*position.tolist(), *quaternion.tolist()]
                    for position, quaternion in zip(
                        new_positions, new_quaternions, strict=True)], dtype=np.float64)
                upper_qvel = np.zeros((14, 6), dtype=np.float64)
                release_qpos = np.concatenate((existing_qpos, upper_qpos), axis=0)
                release_qvel = np.concatenate((existing_qvel, upper_qvel), axis=0)
                release_records = []
                for record, pose in zip(all_records, release_qpos, strict=True):
                    item = dict(record)
                    item["position_world_m"] = pose[:3].tolist()
                    item["orientation_quaternion_wxyz"] = pose[3:7].tolist()
                    release_records.append(item)
                model, binary, model_manifest_47, bridge_manifest_47, _ = (
                    build_model_and_bridge(
                        args, root, _resolve(root, config["capacity_bridge"]["config_path"]),
                        release_records, work, "release-47",
                        Path(restore["binary"])))
                release_bridge = RestoredRuntimeBridge(
                    binary, model, 47,
                    source_bridge.state["mechanism_qpos"],
                    source_bridge.state["mechanism_qvel"],
                    release_qpos, release_qvel)
                source_bridge.close()
                source_bridge_open = False
                release["executed"] = True
                outer = np.asarray(prepared["visual"]["mound_envelope"][
                    "outer_xy_bounds_m"], dtype=np.float64)

                def released_step(frame_number: int) -> tuple[bool, dict]:
                    nonlocal total_incremental_steps
                    after, row = dynamic_step(
                        fixture, release_bridge, all_records, config, frame_number)
                    centers_inside = all(
                        outer[0, 0] <= rock["position"][0] <= outer[1, 0]
                        and outer[0, 1] <= rock["position"][1] <= outer[1, 1]
                        and rock["position"][2] >= config["probe"]["minimum_world_z_m"]
                        for rock in after["rocks"])
                    row["centers_inside"] = bool(centers_inside)
                    release["rock_rock_contact_substeps"] += row[
                        "rock_rock_contact_substeps"]
                    release["external_rigid_support_substeps"] += row[
                        "external_rigid_support_substeps"]
                    release["maximum_forbidden_contacts"] = max(
                        release["maximum_forbidden_contacts"],
                        row["forbidden_contact_count"])
                    release["maximum_global_momentum_residual_ns"] = max(
                        release["maximum_global_momentum_residual_ns"],
                        row["global_momentum_residual_ns"])
                    release["maximum_rock_linear_speed_m_s"] = max(
                        release["maximum_rock_linear_speed_m_s"],
                        row["maximum_rock_linear_speed_m_s"])
                    release["maximum_rock_angular_speed_rad_s"] = max(
                        release["maximum_rock_angular_speed_rad_s"],
                        row["maximum_rock_angular_speed_rad_s"])
                    release["maximum_particle_speed_m_s"] = max(
                        release["maximum_particle_speed_m_s"],
                        row["particle_max_speed_m_s"])
                    release["all_finite"] &= row["finite"]
                    total_incremental_steps += 1
                    return bool(row["stable"] and centers_inside), row

                consecutive = 0
                for frame in range(int(config["probe"]["maximum_release_frames"])):
                    stable, row = released_step(frame)
                    release["steps"] += 1
                    consecutive = consecutive + 1 if stable else 0
                    release["stable_window_frames"] = consecutive
                    if frame == 0 or (frame + 1) % 30 == 0:
                        release["trace"].append(row)
                    if not row["finite"] or not row["centers_inside"]:
                        break
                    if consecutive >= int(config["probe"][
                            "release_stable_window_frames"]):
                        break
                if consecutive >= int(config["probe"]["release_stable_window_frames"]):
                    for frame in range(hold["requested_frames"]):
                        stable, row = released_step(release["steps"] + frame)
                        hold["executed_frames"] += 1
                        hold["trace"].append(row)
                        if not stable or not row["finite"] or not row["centers_inside"]:
                            break
                    hold["all_frames_stable"] = (
                        hold["executed_frames"] == hold["requested_frames"]
                        and all(row["stable"] and row["centers_inside"]
                                for row in hold["trace"]))
            else:
                model_manifest_47 = bridge_manifest_47 = None

            source_after = checkpoint_identity(source_checkpoint)
            absolute_simulation_time = (
                float(arrays["simulation_time_s"][0]) + total_incremental_steps * dt)
            report = {
                "schema": REPORT_SCHEMA,
                "status": RECOVERY_STATUS,
                "checks": {},
                "source_checkpoint": source_before,
                "source_checkpoint_after": source_after,
                "source_metadata": source_metadata,
                "restore": restore,
                "checkpoint_self_check": checkpoint_self_check,
                "expanded_fixture_continuity": {
                    "passed": expanded_continuity_passed, "trace": expanded_rows,
                    "restore": expanded_restore,
                    "warmstart_transfer": expanded_warmstart,
                },
                "geometry": {
                    "minimum_new_vs_existing_sat_gap_m": prepared[
                        "minimum_new_vs_existing_sat_gap_m"],
                    "minimum_new_vs_new_sat_gap_m": prepared[
                        "minimum_new_vs_new_sat_gap_m"],
                    "minimum_pre_cover_runtime_sat_gap_m": pre_cover_gap,
                    "minimum_post_cover_runtime_sat_gap_m": post_cover_gap,
                    "post_cover_runtime_sat_passed": post_cover_sat_passed,
                    "runtime_vertical_clearance_lifts_m": prepared[
                        "vertical_clearance_lifts_m"],
                },
                "particles": {
                    "source_count": len(evolved_arrays["particle_q"]),
                    "upper_cover_raw_count": prepared["raw_count"],
                    "upper_cover_active_count": prepared["active_count"],
                    "combined_count": len(combined_q),
                    "upper_cover_q_sha256": prepared["cover_q_sha256"],
                    "removed_by_rock": prepared["removed_by_rock"],
                    "activation_report": activation_report,
                },
                "construction": {
                    "presses": presses, "settle": settle_report,
                    "kinematic_steps": kinematic_steps,
                    "finite": bool(kinematic_finite),
                    "parked_force_max_n": parked_force_max_n,
                    "maximum_existing_rock_speed_m_s": existing_safety_speed_max,
                    "external_rigid_support_substeps":
                        construction_external_support,
                    "maximum_forbidden_contacts":
                        construction_forbidden_contacts,
                },
                "release": release,
                "additional_uncontrolled_hold": hold,
                "inputs": {
                    "config_sha256": sha256(config_path),
                    "runner_sha256": sha256(Path(__file__).resolve()),
                    "manifest_sha256": sha256(manifest_path),
                    "restored_model_manifest": restore["model_manifest"],
                    "restored_bridge_manifest": restore["bridge_manifest"],
                    "release_model_manifest": model_manifest_47,
                    "release_bridge_manifest": bridge_manifest_47,
                },
                "environment": {
                    "python": platform.python_version(),
                    "newton": newton.__version__, "warp": wp.__version__,
                    "numpy": np.__version__, "device": str(wp.get_device()),
                },
                "execution": {
                    "source_simulation_time_s": float(arrays["simulation_time_s"][0]),
                    "incremental_step_count": total_incremental_steps,
                    "simulation_time_s": absolute_simulation_time,
                    "wall_seconds": time.perf_counter() - started,
                },
                "scope": config["scope"],
                "next_gate": "finalize_saved_four_layer_recovery",
            }
            if data_domain is not None:
                report["execution"]["data_domain"] = data_domain
                report["execution"]["formal_a30_equivalent"] = False
            if not _finite_tree(report):
                raise RuntimeError("fortyseven_nonfinite_report")
            if release_bridge is None:
                checks = _final_checks(report, config, source_after)
                report["checks"] = checks
                report["status"] = "fortyseven_rock_four_layer_failed"
                report["next_gate"] = "fix_four_layer_physical_gate"
                report["checkpoint"] = None
                report["execution"]["wall_seconds_with_checkpoint"] = (
                    time.perf_counter() - started)
                _write_json(output, report)
                return report

            arrays47 = checkpoint_arrays(
                fixture, release_bridge.state, absolute_simulation_time)
            identity = write_checkpoint_atomic(recovery_checkpoint, arrays47)
            report["execution"]["recovery_saved_wall_seconds"] = (
                time.perf_counter() - started)
            _write_json(recovery_evidence_path, report)
            recovery_metadata = {
                "schema": RECOVERY_SCHEMA,
                "status": RECOVERY_STATUS,
                "not_formal_checkpoint": True,
                "requires_finalization": True,
                "recovery_checkpoint_file": recovery_checkpoint.name,
                "recovery_checkpoint_sha256": identity["sha256"],
                "recovery_evidence_file": recovery_evidence_path.name,
                "recovery_evidence_sha256": sha256(recovery_evidence_path),
                "intended_checkpoint_file": checkpoint_out.name,
                "intended_report_file": output.name,
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": source_before["sha256"],
                "source_metadata_sha256": sha256(source_metadata_path),
                "source_config_sha256": sha256(config_path),
                "physics_runner_sha256": sha256(Path(__file__).resolve()),
                "continuation_point":
                    "after_47_rock_120_frame_stable_window_and_240_frame_hold",
                "newton_version": newton.__version__,
                "warp_version": wp.__version__,
                "solver_grid_type": fixture.solver.grid_type,
                "solver_warmstart_mode": fixture.solver._stress_warmstart,
                "collider_velocity_mode": fixture.solver.collider_velocity_mode,
                "array_count": len(arrays47),
                "fields": {name: {"shape": list(value.shape),
                                  "dtype": str(value.dtype)}
                           for name, value in arrays47.items()},
                "particle_count": int(len(arrays47["particle_q"])),
                "rock_count": 47,
                "simulation_time_s": absolute_simulation_time,
                "size_bytes": identity["size_bytes"],
                "data_domain": data_domain,
            }
            _write_json(recovery_metadata_path, recovery_metadata)
            return _finalize_recovery(
                output=output, checkpoint_out=checkpoint_out,
                metadata_out=metadata_out,
                recovery_checkpoint=recovery_checkpoint,
                recovery_metadata_path=recovery_metadata_path,
                recovery_evidence_path=recovery_evidence_path,
                config=config, config_path=config_path,
                source_checkpoint=source_checkpoint,
                source_metadata_path=source_metadata_path,
                data_domain=data_domain)
        finally:
            if release_bridge is not None:
                release_bridge.close(kill=False)
            if source_bridge_open:
                source_bridge.close(kill=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "source_checkpoint": report["source_checkpoint"],
        "particles": report["particles"], "geometry": report["geometry"],
        "release": {key: value for key, value in report["release"].items()
                    if key != "trace"},
        "checkpoint": report["checkpoint"],
        "execution": report["execution"],
        "next_gate": report["next_gate"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "four_layer_checkpoint_saved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
