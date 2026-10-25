#!/usr/bin/env python3
"""Benchmark non-formal solver profiles for local 47-rock teleoperation.

This runner deliberately uses the rejected local diagnostic checkpoint only as
a read-only throughput seed.  Its reports can select a faster solver profile,
but cannot qualify manual collection, training, or a formal A30 result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    checkpoint_identity,
    load_checkpoint_readonly,
    restore_fixture_prefix,
)
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    build_model_and_bridge,
    records_at_qpos,
)
from mpm_rigid_free_velocity import predict_free_linear_velocity
import run_layered_47_excavation_preflight as preflight
import layered_47_interactive_proxy_checkpoint as proxy_checkpoint
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_layered_mound_three_rock_adaptive_press import RuntimeBridge
from run_layered_mound_three_rock_cover_release import _stable_particle
from run_mixed_material_r3_a_probe import _make_transform, _write_json


CONFIG_SCHEMA = "excavation_layered_47_interactive_proxy_benchmark_config_v1"
REPORT_SCHEMA = "excavation_layered_47_interactive_proxy_benchmark_report_v1"
FAILED_STATUS = "layered_47_interactive_proxy_benchmark_failed"
PASSED_STATUS = "layered_47_interactive_proxy_benchmark_passed"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    root = Path(repository_root).resolve()
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "benchmark_nonformal_interactive_solver_profiles_only"):
        raise ValueError("interactive_proxy_benchmark_config_schema_or_status")
    binding = config.get("base_preflight", {})
    base_path = _resolve(root, binding.get("path", ""))
    if (not base_path.is_file()
            or sha256(base_path) != binding.get("sha256")):
        raise ValueError("interactive_proxy_benchmark_base_preflight_binding")
    source = config.get("input_checkpoint", {})
    required_source_fields = {
        "checkpoint_schema", "checkpoint_status", "sha256",
        "source_config_sha256", "source_runner_sha256", "particle_count",
        "rock_count", "newton_version", "warp_version", "read_only",
        "metadata_required",
    }
    if (set(source) != required_source_fields
            or source["checkpoint_status"] !=
            "four_layer_local_robust_hold_rejected_diagnostic"
            or source["read_only"] is not True
            or source["metadata_required"] is not True):
        raise ValueError("interactive_proxy_benchmark_source_contract")
    # Validate the frozen formal scene binding first, then replace only the
    # read-only source identity for this local diagnostic benchmark.  This is
    # the same narrow override used by the previously accepted local speed
    # probe; it does not rewrite the formal config file.
    base = dict(preflight.load_config(base_path, root))
    base["input_checkpoint"] = source
    profiles = config.get("solver_profiles", {})
    expected_profile_names = {
        "formal_5cm_fixed", "voxel_10cm_fixed",
        "voxel_5cm_sparse", "voxel_10cm_sparse",
        "voxel_10cm_sparse_iter25", "voxel_10cm_sparse_iter12",
    }
    if set(profiles) != expected_profile_names:
        raise ValueError("interactive_proxy_benchmark_profile_names")
    formal = preflight.mpm_solver_profile(base)
    expanded_profiles = {}
    for name, patch in profiles.items():
        required = {"voxel_size_m", "grid_type", "grid_padding"}
        allowed = required | {"max_iterations"}
        if not required.issubset(patch) or not set(patch).issubset(allowed):
            raise ValueError("interactive_proxy_benchmark_profile_fields")
        candidate = {
            **formal,
            "voxel_size_m": float(patch["voxel_size_m"]),
            "grid_type": patch["grid_type"],
            "grid_padding": int(patch["grid_padding"]),
            "max_iterations": int(patch.get(
                "max_iterations", formal["max_iterations"])),
        }
        expanded_profiles[name] = preflight.mpm_solver_profile(base, candidate)
    if expanded_profiles["formal_5cm_fixed"] != formal:
        raise ValueError("interactive_proxy_benchmark_formal_control_changed")
    expected_benchmark = {
        "minimum_warmup_frames": 60,
        "maximum_warmup_frames": 240,
        "required_consecutive_stable_frames": 60,
        "measured_frames": 60,
        "require_zero_bucket_contact": True,
        "write_progress_journal": True,
    }
    if config.get("benchmark") != expected_benchmark:
        raise ValueError("interactive_proxy_benchmark_policy")
    expected_scope = {
        "formal_a30_equivalent": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
        "changes_material": False,
        "changes_particle_spacing": False,
        "changes_rock_count": False,
        "changes_bucket_proxy": False,
        "contains_iteration_budget_candidates": True,
        "source_checkpoint_mutated": False,
        "new_formal_checkpoint_written": False,
        "benchmark_recovery_checkpoint_authorized": True,
    }
    if config.get("scope") != expected_scope:
        raise ValueError("interactive_proxy_benchmark_scope")
    return config, {
        "base_path": base_path,
        "base": base,
        "profiles": expanded_profiles,
    }


def _append_journal(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")
        stream.flush()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _advance_frame(fixture, bridge, config: dict, frame: int) -> tuple[dict, float]:
    dt = float(config["probe"]["output_dt_s"])
    gravity = np.asarray(config["mpm"]["gravity_m_s2"], dtype=np.float64)
    started = time.perf_counter()
    prior = bridge.state
    fixture.assign(prior)
    rock_origins = {
        rock["rock_id"]: np.asarray(rock["position"], dtype=np.float64)
        for rock in prior["rocks"]}
    origins = {**rock_origins, "bucket": np.asarray(
        prior["bucket_origin"], dtype=np.float64)}
    transforms = [
        _make_transform(rock["qpos"][:3], rock["qpos"][3:7])
        for rock in prior["rocks"]] + [preflight._bucket_transform(prior)]
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
        transforms, dtype=wp.transform, device=fixture.device))
    fixture.state.body_qd.assign(wp.array(
        velocities, dtype=wp.spatial_vector, device=fixture.device))
    after_assignment = time.perf_counter()
    measured, particles = fixture.step(dt, origins)
    after_mpm = time.perf_counter()
    after = bridge.step(
        dt, int(config["probe"]["mujoco_internal_substeps"]), measured)
    after_mujoco = time.perf_counter()
    wall_seconds = after_mujoco - started
    linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                 for rock in after["rocks"])
    angular = max(float(np.linalg.norm(rock["qvel"][3:]))
                  for rock in after["rocks"])
    position_step = max(float(np.linalg.norm(
        np.asarray(current["qpos"][:3], dtype=np.float64)
        - np.asarray(previous["qpos"][:3], dtype=np.float64)))
        for previous, current in zip(
            prior["rocks"], after["rocks"], strict=True))
    contacts = after.get("contact_substeps", {})
    stable = bool(
        linear <= float(config["probe"][
            "maximum_stable_rock_linear_speed_m_s"])
        and angular <= float(config["probe"][
            "maximum_stable_rock_angular_speed_rad_s"])
        and _stable_particle(particles, config["probe"]))
    row = {
        "frame": frame,
        "wall_seconds": wall_seconds,
        "assignment_wall_seconds": after_assignment - started,
        "mpm_wall_seconds": after_mpm - after_assignment,
        "mujoco_wall_seconds": after_mujoco - after_mpm,
        "finite": bool(particles["finite"] and _finite_tree(after)),
        "stable": stable,
        "maximum_rock_linear_speed_m_s": linear,
        "maximum_rock_angular_speed_rad_s": angular,
        "maximum_rock_position_step_m": position_step,
        "bucket_mpm_contact_samples": int(
            particles["contact_samples_by_body"]["bucket"]),
        "rock_bucket_contact_substeps": int(contacts.get("rock_bucket", 0)),
        "external_rigid_support_substeps": sum(int(
            contacts.get(name, 0)) for name in (
                "rock_ground", "rock_ramp", "rock_truck")),
        "forbidden_contact_count": int(
            after.get("forbidden_contact_count", 0)),
        "closure_m": float(after.get("closure_m", float("inf"))),
        **particles,
    }
    return row, wall_seconds


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = Path(args.checkpoint_metadata).resolve()
    output = Path(args.output).resolve()
    journal = output.with_suffix(output.suffix + ".progress.ndjson")
    recovery_checkpoint = (
        Path(args.recovery_checkpoint).resolve()
        if args.recovery_checkpoint else None)
    recovery_metadata = (
        Path(args.recovery_metadata).resolve()
        if args.recovery_metadata else None)
    if (recovery_checkpoint is None) != (recovery_metadata is None):
        raise ValueError("interactive_proxy_benchmark_recovery_pair")
    if recovery_checkpoint is not None and args.profile != "voxel_10cm_sparse":
        raise ValueError("interactive_proxy_benchmark_recovery_profile")
    existing = [path for path in (
        output, journal, recovery_checkpoint, recovery_metadata)
        if path is not None and path.exists()]
    if existing:
        raise FileExistsError(existing[0])
    if output.exists() or journal.exists():
        raise FileExistsError(output if output.exists() else journal)
    config, loaded = load_config(config_path, root)
    if args.profile not in loaded["profiles"]:
        raise ValueError("interactive_proxy_benchmark_unknown_profile")
    solver_profile = loaded["profiles"][args.profile]
    base = loaded["base"]
    arrays, source_metadata, before = load_checkpoint_readonly(
        checkpoint, metadata_path, config["input_checkpoint"])
    manifest_path = _resolve(
        root, base["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in preflight.LAYERS]
    if ([rock["rock_id"] for rock in records]
            != list(preflight.DEFAULT_PROFILE["expected_ids"])):
        raise ValueError("interactive_proxy_benchmark_manifest_order")
    _, operator = preflight._load_bound_json(
        root, base["operator_start"], "operator_start")
    proxy_path, proxy = preflight._load_bound_json(
        root, base["open_bucket_proxy"], "open_bucket_proxy")
    output.parent.mkdir(parents=True, exist_ok=True)
    if config["benchmark"]["write_progress_journal"]:
        _append_journal(journal, {
            "event": "started", "profile": args.profile,
            "source_checkpoint": before, "solver_profile": solver_profile,
        })

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    bridge = None
    warmup_rows: list[dict] = []
    measured_rows: list[dict] = []
    restore = None
    model_manifest = None
    bridge_manifest = None
    compile_command = None
    recovery_arrays = None
    with tempfile.TemporaryDirectory(
            prefix=f"layered-47-interactive-{args.profile}-") as raw:
        work = Path(raw)
        restored_records = records_at_qpos(records, arrays["rigid_qpos"])
        model, binary, model_manifest, bridge_manifest, compile_command = (
            build_model_and_bridge(
                args, root, root / base["capacity_bridge"]["config_path"],
                restored_records, work, f"interactive-{args.profile}"))
        default_bridge = RuntimeBridge(binary, model, len(records))
        default_bridge.close()
        bridge = RestoredRuntimeBridge(
            binary, model, len(records),
            [float(value) for value in operator["qpos"]], [0.0] * 18,
            arrays["rigid_qpos"], arrays["rigid_qvel"])
        fixture = preflight.ExcavationFixture(
            base, root, records, arrays["grid_reference_particle_q"],
            proxy, bridge.state, args.device, solver_profile=solver_profile)
        restore = restore_fixture_prefix(fixture, arrays)
        fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        fixture.assign(bridge.state)

        strict_tail = 0
        minimum = int(config["benchmark"]["minimum_warmup_frames"])
        maximum = int(config["benchmark"]["maximum_warmup_frames"])
        required = int(config["benchmark"][
            "required_consecutive_stable_frames"])
        for frame in range(maximum):
            row, _ = _advance_frame(fixture, bridge, base, frame)
            warmup_rows.append(row)
            safe = bool(
                row["finite"] and row["stable"]
                and row["forbidden_contact_count"] == 0
                and row["external_rigid_support_substeps"] == 0
                and row["unmapped_active_collider_sample_count"] == 0
                and row["nonfinite_collider_sample_count"] == 0
                and row["bucket_mpm_contact_samples"] == 0
                and row["rock_bucket_contact_substeps"] == 0)
            strict_tail = strict_tail + 1 if safe else 0
            if config["benchmark"]["write_progress_journal"]:
                _append_journal(journal, {
                    "event": "warmup_frame", "profile": args.profile,
                    "row": row, "strict_tail": strict_tail,
                })
            print(
                "PROGRESS_47_INTERACTIVE_BENCHMARK "
                f"profile={args.profile} phase=warmup frame={frame + 1}/{maximum} "
                f"stable={row['stable']} strict_tail={strict_tail} "
                f"wall={row['wall_seconds']:.6f}", flush=True)
            if not row["finite"]:
                break
            if frame + 1 >= minimum and strict_tail >= required:
                break

        stabilized = bool(
            len(warmup_rows) >= minimum and strict_tail >= required)
        if stabilized:
            requested = int(config["benchmark"]["measured_frames"])
            for index in range(requested):
                row, _ = _advance_frame(
                    fixture, bridge, base, len(warmup_rows) + index)
                measured_rows.append(row)
                if config["benchmark"]["write_progress_journal"]:
                    _append_journal(journal, {
                        "event": "measured_frame", "profile": args.profile,
                        "row": row,
                    })
                print(
                    "PROGRESS_47_INTERACTIVE_BENCHMARK "
                    f"profile={args.profile} phase=measured "
                    f"frame={index + 1}/{requested} stable={row['stable']} "
                    f"wall={row['wall_seconds']:.6f}", flush=True)
                if not row["finite"]:
                    break
        recovery_candidate_passed = bool(
            stabilized and len(measured_rows) == requested
            and all(row["finite"] and row["stable"]
                    for row in measured_rows)
            and all(row["forbidden_contact_count"] == 0
                    and row["external_rigid_support_substeps"] == 0
                    and row["unmapped_active_collider_sample_count"] == 0
                    and row["nonfinite_collider_sample_count"] == 0
                    and row["bucket_mpm_contact_samples"] == 0
                    and row["rock_bucket_contact_substeps"] == 0
                    and row["closure_m"] <= 0.001
                    for row in measured_rows))
        if recovery_checkpoint is not None and recovery_candidate_passed:
            recovery_arrays = proxy_checkpoint.capture_arrays(
                fixture, bridge.state,
                float(arrays["simulation_time_s"][0])
                + (len(warmup_rows) + len(measured_rows))
                * float(base["probe"]["output_dt_s"]))
        bridge.close(kill=not measured_rows or not measured_rows[-1]["finite"])
        bridge = None

    after = checkpoint_identity(checkpoint)
    requested = int(config["benchmark"]["measured_frames"])
    require_zero_contact = config["benchmark"]["require_zero_bucket_contact"]
    checks = {
        "source_checkpoint_sha256_verified": (
            before["sha256"] == config["input_checkpoint"]["sha256"]),
        "source_checkpoint_metadata_verified": (
            source_metadata["checkpoint_sha256"] == before["sha256"]),
        "source_checkpoint_unchanged": after == before,
        "particle_fields_restored_exactly": all(
            restore["field_readback_exact"].values()),
        "warmup_reached_stable_tail": stabilized,
        "all_measured_frames_executed": len(measured_rows) == requested,
        "all_measured_states_finite": bool(measured_rows) and all(
            row["finite"] for row in measured_rows),
        "all_measured_frames_stable": bool(measured_rows) and all(
            row["stable"] for row in measured_rows),
        "zero_bucket_contact_during_benchmark": (
            not require_zero_contact or bool(measured_rows) and all(
                row["bucket_mpm_contact_samples"] == 0
                and row["rock_bucket_contact_substeps"] == 0
                for row in measured_rows)),
        "forbidden_contacts_zero": bool(measured_rows) and all(
            row["forbidden_contact_count"] == 0 for row in measured_rows),
        "no_external_rigid_support": bool(measured_rows) and all(
            row["external_rigid_support_substeps"] == 0
            for row in measured_rows),
        "no_unmapped_or_nonfinite_collider_samples": bool(measured_rows) and all(
            row["unmapped_active_collider_sample_count"] == 0
            and row["nonfinite_collider_sample_count"] == 0
            for row in measured_rows),
        "mechanism_closure_within_one_mm": bool(measured_rows) and all(
            row["closure_m"] <= 0.001 for row in measured_rows),
        "no_formal_checkpoint_written": True,
        "manual_collection_not_granted": (
            config["scope"]["manual_collection_eligible"] is False),
        "training_not_granted": (
            config["scope"]["training_eligible"] is False),
    }
    timings = [row["wall_seconds"] for row in measured_rows]
    assignment_timings = [
        row["assignment_wall_seconds"] for row in measured_rows]
    mpm_timings = [row["mpm_wall_seconds"] for row in measured_rows]
    mujoco_timings = [row["mujoco_wall_seconds"] for row in measured_rows]
    mean = float(np.mean(timings)) if timings else None
    output_dt = float(base["probe"]["output_dt_s"])
    timing = {
        "measured_frames": len(timings),
        "mean_wall_seconds_per_frame": mean,
        "median_wall_seconds_per_frame": _percentile(timings, 0.5),
        "p95_wall_seconds_per_frame": _percentile(timings, 0.95),
        "minimum_wall_seconds_per_frame": min(timings) if timings else None,
        "maximum_wall_seconds_per_frame": max(timings) if timings else None,
        "physics_frames_per_wall_second": (1.0 / mean if mean else None),
        "simulated_seconds_per_wall_second": (
            output_dt / mean if mean else None),
        "output_dt_s": output_dt,
        "mean_assignment_wall_seconds": (
            float(np.mean(assignment_timings)) if assignment_timings else None),
        "mean_mpm_wall_seconds": (
            float(np.mean(mpm_timings)) if mpm_timings else None),
        "mean_mujoco_wall_seconds": (
            float(np.mean(mujoco_timings)) if mujoco_timings else None),
    }
    passed = all(checks.values())
    recovery = {
        "requested": recovery_checkpoint is not None,
        "written": False,
        "checkpoint": None,
        "metadata": None,
        "manual_collection_eligible": False,
        "training_eligible": False,
    }
    if passed and recovery_checkpoint is not None:
        if recovery_arrays is None:
            raise RuntimeError("interactive_proxy_benchmark_recovery_missing")
        recovery_identity = proxy_checkpoint.write_checkpoint_atomic(
            recovery_checkpoint, recovery_arrays)
        recovery_document = {
            "schema": proxy_checkpoint.CHECKPOINT_SCHEMA,
            "status": proxy_checkpoint.CHECKPOINT_STATUS,
            "checkpoint_file": recovery_checkpoint.name,
            "checkpoint_sha256": recovery_identity["sha256"],
            "particle_count": len(recovery_arrays["particle_q"]),
            "rock_count": len(recovery_arrays["rigid_qpos"]),
            "simulation_time_s": float(
                recovery_arrays["simulation_time_s"][0]),
            "solver_grid_type": solver_profile["grid_type"],
            "solver_voxel_size_m": solver_profile["voxel_size_m"],
            "solver_grid_padding": solver_profile["grid_padding"],
            "solver_max_iterations": solver_profile["max_iterations"],
            "solver_tolerance": solver_profile["tolerance"],
            "collider_velocity_mode": solver_profile[
                "collider_velocity_mode"],
            "source_checkpoint": before,
            "source_checkpoint_status": config["input_checkpoint"][
                "checkpoint_status"],
            "source_benchmark_report": str(output),
            "source_benchmark_config_sha256": sha256(config_path),
            "source_runner_sha256": sha256(Path(__file__).resolve()),
            "manual_collection_eligible": False,
            "training_eligible": False,
            "formal_a30_equivalent": False,
            "scope_note": (
                "Recovery for independent resume audit only; ancestry from the "
                "rejected local diagnostic checkpoint remains explicit."),
        }
        _write_json(recovery_metadata, recovery_document)
        _, verified_metadata, verified_identity = (
            proxy_checkpoint.load_checkpoint_readonly(
                recovery_checkpoint, recovery_metadata))
        recovery = {
            "requested": True,
            "written": True,
            "checkpoint": verified_identity,
            "metadata": {
                "path": str(recovery_metadata),
                "sha256": sha256(recovery_metadata),
                "status": verified_metadata["status"],
            },
            "manual_collection_eligible": False,
            "training_eligible": False,
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": PASSED_STATUS if passed else FAILED_STATUS,
        "profile": args.profile,
        "solver_profile": solver_profile,
        "checks": checks,
        "timing": timing,
        "warmup": {
            "executed_frames": len(warmup_rows),
            "stabilized": stabilized,
            "trace": warmup_rows,
        },
        "measurement": {
            "requested_frames": requested,
            "executed_frames": len(measured_rows),
            "trace": measured_rows,
        },
        "source_checkpoint": before,
        "source_checkpoint_after": after,
        "source_metadata": source_metadata,
        "restore": restore,
        "inputs": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "base_preflight_path": str(loaded["base_path"]),
            "base_preflight_sha256": sha256(loaded["base_path"]),
            "manifest_sha256": sha256(manifest_path),
            "open_bucket_proxy_sha256": sha256(proxy_path),
        },
        "runtime_build": {
            "model_manifest": model_manifest,
            "bridge_manifest": bridge_manifest,
            "compile_command": compile_command,
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "warmup_step_calls": len(warmup_rows),
            "measured_step_calls": len(measured_rows),
            "progress_journal": str(journal),
        },
        "scope": config["scope"],
        "recovery": recovery,
        "next_gate": (
            "independent_interactive_proxy_resume_audit"
            if passed and recovery["written"] else
            "compare_profile_evidence_before_building_interactive_proxy"
            if passed else "reject_profile_and_preserve_failure"),
    }
    if not _finite_tree(report):
        raise RuntimeError("interactive_proxy_benchmark_nonfinite_report")
    _write_json(output, report)
    if config["benchmark"]["write_progress_journal"]:
        _append_journal(journal, {
            "event": "completed", "profile": args.profile,
            "status": report["status"], "timing": timing,
        })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    parser.add_argument("--recovery-checkpoint")
    parser.add_argument("--recovery-metadata")
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    try:
        report = run(args)
    except (Exception, KeyboardInterrupt) as exc:
        if not output.exists():
            failure = {
                "schema": REPORT_SCHEMA,
                "status": FAILED_STATUS,
                "profile": args.profile,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "inputs": {
                    "config": str(Path(args.config).resolve()),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_metadata": str(
                        Path(args.checkpoint_metadata).resolve()),
                },
                "scope": {
                    "formal_a30_equivalent": False,
                    "manual_collection_eligible": False,
                    "training_eligible": False,
                    "new_checkpoint_written": False,
                },
                "next_gate": "fix_or_reject_profile_without_losing_failure",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, failure)
        print(json.dumps({
            "status": FAILED_STATUS,
            "profile": args.profile,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "report": str(output),
        }, sort_keys=True, allow_nan=False), flush=True)
        return 1
    print(json.dumps({
        "status": report["status"], "profile": report["profile"],
        "checks": report["checks"], "timing": report["timing"],
        "scope": report["scope"], "next_gate": report["next_gate"],
        "report": str(output),
    }, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["status"] == PASSED_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
