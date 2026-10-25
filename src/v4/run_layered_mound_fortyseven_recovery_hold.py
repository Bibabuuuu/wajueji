#!/usr/bin/env python3
"""Resume a saved 47-rock recovery, then admit it only after a strict audit.

The original failed report is immutable evidence.  This runner treats its
post-physics recovery checkpoint as a read-only parent, applies the bounded
sparse-transient rule already accepted for the A30 65-rock continuation, and
writes a restartable candidate before performing a fresh-runtime strict hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    CHECKPOINT_SCHEMA,
    checkpoint_arrays,
    checkpoint_identity,
    load_checkpoint_readonly,
    write_checkpoint_atomic,
)
from layered_mound_checkpoint_runtime import dynamic_step, restore_runtime
from run_layered_mound_fortyseven_rock_four_layer import (
    NEW_LAYER,
    RECOVERY_SCHEMA,
    RECOVERY_STATUS,
    SOURCE_LAYERS,
    _final_checks,
    load_config as load_physical_config,
)
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_fortyseven_recovery_hold_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_fortyseven_recovery_hold_report_v1"
CANDIDATE_STATUS = "four_layer_local_robust_hold_candidate_not_admitted"
FINAL_STATUS = "four_layer_local_robust_continuation_checkpoint_saved"
PARENT_ALLOWED_FAILED_CHECK = "additional_uncontrolled_hold_all_frames_stable"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _candidate_metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".candidate.json")


def _final_metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".json")


def _stage_report_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".staging.json")


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    root = Path(repository_root).resolve()
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "resume_local_47_recovery_then_fresh_runtime_strict_audit"):
        raise ValueError("fortyseven_recovery_hold_config_schema_or_status")

    physical_binding = config.get("physical_config", {})
    physical_path = _resolve(root, physical_binding.get("path", ""))
    if (not physical_path.is_file()
            or sha256(physical_path) != physical_binding.get("sha256")):
        raise ValueError("fortyseven_recovery_hold_physical_config_binding")
    raw_physical = json.loads(physical_path.read_text(encoding="utf-8"))
    physical = load_physical_config(
        physical_path, root,
        expected_input_checkpoint=raw_physical.get("input_checkpoint"))

    recovery = config.get("recovery", {})
    required_recovery = {
        "checkpoint", "checkpoint_sha256", "metadata", "metadata_sha256",
        "evidence", "evidence_sha256", "data_domain",
    }
    if set(recovery) != required_recovery:
        raise ValueError("fortyseven_recovery_hold_recovery_contract_fields")
    for name, hash_name in (
            ("checkpoint", "checkpoint_sha256"),
            ("metadata", "metadata_sha256"),
            ("evidence", "evidence_sha256")):
        target = _resolve(root, recovery[name])
        if not target.is_file() or sha256(target) != recovery[hash_name]:
            raise ValueError(f"fortyseven_recovery_hold_{name}_binding")

    if config.get("robust_hold") != {
            "minimum_total_frames": 240,
            "maximum_total_frames": 360,
            "required_strict_recovery_frames": 60,
            "strict_particle_max_speed_m_s": 0.05,
            "transient_particle_max_speed_ceiling_m_s": 0.10,
            "maximum_transient_frames": 6,
            "maximum_consecutive_transient_frames": 2}:
        raise ValueError("fortyseven_recovery_hold_policy")
    if config.get("strict_restore_audit") != {
            "frames": 120,
            "fresh_runtime": True,
            "all_frames_must_be_strict": True,
            "candidate_saved_before_audit": True}:
        raise ValueError("fortyseven_recovery_hold_audit_policy")
    if config.get("scope") != {
            "data_domain": recovery["data_domain"],
            "formal_a30_equivalent": False,
            "source_recovery_mutated": False,
            "rock_count": 47,
            "full_scene_claimed": False,
            "training": False}:
        raise ValueError("fortyseven_recovery_hold_scope")
    return config, physical


def classify_hold_frame(row: dict, probe: dict, policy: dict) -> dict:
    reasons = []
    checks = {
        "finite": bool(row.get("finite")),
        "centers_inside": bool(row.get("centers_inside")),
        "no_external_support": row.get("external_rigid_support_substeps") == 0,
        "no_forbidden_contact": row.get("forbidden_contact_count") == 0,
        "momentum_accounted_once": (
            float(row.get("global_momentum_residual_ns", float("inf"))) < 1e-4),
        "rock_linear_strict": (
            float(row.get("maximum_rock_linear_speed_m_s", float("inf"))) <=
            float(probe["maximum_stable_rock_linear_speed_m_s"])),
        "rock_angular_strict": (
            float(row.get("maximum_rock_angular_speed_rad_s", float("inf"))) <=
            float(probe["maximum_stable_rock_angular_speed_rad_s"])),
        "particle_p99_strict": (
            float(row.get("particle_p99_speed_m_s", float("inf"))) <=
            float(probe["maximum_stable_particle_p99_speed_m_s"])),
        "particle_above_world_floor": (
            float(row.get("particle_minimum_z_m", float("-inf"))) >=
            float(probe["minimum_world_z_m"])),
        "particle_below_transient_ceiling": (
            float(row.get("particle_max_speed_m_s", float("inf"))) <=
            float(policy["transient_particle_max_speed_ceiling_m_s"])),
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    if reasons:
        classification = "hard_failure"
    elif (float(row["particle_max_speed_m_s"]) >
          float(policy["strict_particle_max_speed_m_s"])):
        classification = "bounded_sparse_transient"
    else:
        classification = "strict"
    return {"classification": classification, "checks": checks,
            "failure_reasons": reasons}


def analyze_trace(trace: list[dict], probe: dict, policy: dict) -> dict:
    transient = consecutive = maximum_consecutive = strict_tail = 0
    hard_failures = []
    classified = []
    for index, row in enumerate(trace):
        result = classify_hold_frame(row, probe, policy)
        kind = result["classification"]
        classified.append({
            "index": index,
            "frame": row.get("frame"),
            "classification": kind,
            "particle_max_speed_m_s": row.get("particle_max_speed_m_s"),
            "particle_p99_speed_m_s": row.get("particle_p99_speed_m_s"),
            "failure_reasons": result["failure_reasons"],
        })
        if kind == "hard_failure":
            hard_failures.append(classified[-1])
            consecutive = 0
            strict_tail = 0
        elif kind == "bounded_sparse_transient":
            transient += 1
            consecutive += 1
            maximum_consecutive = max(maximum_consecutive, consecutive)
            strict_tail = 0
        else:
            consecutive = 0
            strict_tail += 1
    return {
        "total_frames": len(trace),
        "strict_frames": sum(
            row["classification"] == "strict" for row in classified),
        "transient_frames": transient,
        "maximum_consecutive_transient_frames": maximum_consecutive,
        "trailing_consecutive_transient_frames": consecutive,
        "trailing_strict_recovery_frames": strict_tail,
        "hard_failures": hard_failures,
        "classified_trace": classified,
        "within_transient_budget": (
            transient <= int(policy["maximum_transient_frames"])
            and maximum_consecutive <= int(
                policy["maximum_consecutive_transient_frames"])),
    }


def _load_parent(config: dict, physical: dict, root: Path) -> tuple[
        dict[str, np.ndarray], dict, dict, dict, dict]:
    recovery = config["recovery"]
    checkpoint = _resolve(root, recovery["checkpoint"])
    metadata_path = _resolve(root, recovery["metadata"])
    evidence_path = _resolve(root, recovery["evidence"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (metadata.get("schema") != RECOVERY_SCHEMA
            or metadata.get("status") != RECOVERY_STATUS
            or metadata.get("recovery_checkpoint_sha256") !=
                recovery["checkpoint_sha256"]
            or metadata.get("recovery_evidence_sha256") !=
                recovery["evidence_sha256"]
            or metadata.get("source_config_sha256") !=
                config["physical_config"]["sha256"]
            or metadata.get("data_domain") != recovery["data_domain"]
            or metadata.get("particle_count") != 256441
            or metadata.get("rock_count") != 47):
        raise ValueError("fortyseven_recovery_hold_parent_metadata")
    if (evidence.get("status") != RECOVERY_STATUS
            or evidence.get("additional_uncontrolled_hold", {}).get(
                "executed_frames") != 143):
        raise ValueError("fortyseven_recovery_hold_parent_evidence")

    source = Path(metadata["source_checkpoint"]).resolve()
    source_after = checkpoint_identity(source)
    parent_checks = _final_checks(evidence, physical, source_after)
    failed = {name for name, passed in parent_checks.items() if not passed}
    if failed != {PARENT_ALLOWED_FAILED_CHECK}:
        raise ValueError(
            "fortyseven_recovery_hold_unexpected_parent_failures:" +
            ",".join(sorted(failed)))

    synthetic = {
        "schema": CHECKPOINT_SCHEMA,
        "status": RECOVERY_STATUS,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": recovery["checkpoint_sha256"],
        "source_config_sha256": metadata["source_config_sha256"],
        "source_runner_sha256": metadata["physics_runner_sha256"],
        "particle_count": metadata["particle_count"],
        "rock_count": metadata["rock_count"],
        "gravity_predictor_required_on_next_free_step": True,
        "same_solver_material_and_dt_required": True,
        "fixed_grid_reference_saved": True,
        "newton_solver_warmstart_saved": True,
        "newton_version": metadata["newton_version"],
        "warp_version": metadata["warp_version"],
        "solver_grid_type": metadata["solver_grid_type"],
        "solver_warmstart_mode": metadata["solver_warmstart_mode"],
        "collider_velocity_mode": metadata["collider_velocity_mode"],
        "simulation_time_s": metadata["simulation_time_s"],
    }
    contract = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_status": RECOVERY_STATUS,
        "sha256": recovery["checkpoint_sha256"],
        "source_config_sha256": metadata["source_config_sha256"],
        "source_runner_sha256": metadata["physics_runner_sha256"],
        "particle_count": metadata["particle_count"],
        "rock_count": metadata["rock_count"],
        "newton_version": metadata["newton_version"],
        "warp_version": metadata["warp_version"],
    }
    with tempfile.TemporaryDirectory(prefix="local-47-recovery-metadata-") as raw:
        synthetic_path = Path(raw) / "metadata.json"
        synthetic_path.write_text(
            json.dumps(synthetic, sort_keys=True), encoding="utf-8")
        arrays, _, identity = load_checkpoint_readonly(
            checkpoint, synthetic_path, contract)
    return arrays, metadata, evidence, identity, parent_checks


def _records(physical: dict, root: Path) -> list[dict]:
    manifest_path = _resolve(
        root, physical["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in (*SOURCE_LAYERS, NEW_LAYER)]
    expected_ids = (
        [f"lower-{index:03d}" for index in range(9)]
        + [f"lower_middle-{index:03d}" for index in range(11)]
        + [f"middle-{index:03d}" for index in range(13)]
        + [f"upper-{index:03d}" for index in range(14)])
    if len(records) != 47 or [row["rock_id"] for row in records] != expected_ids:
        raise ValueError("fortyseven_recovery_hold_manifest_order")
    return records


def _outer_bounds(physical: dict, root: Path) -> np.ndarray:
    visual_path = _resolve(root, physical["source"]["visual_config_path"])
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    return np.asarray(
        visual["mound_envelope"]["outer_xy_bounds_m"], dtype=np.float64)


def _step(fixture, bridge, records: list[dict], physical: dict,
          outer: np.ndarray, frame: int) -> dict:
    after, row = dynamic_step(fixture, bridge, records, physical, frame)
    row["centers_inside"] = bool(all(
        outer[0, 0] <= rock["position"][0] <= outer[1, 0]
        and outer[0, 1] <= rock["position"][1] <= outer[1, 1]
        and rock["position"][2] >= physical["probe"]["minimum_world_z_m"]
        for rock in after["rocks"]))
    return row


def _checkpoint_metadata(
        *, status: str, checkpoint: Path, identity: dict, arrays: dict,
        config_path: Path, runner_path: Path, recovery_metadata: dict,
        simulation_time_s: float, data_domain: str) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": status,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": identity["sha256"],
        "source_checkpoint_sha256": recovery_metadata[
            "recovery_checkpoint_sha256"],
        "source_config_sha256": sha256(config_path),
        "source_runner_sha256": sha256(runner_path),
        "continuation_point": (
            "after_local_47_robust_hold_and_before_or_after_strict_restore_audit"),
        "gravity_predictor_required_on_next_free_step": True,
        "same_solver_material_and_dt_required": True,
        "fixed_grid_reference_saved": True,
        "newton_solver_warmstart_saved": True,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "solver_grid_type": recovery_metadata["solver_grid_type"],
        "solver_warmstart_mode": recovery_metadata["solver_warmstart_mode"],
        "collider_velocity_mode": recovery_metadata["collider_velocity_mode"],
        "array_count": len(arrays),
        "fields": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
        "particle_count": int(len(arrays["particle_q"])),
        "rock_count": int(len(arrays["rigid_qpos"])),
        "simulation_time_s": simulation_time_s,
        "data_domain": data_domain,
        "formal_a30_equivalent": False,
        "diagnostic_only": status == CANDIDATE_STATUS,
        "training_eligible": False,
        "size_bytes": identity["size_bytes"],
    }


def _contract_from_metadata(metadata: dict) -> dict:
    return {
        "checkpoint_schema": metadata["schema"],
        "checkpoint_status": metadata["status"],
        "sha256": metadata["checkpoint_sha256"],
        "source_config_sha256": metadata["source_config_sha256"],
        "source_runner_sha256": metadata["source_runner_sha256"],
        "particle_count": metadata["particle_count"],
        "rock_count": metadata["rock_count"],
        "newton_version": metadata["newton_version"],
        "warp_version": metadata["warp_version"],
    }


def _particle_hotspots(fixture, limit: int = 10) -> list[dict]:
    q = fixture.state.particle_q.numpy()
    qd = fixture.state.particle_qd.numpy()
    speed = np.linalg.norm(qd, axis=1)
    indices = np.argsort(speed)[-limit:][::-1]
    return [{
        "particle_index": int(index),
        "speed_m_s": float(speed[index]),
        "position_world_m": q[index].astype(float).tolist(),
        "velocity_world_m_s": qd[index].astype(float).tolist(),
    } for index in indices]


def _rejected_paths(checkpoint: Path) -> tuple[Path, Path]:
    rejected = checkpoint.with_name(
        checkpoint.stem + ".rejected" + checkpoint.suffix)
    return rejected, rejected.with_suffix(rejected.suffix + ".json")


def _promote(
        *, stage: dict, output: Path, checkpoint: Path,
        candidate_metadata_path: Path, final_metadata_path: Path) -> dict:
    candidate = json.loads(
        candidate_metadata_path.read_text(encoding="utf-8"))
    if stage.get("status") != "local_47_restore_audit_passed_pending_promotion":
        raise ValueError("fortyseven_recovery_hold_not_ready_for_promotion")
    final_metadata = dict(candidate)
    final_metadata.update({
        "status": FINAL_STATUS,
        "diagnostic_only": False,
        "strict_restore_audit": stage["strict_restore_audit"],
    })
    _write_json(final_metadata_path, final_metadata)
    report = dict(stage)
    report.update({
        "status": "local_47_robust_recovery_checkpoint_saved",
        "checkpoint": {
            **final_metadata,
            "metadata_file": final_metadata_path.name,
            "metadata_sha256": sha256(final_metadata_path),
        },
        "next_gate": "local_47_interactive_precontact_speed_benchmark",
    })
    report["checks"]["final_metadata_written"] = True
    if not _finite_tree(report):
        raise RuntimeError("fortyseven_recovery_hold_nonfinite_final_report")
    _write_json(output, report)
    return report


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.output_checkpoint).resolve()
    candidate_metadata_path = _candidate_metadata_path(checkpoint)
    final_metadata_path = _final_metadata_path(checkpoint)
    stage_path = _stage_report_path(output)
    runner_path = Path(__file__).resolve()
    config, physical = load_config(config_path, root)

    if final_metadata_path.is_file():
        if not output.is_file():
            raise RuntimeError("fortyseven_recovery_hold_final_metadata_without_report")
        return json.loads(output.read_text(encoding="utf-8"))
    if checkpoint.exists() != candidate_metadata_path.exists():
        raise RuntimeError("fortyseven_recovery_hold_incomplete_candidate")

    arrays, recovery_metadata, parent_evidence, parent_identity, parent_checks = (
        _load_parent(config, physical, root))
    parent_analysis = analyze_trace(
        parent_evidence["additional_uncontrolled_hold"]["trace"],
        physical["probe"], config["robust_hold"])
    if (parent_analysis["hard_failures"]
            or not parent_analysis["within_transient_budget"]):
        raise ValueError("fortyseven_recovery_hold_parent_not_eligible")
    records = _records(physical, root)
    outer = _outer_bounds(physical, root)
    policy = config["robust_hold"]
    audit_policy = config["strict_restore_audit"]

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)

    with tempfile.TemporaryDirectory(prefix="local-47-recovery-hold-") as raw:
        work = Path(raw)
        if not checkpoint.exists():
            fixture, bridge, restore = restore_runtime(
                args, physical, root, records, arrays, work, "resume-47")
            resume_trace = []
            combined = dict(parent_analysis)
            rejection = None
            try:
                while (combined["total_frames"] <
                       int(policy["minimum_total_frames"])
                       or combined["trailing_strict_recovery_frames"] <
                       int(policy["required_strict_recovery_frames"])):
                    if combined["total_frames"] >= int(
                            policy["maximum_total_frames"]):
                        raise RuntimeError("fortyseven_recovery_hold_maximum_frames")
                    frame = int(parent_evidence["additional_uncontrolled_hold"][
                        "trace"][-1]["frame"]) + len(resume_trace) + 1
                    row = _step(
                        fixture, bridge, records, physical, outer, frame)
                    resume_trace.append(row)
                    combined = analyze_trace(
                        parent_evidence["additional_uncontrolled_hold"]["trace"]
                        + resume_trace, physical["probe"], policy)
                    last = combined["classified_trace"][-1]
                    if rejection is None and (
                            last["classification"] == "hard_failure"
                            or not combined["within_transient_budget"]):
                        rejection = {
                            "first_rejected_combined_frame_count":
                                combined["total_frames"],
                            "first_rejected_frame": last["frame"],
                            "classification": last["classification"],
                            "transient_frames": combined["transient_frames"],
                            "maximum_consecutive_transient_frames": combined[
                                "maximum_consecutive_transient_frames"],
                            "particle_hotspots": _particle_hotspots(fixture),
                        }
                    if (rejection is not None
                            and last["classification"] == "hard_failure"):
                        break
                    if len(resume_trace) == 1 or len(resume_trace) % 30 == 0:
                        print(
                            "PROGRESS_LOCAL_47_RECOVERY "
                            f"combined={combined['total_frames']}/"
                            f"{policy['minimum_total_frames']} "
                            f"strict_recovery="
                            f"{combined['trailing_strict_recovery_frames']}/"
                            f"{policy['required_strict_recovery_frames']} "
                            f"transients={combined['transient_frames']}/"
                            f"{policy['maximum_transient_frames']} "
                            f"particle_max={row['particle_max_speed_m_s']:.9g}",
                            flush=True)
                simulation_time = (
                    float(arrays["simulation_time_s"][0])
                    + len(resume_trace) * float(physical["probe"]["output_dt_s"]))
                if rejection is not None:
                    rejected_checkpoint, rejected_metadata_path = (
                        _rejected_paths(checkpoint))
                    if rejected_checkpoint.exists() or rejected_metadata_path.exists():
                        raise FileExistsError(
                            rejected_checkpoint if rejected_checkpoint.exists()
                            else rejected_metadata_path)
                    rejected_arrays = checkpoint_arrays(
                        fixture, bridge.state, simulation_time)
                    rejected_identity = write_checkpoint_atomic(
                        rejected_checkpoint, rejected_arrays)
                    rejected_metadata = _checkpoint_metadata(
                        status="four_layer_local_robust_hold_rejected_diagnostic",
                        checkpoint=rejected_checkpoint,
                        identity=rejected_identity, arrays=rejected_arrays,
                        config_path=config_path, runner_path=runner_path,
                        recovery_metadata=recovery_metadata,
                        simulation_time_s=simulation_time,
                        data_domain=config["recovery"]["data_domain"])
                    rejected_metadata.update({
                        "diagnostic_only": True,
                        "training_eligible": False,
                        "rejection": rejection,
                    })
                    _write_json(rejected_metadata_path, rejected_metadata)
                    failure = {
                        "schema": REPORT_SCHEMA,
                        "status": (
                            "local_47_robust_recovery_hold_rejected_"
                            "diagnostic_saved"),
                        "checks": {
                            "parent_only_failed_original_strict_hold": True,
                            "source_recovery_sha256_verified": True,
                            "source_recovery_unchanged": (
                                checkpoint_identity(_resolve(
                                    root, config["recovery"]["checkpoint"])) ==
                                parent_identity),
                            "parent_sparse_transient_within_budget": True,
                            "combined_robust_hold_passed": False,
                            "rejected_diagnostic_checkpoint_saved": True,
                            "final_metadata_written": False,
                        },
                        "source_recovery_checkpoint": parent_identity,
                        "parent_checks": parent_checks,
                        "parent_analysis": parent_analysis,
                        "resume": {
                            "executed_frames": len(resume_trace),
                            "trace": resume_trace,
                            "restore": restore,
                        },
                        "combined_analysis": combined,
                        "rejection": rejection,
                        "rejected_checkpoint": {
                            **rejected_metadata,
                            "metadata_file": rejected_metadata_path.name,
                            "metadata_sha256": sha256(rejected_metadata_path),
                        },
                        "checkpoint": None,
                        "environment": {
                            "python": platform.python_version(),
                            "newton": newton.__version__,
                            "warp": wp.__version__,
                            "numpy": np.__version__,
                            "device": str(wp.get_device()),
                        },
                        "execution": {
                            "resumed_physics_this_run": True,
                            "wall_seconds": time.perf_counter() - started,
                        },
                        "scope": config["scope"],
                        "next_gate": (
                            "review_local_sparse_transient_policy_without_"
                            "rerunning_physics"),
                    }
                    _write_json(output, failure)
                    return failure
                candidate_arrays = checkpoint_arrays(
                    fixture, bridge.state, simulation_time)
                identity = write_checkpoint_atomic(checkpoint, candidate_arrays)
                candidate_metadata = _checkpoint_metadata(
                    status=CANDIDATE_STATUS, checkpoint=checkpoint,
                    identity=identity, arrays=candidate_arrays,
                    config_path=config_path, runner_path=runner_path,
                    recovery_metadata=recovery_metadata,
                    simulation_time_s=simulation_time,
                    data_domain=config["recovery"]["data_domain"])
                _write_json(candidate_metadata_path, candidate_metadata)
                stage = {
                    "schema": REPORT_SCHEMA,
                    "status": "local_47_robust_candidate_saved_pending_audit",
                    "checks": {
                        "parent_only_failed_original_strict_hold": True,
                        "source_recovery_sha256_verified": True,
                        "source_recovery_unchanged": (
                            checkpoint_identity(_resolve(
                                root, config["recovery"]["checkpoint"])) ==
                            parent_identity),
                        "parent_sparse_transient_within_budget": True,
                        "combined_robust_hold_passed": True,
                        "candidate_saved_before_audit": True,
                        "final_metadata_written": False,
                    },
                    "source_recovery_checkpoint": parent_identity,
                    "parent_checks": parent_checks,
                    "parent_analysis": parent_analysis,
                    "resume": {
                        "executed_frames": len(resume_trace),
                        "trace": resume_trace,
                        "restore": restore,
                    },
                    "combined_analysis": combined,
                    "candidate_checkpoint": {
                        **candidate_metadata,
                        "metadata_file": candidate_metadata_path.name,
                        "metadata_sha256": sha256(candidate_metadata_path),
                    },
                    "inputs": {
                        "config_sha256": sha256(config_path),
                        "runner_sha256": sha256(runner_path),
                        "physical_config_sha256":
                            config["physical_config"]["sha256"],
                    },
                    "environment": {
                        "python": platform.python_version(),
                        "newton": newton.__version__, "warp": wp.__version__,
                        "numpy": np.__version__, "device": str(wp.get_device()),
                    },
                    "execution": {
                        "resumed_physics_this_run": True,
                        "resume_wall_seconds": time.perf_counter() - started,
                    },
                    "scope": config["scope"],
                    "next_gate": "fresh_runtime_120_frame_strict_restore_audit",
                }
                _write_json(stage_path, stage)
            finally:
                bridge.close(kill=False)
        else:
            candidate_metadata = json.loads(
                candidate_metadata_path.read_text(encoding="utf-8"))
            if (candidate_metadata.get("status") != CANDIDATE_STATUS
                    or candidate_metadata.get("source_config_sha256") !=
                        sha256(config_path)
                    or candidate_metadata.get("source_runner_sha256") !=
                        sha256(runner_path)):
                raise ValueError("fortyseven_recovery_hold_candidate_binding")
            candidate_arrays, _, _ = load_checkpoint_readonly(
                checkpoint, candidate_metadata_path,
                _contract_from_metadata(candidate_metadata))
            if not stage_path.is_file():
                raise RuntimeError("fortyseven_recovery_hold_candidate_without_stage")
            stage = json.loads(stage_path.read_text(encoding="utf-8"))

        if stage.get("status") == "local_47_restore_audit_passed_pending_promotion":
            return _promote(
                stage=stage, output=output, checkpoint=checkpoint,
                candidate_metadata_path=candidate_metadata_path,
                final_metadata_path=final_metadata_path)

        candidate_arrays, candidate_loaded_metadata, candidate_identity = (
            load_checkpoint_readonly(
                checkpoint, candidate_metadata_path,
                _contract_from_metadata(candidate_metadata)))
        audit_fixture, audit_bridge, audit_restore = restore_runtime(
            args, physical, root, records, candidate_arrays, work, "audit-47")
        audit_trace = []
        try:
            for frame in range(int(audit_policy["frames"])):
                row = _step(
                    audit_fixture, audit_bridge, records, physical, outer, frame)
                audit_trace.append(row)
                classification = classify_hold_frame(
                    row, physical["probe"], policy)["classification"]
                if frame == 0 or (frame + 1) % 30 == 0:
                    print(
                        "PROGRESS_LOCAL_47_STRICT_AUDIT "
                        f"frame={frame + 1}/{audit_policy['frames']} "
                        f"class={classification} "
                        f"particle_max={row['particle_max_speed_m_s']:.9g}",
                        flush=True)
                if classification != "strict":
                    break
        finally:
            audit_bridge.close(kill=False)

        audit_analysis = analyze_trace(
            audit_trace, physical["probe"], policy)
        audit_passed = (
            len(audit_trace) == int(audit_policy["frames"])
            and audit_analysis["strict_frames"] == len(audit_trace))
        after = checkpoint_identity(checkpoint)
        audit = {
            "requested_frames": int(audit_policy["frames"]),
            "executed_frames": len(audit_trace),
            "all_frames_strict": audit_passed,
            "analysis": audit_analysis,
            "trace": audit_trace,
            "restore": audit_restore,
            "candidate_identity_before": candidate_identity,
            "candidate_identity_after": after,
            "candidate_unchanged": after == candidate_identity,
            "metadata_checkpoint_sha256_verified": (
                candidate_loaded_metadata["checkpoint_sha256"] ==
                candidate_identity["sha256"]),
        }
        stage["strict_restore_audit"] = audit
        stage["checks"].update({
            "candidate_restore_particle_fields_exact": all(
                audit_restore["restore"]["field_readback_exact"].values()),
            "candidate_restore_solver_grid_exact": all(
                audit_restore["restore"]["solver_continuation"]
                ["grid_readback_exact"].values()),
            "candidate_restore_solver_warmstart_exact": all(
                audit_restore["restore"]["solver_continuation"]
                ["solver_field_readback_exact"].values()),
            "candidate_restore_rigid_qpos": audit_restore["restore"]
                ["rigid_qpos_readback_within_tolerance"],
            "candidate_restore_rigid_qvel": audit_restore["restore"]
                ["rigid_qvel_readback_within_tolerance"],
            "candidate_unchanged_by_audit": after == candidate_identity,
            "strict_restore_audit_passed": audit_passed,
        })
        stage["execution"]["audit_wall_seconds"] = (
            time.perf_counter() - started)
        if not audit_passed:
            stage["status"] = "local_47_restore_audit_failed_candidate_preserved"
            stage["next_gate"] = "inspect_strict_audit_without_rerunning_resume"
            _write_json(stage_path, stage)
            _write_json(output, stage)
            return stage
        stage["status"] = "local_47_restore_audit_passed_pending_promotion"
        stage["next_gate"] = "promote_preserved_candidate_metadata"
        _write_json(stage_path, stage)
        return _promote(
            stage=stage, output=output, checkpoint=checkpoint,
            candidate_metadata_path=candidate_metadata_path,
            final_metadata_path=final_metadata_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"],
        "checks": report["checks"],
        "checkpoint": report.get("checkpoint"),
        "candidate_checkpoint": report.get("candidate_checkpoint"),
        "strict_restore_audit": {
            key: value for key, value in report.get(
                "strict_restore_audit", {}).items()
            if key not in {"trace", "restore", "analysis"}
        },
        "next_gate": report["next_gate"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == (
        "local_47_robust_recovery_checkpoint_saved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
