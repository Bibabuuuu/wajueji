#!/usr/bin/env python3
"""Independently restore and hold the non-formal 47-rock speed proxy.

The source ancestry remains diagnostic-only.  Passing this audit proves that
the speed proxy can resume without losing its expensive settled state; it does
not grant manual collection, training, or formal A30 equivalence.
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
from layered_mound_checkpoint import checkpoint_identity, restore_fixture_prefix
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    build_model_and_bridge,
    records_at_qpos,
)
import layered_47_interactive_proxy_checkpoint as proxy_checkpoint
import run_layered_47_excavation_preflight as preflight
import run_layered_47_interactive_proxy_benchmark as benchmark
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_layered_mound_three_rock_adaptive_press import RuntimeBridge
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_47_interactive_proxy_resume_audit_config_v1"
REPORT_SCHEMA = "excavation_layered_47_interactive_proxy_resume_audit_report_v1"
PASSED_STATUS = "layered_47_interactive_proxy_resume_audit_passed"
FAILED_STATUS = "layered_47_interactive_proxy_resume_audit_failed"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "independent_read_only_resume_audit_nonformal_proxy"):
        raise ValueError("interactive_proxy_resume_audit_config_schema_or_status")
    binding = config.get("benchmark_config", {})
    if set(binding) != {"path", "sha256", "profile"}:
        raise ValueError("interactive_proxy_resume_audit_benchmark_binding_fields")
    benchmark_path = _resolve(root, binding["path"])
    if (not benchmark_path.is_file()
            or sha256(benchmark_path) != binding["sha256"]
            or binding["profile"] != "voxel_10cm_sparse"):
        raise ValueError("interactive_proxy_resume_audit_benchmark_binding")
    benchmark_config, loaded = benchmark.load_config(benchmark_path, root)
    source = config.get("input_checkpoint", {})
    if set(source) != {
            "path", "metadata", "sha256", "metadata_sha256"}:
        raise ValueError("interactive_proxy_resume_audit_input_fields")
    for name, digest_name in (
            ("path", "sha256"), ("metadata", "metadata_sha256")):
        target = _resolve(root, source[name])
        if not target.is_file() or sha256(target) != source[digest_name]:
            raise ValueError(f"interactive_proxy_resume_audit_{name}_binding")
    if config.get("audit") != {
            "frames": 120,
            "all_frames_must_be_strict": True,
            "writes_checkpoint": False,
            "writes_progress_journal": True,
            "source_identity_rechecked_after_run": True}:
        raise ValueError("interactive_proxy_resume_audit_policy")
    if config.get("scope") != {
            "resume_continuity_only": True,
            "formal_a30_equivalent": False,
            "manual_collection_eligible": False,
            "training_eligible": False,
            "source_checkpoint_mutated": False,
            "new_checkpoint_written": False}:
        raise ValueError("interactive_proxy_resume_audit_scope")
    return config, {
        "benchmark_path": benchmark_path,
        "benchmark_config": benchmark_config,
        **loaded,
    }


def _append_journal(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")
        stream.flush()


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    journal = output.with_suffix(output.suffix + ".progress.ndjson")
    if output.exists() or journal.exists():
        raise FileExistsError(output if output.exists() else journal)
    config, loaded = load_config(config_path, root)
    source = config["input_checkpoint"]
    checkpoint = _resolve(root, source["path"])
    metadata_path = _resolve(root, source["metadata"])
    arrays, source_metadata, before = (
        proxy_checkpoint.load_checkpoint_readonly(checkpoint, metadata_path))
    base = loaded["base"]
    profile_name = config["benchmark_config"]["profile"]
    solver_profile = loaded["profiles"][profile_name]
    manifest_path = _resolve(root, base["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in preflight.LAYERS]
    if ([rock["rock_id"] for rock in records]
            != list(preflight.DEFAULT_PROFILE["expected_ids"])):
        raise ValueError("interactive_proxy_resume_audit_manifest_order")
    proxy_path, proxy = preflight._load_bound_json(
        root, base["open_bucket_proxy"], "open_bucket_proxy")
    output.parent.mkdir(parents=True, exist_ok=True)
    _append_journal(journal, {
        "event": "started", "source_checkpoint": before,
        "solver_profile": solver_profile,
    })

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    trace: list[dict] = []
    restore = None
    model_manifest = None
    bridge_manifest = None
    compile_command = None
    bridge = None
    with tempfile.TemporaryDirectory(
            prefix="layered-47-interactive-resume-audit-") as raw:
        work = Path(raw)
        restored_records = records_at_qpos(records, arrays["rigid_qpos"])
        model, binary, model_manifest, bridge_manifest, compile_command = (
            build_model_and_bridge(
                args, root, root / base["capacity_bridge"]["config_path"],
                restored_records, work, "interactive-resume-audit"))
        default_bridge = RuntimeBridge(binary, model, len(records))
        default_bridge.close()
        bridge = RestoredRuntimeBridge(
            binary, model, len(records),
            arrays["mechanism_qpos"].tolist(),
            arrays["mechanism_qvel"].tolist(),
            arrays["rigid_qpos"], arrays["rigid_qvel"])
        fixture = preflight.ExcavationFixture(
            base, root, records, arrays["grid_reference_particle_q"],
            proxy, bridge.state, args.device, solver_profile=solver_profile)
        field_restore = restore_fixture_prefix(fixture, arrays)
        fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        fixture.assign(bridge.state)
        solver_restore = proxy_checkpoint.restore_solver_continuation(
            fixture, arrays)
        actual_rock_qpos = np.asarray(
            [rock["qpos"] for rock in bridge.state["rocks"]],
            dtype=np.float64)
        actual_rock_qvel = np.asarray(
            [rock["qvel"] for rock in bridge.state["rocks"]],
            dtype=np.float64)
        actual_mechanism_qpos = np.asarray(
            bridge.state["mechanism_qpos"], dtype=np.float64)
        actual_mechanism_qvel = np.asarray(
            bridge.state["mechanism_qvel"], dtype=np.float64)
        restore = {
            "particle_fields": field_restore,
            "solver_continuation": solver_restore,
            "rigid_qpos_max_abs_readback_error": float(np.max(np.abs(
                actual_rock_qpos - arrays["rigid_qpos"]))),
            "rigid_qvel_max_abs_readback_error": float(np.max(np.abs(
                actual_rock_qvel - arrays["rigid_qvel"]))),
            "mechanism_qpos_max_abs_readback_error": float(np.max(np.abs(
                actual_mechanism_qpos - arrays["mechanism_qpos"]))),
            "mechanism_qvel_max_abs_readback_error": float(np.max(np.abs(
                actual_mechanism_qvel - arrays["mechanism_qvel"]))),
        }
        requested = int(config["audit"]["frames"])
        try:
            for frame in range(requested):
                row, _ = benchmark._advance_frame(
                    fixture, bridge, base, frame)
                safe = bool(
                    row["finite"] and row["stable"]
                    and row["forbidden_contact_count"] == 0
                    and row["external_rigid_support_substeps"] == 0
                    and row["unmapped_active_collider_sample_count"] == 0
                    and row["nonfinite_collider_sample_count"] == 0
                    and row["bucket_mpm_contact_samples"] == 0
                    and row["rock_bucket_contact_substeps"] == 0
                    and row["closure_m"] <= 0.001)
                row["strict"] = safe
                trace.append(row)
                _append_journal(journal, {
                    "event": "audit_frame", "row": row,
                })
                if frame == 0 or (frame + 1) % 30 == 0:
                    print(
                        "PROGRESS_47_INTERACTIVE_RESUME_AUDIT "
                        f"frame={frame + 1}/{requested} strict={safe} "
                        f"wall={row['wall_seconds']:.6f}", flush=True)
                if not row["finite"]:
                    break
        finally:
            bridge.close(kill=not trace or not trace[-1]["finite"])
            bridge = None

    after = checkpoint_identity(checkpoint)
    requested = int(config["audit"]["frames"])
    dt = float(base["probe"]["output_dt_s"])
    maximum_first_step = float(base["probe"][
        "maximum_stable_rock_linear_speed_m_s"]) * dt * 1.01
    timings = [row["wall_seconds"] for row in trace]
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] == source["sha256"],
        "source_checkpoint_metadata_verified": (
            source_metadata["checkpoint_sha256"] == before["sha256"]),
        "particle_fields_restored_exactly": all(
            restore["particle_fields"]["field_readback_exact"].values()),
        "solver_grid_restored_exactly": all(
            restore["solver_continuation"]["grid_readback_exact"].values()),
        "solver_warmstart_restored_exactly": all(
            restore["solver_continuation"]
            ["solver_field_readback_exact"].values()),
        "rigid_qpos_restored": (
            restore["rigid_qpos_max_abs_readback_error"] <= 1e-12),
        "rigid_qvel_restored": (
            restore["rigid_qvel_max_abs_readback_error"] <= 1e-12),
        "mechanism_qpos_restored": (
            restore["mechanism_qpos_max_abs_readback_error"] <= 1e-12),
        "mechanism_qvel_restored": (
            restore["mechanism_qvel_max_abs_readback_error"] <= 1e-12),
        "all_requested_frames_executed": len(trace) == requested,
        "all_frames_strict": bool(trace) and all(row["strict"] for row in trace),
        "first_step_position_continuous": bool(trace) and trace[0][
            "maximum_rock_position_step_m"] <= maximum_first_step,
        "source_checkpoint_unchanged": after == before,
        "no_checkpoint_written": True,
        "manual_collection_not_granted": not config["scope"][
            "manual_collection_eligible"],
        "training_not_granted": not config["scope"]["training_eligible"],
    }
    passed = all(checks.values())
    mean = float(np.mean(timings)) if timings else None
    report = {
        "schema": REPORT_SCHEMA,
        "status": PASSED_STATUS if passed else FAILED_STATUS,
        "checks": checks,
        "source_checkpoint": before,
        "source_checkpoint_after": after,
        "source_metadata": source_metadata,
        "restore": restore,
        "hold": {
            "requested_frames": requested,
            "executed_frames": len(trace),
            "maximum_first_step_position_m": maximum_first_step,
            "maximum_rock_position_step_m": max((
                row["maximum_rock_position_step_m"] for row in trace),
                default=None),
            "maximum_rock_linear_speed_m_s": max((
                row["maximum_rock_linear_speed_m_s"] for row in trace),
                default=None),
            "maximum_rock_angular_speed_rad_s": max((
                row["maximum_rock_angular_speed_rad_s"] for row in trace),
                default=None),
            "trace": trace,
        },
        "timing": {
            "mean_wall_seconds_per_frame": mean,
            "physics_frames_per_wall_second": 1.0 / mean if mean else None,
            "mean_mpm_wall_seconds": float(np.mean([
                row["mpm_wall_seconds"] for row in trace])) if trace else None,
            "mean_mujoco_wall_seconds": float(np.mean([
                row["mujoco_wall_seconds"] for row in trace])) if trace else None,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "benchmark_config_sha256": sha256(loaded["benchmark_path"]),
            "manifest_sha256": sha256(manifest_path),
            "open_bucket_proxy_sha256": sha256(proxy_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
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
            "newton_solver_step_calls": len(trace),
            "mujoco_step_calls": len(trace),
            "progress_journal": str(journal),
        },
        "scope": config["scope"],
        "next_gate": (
            "build_clean_ancestry_interactive_proxy_before_collection"
            if passed else "reject_resume_and_preserve_failure"),
    }
    if not _finite_tree(report):
        raise RuntimeError("interactive_proxy_resume_audit_nonfinite_report")
    _write_json(output, report)
    _append_journal(journal, {
        "event": "completed", "status": report["status"],
        "timing": report["timing"],
    })
    return report


def _parser() -> argparse.ArgumentParser:
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
                "failure": {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "inputs": {
                    "config": str(Path(args.config).resolve()),
                },
                "scope": {
                    "formal_a30_equivalent": False,
                    "manual_collection_eligible": False,
                    "training_eligible": False,
                    "new_checkpoint_written": False,
                },
                "next_gate": "fix_or_reject_resume_without_losing_failure",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, failure)
        print(json.dumps({
            "status": FAILED_STATUS,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc), "report": str(output),
        }, sort_keys=True, allow_nan=False), flush=True)
        return 1
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "timing": report["timing"], "next_gate": report["next_gate"],
        "report": str(output),
    }, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["status"] == PASSED_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
