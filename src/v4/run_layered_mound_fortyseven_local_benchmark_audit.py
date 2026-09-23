#!/usr/bin/env python3
"""Strict read-only audit for a local-only 47-rock benchmark candidate."""

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
from layered_mound_checkpoint import checkpoint_identity, load_checkpoint_readonly
from layered_mound_checkpoint_runtime import restore_runtime
from run_layered_mound_fortyseven_recovery_hold import (
    _contract_from_metadata,
    _outer_bounds,
    _records,
    _resolve,
    _step,
    classify_hold_frame,
)
from run_layered_mound_fortyseven_rock_four_layer import (
    load_config as load_physical_config,
)
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = (
    "excavation_layered_mound_fortyseven_local_benchmark_audit_config_v1")
REPORT_SCHEMA = (
    "excavation_layered_mound_fortyseven_local_benchmark_audit_report_v1")


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "read_only_strict_audit_for_local_manual_benchmark_only"):
        raise ValueError("local_47_benchmark_audit_config_schema_or_status")
    physical_binding = config.get("physical_config", {})
    physical_path = _resolve(root, physical_binding.get("path", ""))
    if (not physical_path.is_file()
            or sha256(physical_path) != physical_binding.get("sha256")):
        raise ValueError("local_47_benchmark_audit_physical_config")
    raw_physical = json.loads(physical_path.read_text(encoding="utf-8"))
    physical = load_physical_config(
        physical_path, root,
        expected_input_checkpoint=raw_physical.get("input_checkpoint"))
    source = config.get("input_checkpoint", {})
    required = {"path", "metadata", "sha256", "metadata_sha256", "status"}
    if set(source) != required:
        raise ValueError("local_47_benchmark_audit_input_fields")
    for name, digest_name in (("path", "sha256"),
                              ("metadata", "metadata_sha256")):
        target = _resolve(root, source[name])
        if not target.is_file() or sha256(target) != source[digest_name]:
            raise ValueError(f"local_47_benchmark_audit_{name}_binding")
    if config.get("audit") != {
            "frames": 120,
            "all_frames_must_be_strict": True,
            "writes_checkpoint": False,
            "source_identity_rechecked_after_run": True}:
        raise ValueError("local_47_benchmark_audit_policy")
    if config.get("robust_thresholds") != {
            "strict_particle_max_speed_m_s": 0.05,
            "transient_particle_max_speed_ceiling_m_s": 0.10,
            "maximum_transient_frames": 6,
            "maximum_consecutive_transient_frames": 2}:
        raise ValueError("local_47_benchmark_audit_thresholds")
    if config.get("scope") != {
            "local_manual_benchmark_only": True,
            "formal_a30_equivalent": False,
            "training_eligible": False,
            "source_checkpoint_mutated": False,
            "new_checkpoint_written": False}:
        raise ValueError("local_47_benchmark_audit_scope")
    return config, physical


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config, physical = load_config(config_path, root)
    source = config["input_checkpoint"]
    checkpoint = _resolve(root, source["path"])
    metadata_path = _resolve(root, source["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != source["status"]:
        raise ValueError("local_47_benchmark_audit_source_status")
    arrays, loaded_metadata, before = load_checkpoint_readonly(
        checkpoint, metadata_path, _contract_from_metadata(metadata))
    records = _records(physical, root)
    outer = _outer_bounds(physical, root)

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    trace = []
    with tempfile.TemporaryDirectory(prefix="local-47-benchmark-audit-") as raw:
        fixture, bridge, restore = restore_runtime(
            args, physical, root, records, arrays, Path(raw), "audit-47")
        try:
            for frame in range(int(config["audit"]["frames"])):
                row = _step(
                    fixture, bridge, records, physical, outer, frame)
                classification = classify_hold_frame(
                    row, physical["probe"], config["robust_thresholds"])
                row["strict_classification"] = classification
                trace.append(row)
                if frame == 0 or (frame + 1) % 30 == 0:
                    print(
                        "PROGRESS_LOCAL_47_BENCHMARK_AUDIT "
                        f"frame={frame + 1}/{config['audit']['frames']} "
                        f"class={classification['classification']} "
                        f"particle_max={row['particle_max_speed_m_s']:.9g}",
                        flush=True)
                if classification["classification"] != "strict":
                    break
        finally:
            bridge.close(kill=False)
    after = checkpoint_identity(checkpoint)
    executed = len(trace)
    all_strict = (
        executed == int(config["audit"]["frames"])
        and all(row["strict_classification"]["classification"] == "strict"
                for row in trace))
    solver_restore = restore["restore"]["solver_continuation"]
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] == source["sha256"],
        "source_checkpoint_metadata_verified": (
            loaded_metadata["checkpoint_sha256"] == before["sha256"]),
        "particle_fields_restored_exactly": all(
            restore["restore"]["field_readback_exact"].values()),
        "solver_grid_restored_exactly": all(
            solver_restore["grid_readback_exact"].values()),
        "solver_warmstart_restored_exactly": all(
            solver_restore["solver_field_readback_exact"].values()),
        "rigid_qpos_restored": restore["restore"]
            ["rigid_qpos_readback_within_tolerance"],
        "rigid_qvel_restored": restore["restore"]
            ["rigid_qvel_readback_within_tolerance"],
        "all_requested_frames_executed": executed == int(
            config["audit"]["frames"]),
        "all_frames_strict": all_strict,
        "source_checkpoint_unchanged": after == before,
        "no_checkpoint_written": True,
    }
    passed = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("local_47_manual_benchmark_audit_passed" if passed else
                   "local_47_manual_benchmark_audit_failed"),
        "checks": checks,
        "source_checkpoint": before,
        "source_checkpoint_after": after,
        "source_metadata": loaded_metadata,
        "restore": restore,
        "hold": {
            "requested_frames": int(config["audit"]["frames"]),
            "executed_frames": executed,
            "maximum_particle_speed_m_s": max(
                (row["particle_max_speed_m_s"] for row in trace), default=None),
            "maximum_particle_p99_speed_m_s": max(
                (row["particle_p99_speed_m_s"] for row in trace), default=None),
            "maximum_rock_linear_speed_m_s": max(
                (row["maximum_rock_linear_speed_m_s"] for row in trace),
                default=None),
            "maximum_rock_angular_speed_rad_s": max(
                (row["maximum_rock_angular_speed_rad_s"] for row in trace),
                default=None),
            "trace": trace,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "physical_config_sha256": config["physical_config"]["sha256"],
            "runner_sha256": sha256(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "newton_solver_step_calls": executed,
            "mujoco_step_calls": executed,
        },
        "scope": config["scope"],
        "next_gate": ("local_manual_precontact_speed_benchmark" if passed else
                      "do_not_use_local_47_for_manual_benchmark"),
    }
    if not _finite_tree(report):
        raise RuntimeError("local_47_benchmark_audit_nonfinite_report")
    _write_json(output, report)
    return report


def main() -> int:
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
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "source_checkpoint": report["source_checkpoint"],
        "hold": {key: value for key, value in report["hold"].items()
                 if key != "trace"},
        "execution": report["execution"], "next_gate": report["next_gate"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == (
        "local_47_manual_benchmark_audit_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
