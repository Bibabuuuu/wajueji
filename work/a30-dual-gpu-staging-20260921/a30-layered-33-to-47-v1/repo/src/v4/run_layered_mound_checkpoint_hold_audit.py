#!/usr/bin/env python3
"""Read-only restore and short hold audit for the admitted 33-rock checkpoint."""

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
    checkpoint_identity,
    load_checkpoint_readonly,
)
from layered_mound_checkpoint_runtime import restore_runtime, run_continuity_hold
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_checkpoint_hold_audit_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_checkpoint_hold_audit_report_v1"
LAYERS = ("lower", "lower_middle", "middle")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "read_only_restore_33_then_short_uncontrolled_hold"):
        raise ValueError("checkpoint_audit_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("runtime_dependencies", ("checkpoint", "runtime", "predictor"))):
        binding = config.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if not target.is_file() or sha256(target) != binding.get(field + "_sha256"):
                raise ValueError(f"checkpoint_audit_{group}_{field}_binding")
    source = config.get("input_checkpoint", {})
    if source != {
            "default_path": "/home/user01/a30-layered-checkpoint-v2/checkpoints/three-layer-checkpoint-cuda-0-20260921-170946-35334.npz",
            "sha256": "d30db1bd42c064934badc21716d0ac6f1db80e24d009c71d27a4751b77f3f545",
            "source_config_sha256": "bbe3c6b3a1893093d87ee6700169fcb77960d4c3428078a533f8edde087508b1",
            "source_runner_sha256": "71b0dbca886d476462ab6e043963393b4cc6ee3276fb44054113ba1151d71a86",
            "particle_count": 221882,
            "rock_count": 33,
            "read_only": True,
            "metadata_required": True}:
        raise ValueError("checkpoint_audit_input_contract")
    if config.get("audit") != {
            "hold_frames": 120,
            "writes_checkpoint": False,
            "source_identity_rechecked_after_run": True}:
        raise ValueError("checkpoint_audit_hold_contract")
    if config.get("construction") != {"start_surface_clearance_m": 0.005}:
        raise ValueError("checkpoint_audit_construction_contract")
    probe = config.get("probe", {})
    if (probe.get("output_dt_s") != 1.0 / 120.0
            or probe.get("mujoco_internal_substeps") != 40
            or probe.get("maximum_stable_rock_linear_speed_m_s") != 0.01
            or probe.get("maximum_stable_rock_angular_speed_rad_s") != 0.15
            or probe.get("maximum_stable_particle_speed_m_s") != 0.05
            or probe.get("maximum_stable_particle_p99_speed_m_s") != 0.01):
        raise ValueError("checkpoint_audit_probe_contract")
    if config.get("scope") != {
            "source_checkpoint_mutated": False,
            "new_particles_added": False,
            "new_rocks_added": False,
            "new_checkpoint_written": False,
            "independent_gpu_audit": True,
            "full_scene_claimed": False,
            "training": False}:
        raise ValueError("checkpoint_audit_scope_contract")
    return config


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    metadata = Path(args.checkpoint_metadata).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path, root)
    arrays, source_metadata, before = load_checkpoint_readonly(
        checkpoint, metadata, config["input_checkpoint"])
    manifest_path = _resolve(root, config["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"] if rock["layer_id"] in LAYERS]
    if (len(records) != 33 or [rock["rock_id"] for rock in records[:9]] !=
            [f"lower-{index:03d}" for index in range(9)]
            or [rock["rock_id"] for rock in records[9:20]] !=
            [f"lower_middle-{index:03d}" for index in range(11)]
            or [rock["rock_id"] for rock in records[20:]] !=
            [f"middle-{index:03d}" for index in range(13)]):
        raise ValueError("checkpoint_audit_manifest_order")
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    with tempfile.TemporaryDirectory(prefix="checkpoint-hold-audit-") as raw:
        work = Path(raw)
        fixture, bridge, restore = restore_runtime(
            args, config, root, records, arrays, work, "restored-33")
        try:
            hold = run_continuity_hold(
                fixture, bridge, records, config,
                int(config["audit"]["hold_frames"]))
        finally:
            bridge.close(kill=not all(hold["checks"].values()) if "hold" in locals()
                         else True)
    after = checkpoint_identity(checkpoint)
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] ==
            config["input_checkpoint"]["sha256"],
        "source_checkpoint_metadata_verified": source_metadata["checkpoint_sha256"] ==
            before["sha256"],
        "all_particle_fields_restored_exactly": all(
            restore["restore"]["field_readback_exact"].values()),
        "rigid_qpos_restored": restore["restore"]["rigid_qpos_readback_within_tolerance"],
        "rigid_qvel_restored": restore["restore"]["rigid_qvel_readback_within_tolerance"],
        "one_step_continuity_passed": (
            hold["checks"]["first_step_stable"]
            and hold["checks"]["first_step_position_continuous"]),
        "short_hold_all_checks_passed": all(hold["checks"].values()),
        "source_checkpoint_unchanged": after == before,
        "no_checkpoint_written": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "checkpoint_hold_audit_passed" if all(checks.values()) else
                  "checkpoint_hold_audit_failed",
        "checks": checks,
        "source_checkpoint": before,
        "source_checkpoint_after": after,
        "source_metadata": source_metadata,
        "restore": restore,
        "hold": hold,
        "inputs": {
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "newton_solver_step_calls": hold["executed_frames"],
            "mujoco_step_calls": hold["executed_frames"],
            "wall_seconds": time.perf_counter() - started,
        },
        "scope": config["scope"],
        "next_gate": "independent_audit_complete_no_mainline_write",
    }
    if not _finite_tree(report):
        raise RuntimeError("checkpoint_audit_nonfinite_report")
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
        "hold": {key: value for key, value in report["hold"].items()
                 if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "checkpoint_hold_audit_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
