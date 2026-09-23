#!/usr/bin/env python3
"""Replay the admitted three-layer gate and save its full continuation state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import run_layered_mound_thirtythree_rock_three_layer as base
from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    CHECKPOINT_SCHEMA,
    MPM_STATE_FIELDS,
    SOLVER_STATE_FIELDS,
    checkpoint_arrays,
    write_checkpoint_atomic,
)
from run_mixed_material_r3_a_probe import _write_json


REPORT_SCHEMA = "excavation_layered_mound_three_layer_checkpoint_report_v1"


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    for path in (output, checkpoint, metadata_path):
        if path.exists():
            raise FileExistsError(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    instances = []
    bridges = []
    saved = {}
    OriginalFixture = base.MultiRockFixture
    OriginalBridge = base.RuntimeBridge

    class CapturedFixture(OriginalFixture):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            instances.append(self)

    class CapturedBridge(OriginalBridge):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            bridges.append(self)

    base.MultiRockFixture = CapturedFixture
    base.RuntimeBridge = CapturedBridge
    try:
        report = base.run(args)
    finally:
        base.MultiRockFixture = OriginalFixture
        base.RuntimeBridge = OriginalBridge
    if report["status"] != "thirtythree_rock_three_layer_passed":
        raise RuntimeError("checkpoint_source_gate_not_passed")
    if len(instances) != 1 or len(bridges) != 1:
        raise RuntimeError("checkpoint_capture_contract")
    fixture, bridge = instances[0], bridges[0]
    arrays = checkpoint_arrays(
        fixture, bridge.state, float(bridge.state["time_s"]))
    identity = write_checkpoint_atomic(checkpoint, arrays)
    saved.update({
        "array_count": len(arrays),
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in arrays.items()},
        "particle_count": int(len(arrays["particle_q"])),
        "rock_count": int(len(arrays["rigid_qpos"])),
        "simulation_time_s": float(arrays["simulation_time_s"][0]),
    })
    checkpoint_hash = identity["sha256"]
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "status": "three_layer_continuation_checkpoint_saved",
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": checkpoint_hash,
        "source_config_sha256": sha256(Path(args.config).resolve()),
        "source_runner_sha256": sha256(
            root / "src/v4/run_layered_mound_thirtythree_rock_three_layer.py"),
        "continuation_point": "after_120_frame_stable_window_and_240_frame_hold",
        "gravity_predictor_required_on_next_free_step": True,
        "same_solver_material_and_dt_required": True,
        "fixed_grid_reference_saved": True,
        "newton_solver_warmstart_saved": True,
        "newton_version": report["environment"]["newton"],
        "warp_version": report["environment"]["warp"],
        "solver_grid_type": fixture.solver.grid_type,
        "solver_warmstart_mode": fixture.solver._stress_warmstart,
        "collider_velocity_mode": fixture.solver.collider_velocity_mode,
        **saved,
    }
    _write_json(metadata_path, metadata)
    metadata_hash = sha256(metadata_path)
    report["schema"] = REPORT_SCHEMA
    report["status"] = "three_layer_checkpoint_saved"
    report["checkpoint"] = {
        **metadata, "metadata_file": metadata_path.name,
        "metadata_sha256": metadata_hash,
        "size_bytes": identity["size_bytes"],
    }
    report["inputs"]["checkpoint_wrapper_sha256"] = sha256(Path(__file__).resolve())
    report["execution"]["wall_seconds_with_checkpoint"] = time.perf_counter() - started
    report["next_gate"] = "restore_checkpoint_and_verify_one_step_continuity"
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "checkpoint": report["checkpoint"], "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "three_layer_checkpoint_saved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
