#!/usr/bin/env python3
"""Replay the admitted three-layer gate and save its full continuation state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

import run_layered_mound_thirtythree_rock_three_layer as base
from build_mixed_material_capacity_preflight import sha256
from run_mixed_material_r3_a_probe import _write_json


REPORT_SCHEMA = "excavation_layered_mound_three_layer_checkpoint_report_v1"
CHECKPOINT_SCHEMA = "excavation_layered_mound_continuation_checkpoint_v1"
MPM_STATE_FIELDS = (
    "particle_elastic_strain", "particle_transform", "particle_qd_grad",
    "particle_stress", "particle_Jp")


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
    arrays = {
        "particle_q": fixture.state.particle_q.numpy(),
        "particle_qd": fixture.state.particle_qd.numpy(),
        "particle_f": fixture.state.particle_f.numpy(),
        "particle_flags": fixture.model.particle_flags.numpy(),
        **{name: getattr(fixture.state.mpm, name).numpy()
           for name in MPM_STATE_FIELDS},
        "rigid_ids": np.asarray(
            [rock["rock_id"] for rock in bridge.state["rocks"]]),
        "rigid_qpos": np.asarray(
            [rock["qpos"] for rock in bridge.state["rocks"]], dtype=np.float64),
        "rigid_qvel": np.asarray(
            [rock["qvel"] for rock in bridge.state["rocks"]], dtype=np.float64),
        "rigid_reported_linear_velocity": np.asarray([
            rock["linear_velocity_world_m_s"]
            for rock in bridge.state["rocks"]], dtype=np.float64),
        "rigid_reported_angular_velocity": np.asarray([
            rock["angular_velocity_world_rad_s"]
            for rock in bridge.state["rocks"]], dtype=np.float64),
        "simulation_time_s": np.asarray(
            [bridge.state["time_s"]], dtype=np.float64),
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items()
               if key != "rigid_ids"):
        raise RuntimeError("checkpoint_nonfinite_array")
    with tempfile.NamedTemporaryFile(
            dir=checkpoint.parent, prefix=checkpoint.name + ".",
            suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    os.replace(temporary, checkpoint)
    saved.update({
        "array_count": len(arrays),
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in arrays.items()},
        "particle_count": int(len(arrays["particle_q"])),
        "rock_count": int(len(arrays["rigid_qpos"])),
        "simulation_time_s": float(arrays["simulation_time_s"][0]),
    })
    checkpoint_hash = sha256(checkpoint)
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
        **saved,
    }
    _write_json(metadata_path, metadata)
    metadata_hash = sha256(metadata_path)
    report["schema"] = REPORT_SCHEMA
    report["status"] = "three_layer_checkpoint_saved"
    report["checkpoint"] = {
        **metadata, "metadata_file": metadata_path.name,
        "metadata_sha256": metadata_hash,
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
