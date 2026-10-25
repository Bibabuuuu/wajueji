#!/usr/bin/env python3
"""Validate the source checkpoint and portable MuJoCo models before MPM work."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from layered_mound_checkpoint import load_checkpoint_readonly
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    build_model_and_bridge,
    records_at_qpos,
)
from run_layered_mound_checkpoint_hold_audit import load_config as load_audit
from run_layered_mound_fortyseven_rock_four_layer import load_config as load_main
from run_layered_mound_three_rock_adaptive_press import RuntimeBridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata", required=True)
    parser.add_argument("--output-rock-count", type=int, choices=(33, 47), required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if raw_config["schema"].endswith("hold_audit_config_v1"):
        config = load_audit(config_path, root)
    else:
        config = load_main(config_path, root)
    arrays, metadata, identity = load_checkpoint_readonly(
        args.checkpoint, args.checkpoint_metadata, config["input_checkpoint"])
    manifest = json.loads((
        root / config["physical_geometry"]["manifest_path"]
    ).read_text(encoding="utf-8"))
    source_layers = {"lower", "lower_middle", "middle"}
    source_records = [rock for rock in manifest["rocks"]
                      if rock["layer_id"] in source_layers]
    records33 = records_at_qpos(source_records, arrays["rigid_qpos"])
    capacity = root / config["capacity_bridge"]["config_path"]
    with tempfile.TemporaryDirectory(prefix="checkpoint-bridge-preflight-") as raw:
        work = Path(raw)
        model33, binary, model33_manifest, _, _ = build_model_and_bridge(
            args, root, capacity, records33, work, "restored-33")
        default = RuntimeBridge(binary, model33, 33)
        mechanism_qpos = default.state["mechanism_qpos"]
        mechanism_qvel = default.state["mechanism_qvel"]
        default.close()
        restored = RestoredRuntimeBridge(
            binary, model33, 33, mechanism_qpos, mechanism_qvel,
            arrays["rigid_qpos"], arrays["rigid_qvel"])
        restored.close()
        model47_manifest = None
        if args.output_rock_count == 47:
            upper = [rock for rock in manifest["rocks"] if rock["layer_id"] == "upper"]
            records47 = records33 + upper
            model47, _, model47_manifest, _, _ = build_model_and_bridge(
                args, root, capacity, records47, work, "portable-47", binary)
            bridge47 = RuntimeBridge(binary, model47, 47)
            bridge47.close()
    print(json.dumps({
        "status": "PORTABLE_CHECKPOINT_AND_MUJOCO_PREFLIGHT_PASS",
        "checkpoint_sha256": identity["sha256"],
        "checkpoint_particle_count": metadata["particle_count"],
        "checkpoint_rock_count": metadata["rock_count"],
        "output_rock_count": args.output_rock_count,
        "restored_model_resources": model33_manifest["model_resources"],
        "output_model_resources": (
            model47_manifest["model_resources"] if model47_manifest else None),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
