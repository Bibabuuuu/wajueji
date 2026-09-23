#!/usr/bin/env python3
"""Bind the audited 65-rock checkpoint to the stationary excavation scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_mixed_material_capacity_preflight import sha256
import run_layered_47_excavation_preflight as shared


CONFIG_SCHEMA = "excavation_layered_65_excavation_preflight_config_v1"
REPORT_SCHEMA = "excavation_layered_65_excavation_preflight_report_v1"
LAYERS = ("lower", "lower_middle", "middle", "upper", "top")
PROFILE = {
    "layers": LAYERS,
    "expected_ids": (
        [f"lower-{index:03d}" for index in range(9)]
        + [f"lower_middle-{index:03d}" for index in range(11)]
        + [f"middle-{index:03d}" for index in range(13)]
        + [f"upper-{index:03d}" for index in range(14)]
        + [f"top-{index:03d}" for index in range(18)]),
    "temporary_directory_prefix": "layered-65-excavation-preflight-",
    "restored_runtime_label": "source-restored-65",
    "progress_label": "PROGRESS_65_PREFLIGHT",
    "report_schema": REPORT_SCHEMA,
    "passed_status": "layered_65_excavation_preflight_passed",
    "failed_status": "layered_65_excavation_preflight_failed",
    "passed_next_gate":
        "scripted_approach_to_program_selected_noncontact_prepoint",
    "failed_next_gate": "fix_precontact_scene_binding",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "bind_audited_65_to_stationary_open_bucket_scene"):
        raise ValueError("excavation_65_preflight_config_schema_or_status")
    for group, fields in (
            ("physical_geometry", ("manifest",)),
            ("capacity_bridge", ("config",)),
            ("runtime_dependencies", ("checkpoint", "runtime", "predictor"))):
        binding = config.get(group, {})
        for field in fields:
            target = _resolve(root, binding.get(field + "_path", ""))
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(
                    f"excavation_65_preflight_{group}_{field}_binding")
    _, operator = shared._load_bound_json(
        root, config.get("operator_start", {}), "65_operator_start")
    _, proxy = shared._load_bound_json(
        root, config.get("open_bucket_proxy", {}), "65_open_bucket_proxy")
    _, receiver = shared._load_bound_json(
        root, config.get("truck_receiver_observer", {}), "65_truck_receiver")
    required_source = {
        "default_directory_from_home":
            "a30-layered-47-to-65-v5/checkpoints",
        "filename_pattern": "five-layer-65-checkpoint-cuda-0-*.npz",
        "checkpoint_schema":
            "excavation_layered_mound_continuation_checkpoint_v2",
        "checkpoint_status": "five_layer_continuation_checkpoint_saved",
        "sha256":
            "22fe16dfecc178257754d6646142de29a231ff91e0f09bee03210a414d64e9b7",
        "source_config_sha256":
            "ef8484dd54aa00e414bd7682ecdc0c69be9c70f1e74a4a64017455545f50c8fd",
        "source_runner_sha256":
            "8d89eabfa8d1fe9c182889c58a162c62040032a73443ea8965fd33ff9ed566b2",
        "particle_count": 266091,
        "rock_count": 65,
        "newton_version": "1.5.1",
        "warp_version": "1.17.0",
        "read_only": True,
        "metadata_required": True,
    }
    if config.get("input_checkpoint") != required_source:
        raise ValueError("excavation_65_preflight_input_checkpoint_contract")
    qpos = operator.get("qpos")
    if (not isinstance(qpos, list) or len(qpos) != 19
            or not np.isfinite(np.asarray(qpos, dtype=np.float64)).all()
            or operator.get("physics_steps") != 0):
        raise ValueError("excavation_65_preflight_operator_start_contract")
    if (proxy.get("schema") != "excavation_open_bucket_collision_proxy_v1"
            or proxy.get("geom_count") != 18
            or proxy.get("has_mouth_cap") is not False
            or receiver.get("schema") != "truck_receiver_triangles_v1"):
        raise ValueError("excavation_65_preflight_scene_geometry_contract")
    expected_preflight = {
        "hold_frames": 120,
        "minimum_initial_vertical_clearance_m": 0.02,
        "maximum_bucket_origin_drift_m": 0.002,
        "maximum_bucket_rotation_drift_rad": 0.002,
        "writes_checkpoint": False,
        "requires_zero_bucket_contact_during_hold": True,
    }
    if config.get("preflight") != expected_preflight:
        raise ValueError("excavation_65_preflight_gate_contract")
    expected_scope = {
        "source_checkpoint_mutated": False,
        "new_particles_added": False,
        "new_rocks_added": False,
        "new_checkpoint_written": False,
        "bucket_collision_active": True,
        "bucket_motion_commanded": False,
        "truck_visual_and_receiver_observer_reused": True,
        "truck_physics_collider_qualified": False,
        "digging_executed": False,
        "training": False,
    }
    if config.get("scope") != expected_scope:
        raise ValueError("excavation_65_preflight_scope_contract")
    return config


def run(args) -> dict:
    return shared.run(args, config_loader=load_config, profile=PROFILE)


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
        "initial_clearance": report["initial_clearance"],
        "hold": {key: value for key, value in report["hold"].items()
                 if key != "trace"},
        "execution": report["execution"],
        "next_gate": report["next_gate"],
    }, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == PROFILE["passed_status"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
