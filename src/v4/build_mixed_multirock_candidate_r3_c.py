#!/usr/bin/env python3
"""Build the isolated R3-C two-rock bucket-push model from frozen R1 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from mixed_material_r1_mechanism import build_candidate


CONFIG_SCHEMA = "excavation_mixed_material_r3_c_config_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_r3_c_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("r3_b_config_path", "r3_b_config_sha256"),
        ("r3_b_bridge_generator_path", "r3_b_bridge_generator_sha256"),
        ("r3_b_report_path", "r3_b_report_sha256"),
        ("r1_config_path", "r1_config_sha256"),
        ("mechanism_model_path", "mechanism_model_sha256"),
        ("rock_asset_path", "rock_asset_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError("r3_c_frozen_input_sha256_mismatch:" + path_key)
    report = json.loads(
        (repository / frozen["r3_b_report_path"]).read_text(encoding="utf-8"))
    if (report.get("status") != "r3_b_bucket_force_partition_probe_passed" or
            not report.get("physics_qualified") or
            not all(report.get("checks", {}).values())):
        raise ValueError("r3_c_r3_b_qualification_missing")
    bridge = config["bridge"]
    if bridge != {
        "ready_schema": "excavation_mixed_coupled_bridge_r3_c_v1",
        "state_schema": "excavation_mixed_coupled_bridge_r3_c_state_v1",
        "required_rock_count": 2,
        "allowed_external_wrench_bodies": [
            "bucket", "r1-rock-000", "r1-rock-001"],
        "external_wrench_frame": "world",
        "bucket_torque_reference": (
            "bucket_origin_at_mujoco_interval_start_fixed_for_interval"),
        "rock_torque_reference": (
            "corresponding_rock_origin_at_previous_mpm_interval_end"),
        "coupling_schedule": (
            "previous_mpm_interval_per_body_wrenches_applied_to_next_mujoco_interval"),
    }:
        raise ValueError("invalid_r3_c_bridge_contract")
    scope = config["scope"]
    if (not scope["exactly_two_rocks"] or not scope["small_fixture_only"] or
            scope["multi_rock_mpm_physics"] or scope["official_envelope"] or
            scope["material_calibration"] or scope["complete_in_process_reset"] or
            scope["training"] or scope["frozen_r1_r2_r3_a_r3_b_modified"]):
        raise ValueError("r3_c_scope_expanded")
    gate = config["first_gate"]
    if gate != {
        "model_generation": True,
        "bridge_generation": True,
        "bridge_compile": True,
        "time_zero_protocol": True,
        "gpu_mpm_step": False,
        "physics_qualified": False,
        "training_qualified": False,
    }:
        raise ValueError("r3_c_first_gate_scope")
    fixture = config["two_rock_fixture"]
    rocks = fixture["rocks"]
    if (fixture["scenario_name"] != "r3_c_two_rock_bucket_push" or
            fixture["command"] != [0.06, 0.0, 0.0, 0.0] or
            float(fixture["duration_s"]) != 1.0 or len(rocks) != 2):
        raise ValueError("r3_c_fixture_contract")
    for index, rock in enumerate(rocks):
        if (rock["rock_id"] != f"r1-rock-{index:03d}" or
                rock["asset_path"] != frozen["rock_asset_path"] or
                float(rock["equivalent_diameter_m"]) !=
                float(frozen["rock_equivalent_diameter_m"]) or
                float(rock["density_kg_m3"]) !=
                float(frozen["rock_density_kg_m3"])):
            raise ValueError("r3_c_rock_contract")
    positions = np.asarray(
        [rock["initial_position_world_m"] for rock in rocks], dtype=np.float64)
    separation = float(np.linalg.norm(positions[1] - positions[0]))
    if (not np.isfinite(positions).all() or separation + 1.0e-12 <
            float(fixture["minimum_initial_center_separation_m"])):
        raise ValueError("r3_c_initial_separation")


def build_two_rock_candidate(config_path: str | Path,
                             repository_root: str | Path,
                             inventory_path: str | Path,
                             output_model: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    fixture = config["two_rock_fixture"]
    r1_config = json.loads(
        (repository / frozen["r1_config_path"]).read_text(encoding="utf-8"))
    scenario = {
        "duration_s": fixture["duration_s"],
        "command": fixture["command"],
        "acceptance": {
            "status": "r3_c_structural_candidate_only"
        },
        "rocks": fixture["rocks"],
        "fixture": r1_config["mechanism_candidate"]["unit_scenarios"]
        ["bucket_push"]["fixture"],
    }
    scenario_name = fixture["scenario_name"]
    r1_config["mechanism_candidate"]["unit_scenarios"][scenario_name] = scenario
    derived_config_bytes = _canonical_bytes(r1_config)
    with tempfile.TemporaryDirectory(prefix="mixed-material-r3-c-model-") as temporary:
        derived_config = Path(temporary) / "derived-r1-config.json"
        derived_config.write_bytes(derived_config_bytes)
        manifest = build_candidate(
            repository / frozen["mechanism_model_path"], output_model,
            repository, inventory_path, derived_config, scenario_name)
    if manifest["expected_candidate_contract"] != {
        "nq": 33, "nv": 30, "nu": 12, "nplugin": 0, "rock_count": 2,
    }:
        raise ValueError("r3_c_model_contract")
    manifest.update({
        "schema": "excavation_mixed_multirock_candidate_r3_c_manifest_v1",
        "status": "r3_c_two_rock_model_generated",
        "r3_c_config_sha256": sha256(config_path),
        "derived_r1_config_sha256": hashlib.sha256(
            derived_config_bytes).hexdigest(),
        "frozen_r1_r2_r3_a_r3_b_modified": False,
        "physics_qualified": False,
        "training_qualified": False,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    manifest = build_two_rock_candidate(
        args.config, args.repository_root, args.inventory, args.output_model)
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
