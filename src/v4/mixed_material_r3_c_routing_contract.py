"""Read-only contract validation for the isolated R3-C per-rock routing gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_mixed_multirock_candidate_r3_c import sha256


CONFIG_SCHEMA = "excavation_mixed_material_r3_c_routing_config_v1"


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_r3_c_routing_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("r3_c_config_path", "r3_c_config_sha256"),
        ("r3_c_model_generator_path", "r3_c_model_generator_sha256"),
        ("r3_c_bridge_generator_path", "r3_c_bridge_generator_sha256"),
        ("r3_c_preflight_report_path", "r3_c_preflight_report_sha256"),
        ("r3_b_config_path", "r3_b_config_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError("r3_c_routing_frozen_input_sha256_mismatch:" + path_key)
    preflight = json.loads(
        (repository / frozen["r3_c_preflight_report_path"])
        .read_text(encoding="utf-8"))
    if (preflight.get("status") != "r3_c_two_rock_bridge_preflight_passed" or
            not preflight.get("bridge_preflight_qualified") or
            preflight.get("physics_qualified") or
            not all(preflight.get("checks", {}).values())):
        raise ValueError("r3_c_routing_preflight_missing")
    probe = config["routing_probe"]
    if (probe.get("rock_ids") != ["r1-rock-000", "r1-rock-001"] or
            probe.get("rounds") !=
            ["rock_000_mpm_only", "rock_001_mpm_only"] or
            probe.get("command") != [0.0, 0.0, 0.0, 0.0] or
            int(probe.get("frames", 0)) < 1 or
            int(probe.get("mujoco_internal_substeps", 0)) < 1 or
            float(probe.get("output_dt_s", 0.0)) <= 0.0):
        raise ValueError("r3_c_routing_probe_contract")
    numeric_positive = (
        "particle_spacing_m", "minimum_target_direct_impulse_ns",
        "maximum_wrench_echo_residual", "maximum_reference_point_residual_m",
        "maximum_mass_error_kg",
    )
    if any(float(probe.get(key, 0.0)) <= 0.0 for key in numeric_positive):
        raise ValueError("r3_c_routing_threshold_contract")
    if float(probe.get("maximum_inactive_direct_impulse_ns", -1.0)) < 0.0:
        raise ValueError("r3_c_routing_threshold_contract")
    for key in ("matrix_patch_size_xyz_m",
                "matrix_patch_lower_offset_from_target_rock_m"):
        value = np.asarray(probe.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("r3_c_routing_patch_contract")
        if key == "matrix_patch_size_xyz_m" and np.any(value <= 0.0):
            raise ValueError("r3_c_routing_patch_contract")
    scope = config["scope"]
    if scope != {
        "exactly_two_rocks": True,
        "per_id_newton_body_and_collider_mapping": True,
        "two_single_rock_mpm_routes": True,
        "bucket_mpm_collision": False,
        "four_round_force_partition": False,
        "feedback_qualification": False,
        "repeat_qualification": False,
        "dt_refinement_qualification": False,
        "complete_in_process_reset": False,
        "r3_c_physics_qualified": False,
        "official_envelope": False,
        "training": False,
        "frozen_r1_r2_r3_a_r3_b_and_r3_c_preflight_modified": False,
    }:
        raise ValueError("r3_c_routing_scope_expanded")
