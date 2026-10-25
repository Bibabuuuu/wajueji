"""Contract validation for the R3-C shared-matrix four-round gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_mixed_multirock_candidate_r3_c import sha256


SCHEMA = "excavation_mixed_material_r3_c_shared_four_round_config_v1"


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != SCHEMA:
        raise ValueError("invalid_r3_c_shared_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("r3_c_config_path", "r3_c_config_sha256"),
        ("routing_config_path", "routing_config_sha256"),
        ("routing_report_path", "routing_report_sha256"),
        ("dynamics_bridge_generator_path", "dynamics_bridge_generator_sha256"),
        ("r3_b_config_path", "r3_b_config_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError("r3_c_shared_frozen_input_sha256_mismatch:" + path_key)
    routing = json.loads(
        (repository / frozen["routing_report_path"]).read_text(encoding="utf-8"))
    if (routing.get("status") != "r3_c_per_rock_routing_probe_passed" or
            not routing.get("routing_physics_qualified") or
            routing.get("r3_c_physics_qualified") or
            not all(routing.get("checks", {}).values())):
        raise ValueError("r3_c_shared_routing_gate_missing")
    probe = config["probe"]
    if (probe.get("rock_ids") != ["r1-rock-000", "r1-rock-001"] or
            probe.get("qualification_rounds") !=
            ["clear", "mpm_only", "rigid_only", "combined"] or
            probe.get("command") != [0.06, 0.0, 0.0, 0.0] or
            float(probe.get("duration_s", 0.0)) != 1.0 or
            float(probe.get("output_dt_s", 0.0)) <= 0.0 or
            int(probe.get("mujoco_internal_substeps", 0)) < 20):
        raise ValueError("r3_c_shared_probe_contract")
    for key in ("matrix_patch_lower_world_m", "matrix_patch_size_xyz_m"):
        value = np.asarray(probe.get(key), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("r3_c_shared_patch_contract")
        if key.endswith("size_xyz_m") and np.any(value <= 0.0):
            raise ValueError("r3_c_shared_patch_contract")
    for key in (
        "particle_spacing_m", "minimum_active_component_impulse_ns",
        "maximum_force_partition_residual_n",
        "maximum_torque_partition_residual_nm",
        "maximum_reference_point_residual_m", "maximum_mass_error_kg",
        "maximum_closure_m",
    ):
        if float(probe.get(key, 0.0)) <= 0.0:
            raise ValueError("r3_c_shared_threshold_contract")
    if float(probe.get("maximum_inactive_component_impulse_ns", -1.0)) < 0.0:
        raise ValueError("r3_c_shared_threshold_contract")
    scope = config["scope"]
    if scope != {
        "exactly_two_rocks": True,
        "one_shared_mpm_matrix": True,
        "bucket_and_both_rocks_mpm_collision_in_combined": True,
        "four_round_force_partition": True,
        "per_rock_rigid_bucket_accounting": True,
        "per_rock_mpm_reference_echo": True,
        "repeat_qualification": False,
        "dt_refinement_qualification": False,
        "complete_in_process_reset": False,
        "r3_c_physics_qualified": False,
        "official_envelope": False,
        "training": False,
        "frozen_prior_gates_modified": False,
    }:
        raise ValueError("r3_c_shared_scope_expanded")
