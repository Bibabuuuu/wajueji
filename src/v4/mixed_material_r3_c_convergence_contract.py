"""Contract validation for the R3-C repeat, refinement, and feedback gate."""

from __future__ import annotations

import json
from pathlib import Path

from build_mixed_multirock_candidate_r3_c import sha256


SCHEMA = "excavation_mixed_material_r3_c_convergence_config_v1"


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != SCHEMA:
        raise ValueError("invalid_r3_c_convergence_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("shared_config_path", "shared_config_sha256"),
        ("shared_report_path", "shared_report_sha256"),
        ("shared_runner_path", "shared_runner_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError(
                "r3_c_convergence_frozen_input_sha256_mismatch:" + path_key)
    shared = json.loads(
        (repository / frozen["shared_report_path"]).read_text(encoding="utf-8"))
    if (shared.get("status") != "r3_c_shared_matrix_four_round_passed" or
            not shared.get("shared_four_round_qualified") or
            shared.get("r3_c_physics_qualified") or
            not all(shared.get("checks", {}).values())):
        raise ValueError("r3_c_convergence_shared_gate_missing")
    probe = config["probe"]
    if (probe.get("qualification_mode") != "combined" or
            float(probe.get("coarse_output_dt_s", 0.0)) != 1.0 / 120.0 or
            float(probe.get("refined_output_dt_s", 0.0)) != 1.0 / 240.0 or
            int(probe.get("coarse_mujoco_internal_substeps", 0)) != 40 or
            int(probe.get("refined_mujoco_internal_substeps", 0)) != 20 or
            float(probe.get("duration_s", 0.0)) != 1.0):
        raise ValueError("r3_c_convergence_probe_contract")
    if probe.get("dt_acceptance_impulses") != [
        "bucket_mpm_ns", "bucket_total_ns",
        "rock_mpm_ns.r1-rock-000", "rock_mpm_ns.r1-rock-001",
    ]:
        raise ValueError("r3_c_convergence_impulse_contract")
    for key in (
        "minimum_feedback_mechanism_qpos_difference_m",
        "maximum_feedback_mechanism_qpos_difference_m",
        "maximum_repeat_state_difference",
        "maximum_repeat_impulse_relative_difference",
        "maximum_dt_state_difference",
        "maximum_dt_impulse_relative_difference",
    ):
        if float(probe.get(key, 0.0)) <= 0.0:
            raise ValueError("r3_c_convergence_threshold_contract")
    if (float(probe["minimum_feedback_mechanism_qpos_difference_m"]) >=
            float(probe["maximum_feedback_mechanism_qpos_difference_m"])):
        raise ValueError("r3_c_convergence_threshold_contract")
    if config["scope"] != {
        "exactly_two_rocks": True,
        "one_shared_mpm_matrix": True,
        "reconstructed_repeat_qualification": True,
        "dt_refinement_qualification": True,
        "feedback_causality_qualification": True,
        "complete_in_process_reset": False,
        "r3_c_physics_qualification_candidate": True,
        "official_envelope": False,
        "training": False,
        "frozen_prior_gates_modified": False,
    }:
        raise ValueError("r3_c_convergence_scope_expanded")
