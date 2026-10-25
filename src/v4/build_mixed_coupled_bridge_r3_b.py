"""Derive the isolated R3-B body-origin wrench bridge from frozen R3-A logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_mixed_coupled_bridge_r3_a import render_bridge as render_r3_a
from build_mixed_coupled_bridge_r3_a import sha256


CONFIG_SCHEMA = "excavation_mixed_material_r3_b_config_v1"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"r3_b_bridge_transform_anchor:{label}:{count}")
    return source.replace(old, new)


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_r3_b_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("r3_a_config_path", "r3_a_config_sha256"),
        ("r3_a_bridge_generator_path", "r3_a_bridge_generator_sha256"),
        ("r1_config_path", "r1_config_sha256"),
        ("mechanism_model_path", "mechanism_model_sha256"),
        ("rock_asset_path", "rock_asset_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError("frozen_input_sha256_mismatch:" + path_key)
    if config["bridge"] != {
        "ready_schema": "excavation_mixed_coupled_bridge_r3_b_v1",
        "state_schema": "excavation_mixed_coupled_bridge_r3_b_state_v1",
        "allowed_external_wrench_bodies": ["bucket", "r1-rock-000"],
        "external_wrench_frame": "world",
        "external_force_application": (
            "bucket_origin_at_mujoco_interval_start_fixed_for_interval"),
        "external_torque_reference": (
            "bucket_origin_at_mujoco_interval_start_fixed_for_interval"),
        "coupling_schedule": (
            "previous_mpm_interval_wrench_applied_to_next_mujoco_interval"),
    }:
        raise ValueError("invalid_r3_b_bridge_contract")
    scope = config["scope"]
    if (not scope["one_frozen_r0_rock"] or
            not scope["small_mpm_patch"] or
            not scope["bucket_mpm_contact_enabled"] or
            not scope["per_body_mpm_wrench"] or
            not scope["three_component_bucket_wrench_accounting"] or
            scope["multi_rock"] or scope["official_envelope"] or
            scope["training"] or scope["frozen_r1_r2_or_r3_a_modified"]):
        raise ValueError("r3_b_scope_expanded")
    probe = config["probe"]
    if (probe["r1_scenario"] != "bucket_push" or
            probe["collider_velocity_mode"] != "backward" or
            probe["collider_velocity_semantics"] !=
            "previous_to_current_outer_step_pose_difference" or
            probe["command"] != [0.06, 0.0, 0.0, 0.0] or
            float(probe["duration_s"]) != 1.0):
        raise ValueError("r3_b_probe_contract")
    if probe["qualification_rounds"] != {
        "clear": "clear_rigid_model_with_bucket_and_rock_mpm_collision_disabled",
        "mpm_only": "clear_rigid_model_with_bucket_mpm_collision_only",
        "rigid_only": "push_rigid_model_with_bucket_and_rock_mpm_collision_disabled",
        "combined": "push_rigid_model_with_bucket_and_rock_mpm_collision_enabled",
    }:
        raise ValueError("r3_b_qualification_round_contract")
    if probe["dt_refinement_acceptance_impulses"] != [
            "bucket_mpm_ns", "bucket_measured_total_ns"]:
        raise ValueError("r3_b_dt_refinement_acceptance_contract")


def render_bridge(config_path: str | Path,
                  repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    source, r3_a_manifest = render_r3_a(
        repository / frozen["r3_a_config_path"], repository)
    source = _replace_once(
        source,
        "// Generated isolated R3-A one-rock/MPM bridge; frozen R1/R2 bridges are untouched.",
        "// Generated isolated R3-B bucket/rock MPM bridge; frozen R1/R2/R3-A are untouched.",
        "header",
    )
    source = _replace_once(
        source, "mixed_coupled_bridge_r3_a model.xml",
        "mixed_coupled_bridge_r3_b model.xml", "usage")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_a_state_v1",
        "excavation_mixed_coupled_bridge_r3_b_state_v1", "state_schema")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_a_v1",
        "excavation_mixed_coupled_bridge_r3_b_v1", "ready_schema")
    source = _replace_once(
        source,
        "named_world_wrench_about_fixed_world_point",
        "named_world_wrench_about_interval_start_body_origin",
        "wrench_semantics",
    )
    bucket_velocity = """
      mjtNum bucket_velocity[6] = {};
      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, bucket_body,
                        bucket_velocity, 0);
      array("bucket_linear_velocity_world_m_s", bucket_velocity + 3, 3);
      array("bucket_angular_velocity_world_rad_s", bucket_velocity, 3);
"""
    source = _replace_once(
        source,
        "      mjtNum rock_velocity[6] = {};\n",
        bucket_velocity + "      mjtNum rock_velocity[6] = {};\n",
        "bucket_velocity",
    )
    source = _replace_once(
        source,
        "    std::array<double, 3> applied_rock_mpm_torque{};\n",
        "    std::array<double, 3> applied_rock_mpm_torque{};\n"
        "    std::array<double, 3> bucket_interval_reference_point{};\n",
        "interval_reference_state",
    )
    source = _replace_once(
        source,
        '      array("applied_rock_mpm_torque_world_nm", applied_rock_mpm_torque.data(), 3);\n',
        '      array("applied_rock_mpm_torque_world_nm", applied_rock_mpm_torque.data(), 3);\n'
        '      array("bucket_interval_reference_point_world_m",\n'
        '            bucket_interval_reference_point.data(), 3);\n',
        "emit_interval_reference",
    )
    source = _replace_once(
        source,
        "        ready = true;\n"
        "        emit();\n",
        "        ready = true;\n"
        "        std::copy(data->xpos + 3 * bucket_body,\n"
        "                  data->xpos + 3 * bucket_body + 3,\n"
        "                  bucket_interval_reference_point.begin());\n"
        "        emit();\n",
        "initial_interval_reference",
    )
    source = _replace_once(
        source,
        "      rock_ground_contact_substeps = rock_ramp_contact_substeps = 0;\n",
        "      std::copy(data->xpos + 3 * bucket_body,\n"
        "                data->xpos + 3 * bucket_body + 3,\n"
        "                bucket_interval_reference_point.begin());\n"
        "      rock_ground_contact_substeps = rock_ramp_contact_substeps = 0;\n",
        "step_interval_reference",
    )
    source = _replace_once(
        source,
        "            const mjtNum* bucket_origin = data->xpos + 3 * bucket_body;\n",
        "            const double* bucket_origin =\n"
        "                bucket_interval_reference_point.data();\n",
        "rigid_torque_reference",
    )
    source = _replace_once(
        source, '"mixed_coupled_bridge_r3_a: "',
        '"mixed_coupled_bridge_r3_b: "', "error_prefix")
    manifest = {
        "schema": "excavation_mixed_coupled_bridge_source_manifest_r3_b_v1",
        "status": "r3_b_bridge_source_generated",
        "r3_a_config_path": frozen["r3_a_config_path"],
        "r3_a_config_sha256": sha256(repository / frozen["r3_a_config_path"]),
        "r3_a_generated_source_sha256": sha256_bytes(
            render_r3_a(repository / frozen["r3_a_config_path"], repository)[0]),
        "r3_a_manifest": r3_a_manifest,
        "ready_schema": config["bridge"]["ready_schema"],
        "state_schema": config["bridge"]["state_schema"],
        "force_application": config["bridge"]["external_force_application"],
        "torque_reference": config["bridge"]["external_torque_reference"],
        "frozen_r1_r2_or_r3_a_modified": False,
    }
    return source, manifest


def sha256_bytes(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_bridge(config_path: str | Path, repository_root: str | Path,
                 output_source: str | Path) -> dict:
    rendered, manifest = render_bridge(config_path, repository_root)
    output = Path(output_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    manifest["generated_source_sha256"] = sha256(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-source", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    manifest = write_bridge(
        args.config, args.repository_root, args.output_source)
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    else:
        print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
