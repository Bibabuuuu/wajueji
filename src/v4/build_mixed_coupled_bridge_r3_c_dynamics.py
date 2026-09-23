#!/usr/bin/env python3
"""Add per-rock reference and rigid-contact observability to the frozen R3-C bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_mixed_coupled_bridge_r3_c import render_bridge as render_r3_c
from build_mixed_multirock_candidate_r3_c import sha256, validate_config


READY_SCHEMA = "excavation_mixed_coupled_bridge_r3_c_dynamics_v1"
STATE_SCHEMA = "excavation_mixed_coupled_bridge_r3_c_dynamics_state_v1"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"r3_c_dynamics_bridge_anchor:{label}:{count}")
    return source.replace(old, new)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_bridge(config_path: str | Path,
                  repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    source, base_manifest = render_r3_c(config_path, repository)
    source = _replace_once(
        source,
        "// Generated isolated R3-C two-rock MPM bridge; frozen R1/R2/R3-A/R3-B are untouched.",
        "// Generated R3-C dynamics observability bridge; frozen R1/R2/R3-A/R3-B/R3-C base are untouched.",
        "header")
    source = _replace_once(
        source, "mixed_coupled_bridge_r3_c model.xml",
        "mixed_coupled_bridge_r3_c_dynamics model.xml", "usage")
    source = _replace_once(
        source, config["bridge"]["state_schema"], STATE_SCHEMA, "state_schema")
    source = _replace_once(
        source, config["bridge"]["ready_schema"], READY_SCHEMA, "ready_schema")
    source = _replace_once(
        source,
        "    std::array<double, 3> last_bucket_rock_force{};\n"
        "    std::array<double, 3> last_bucket_rock_torque{};\n",
        "    std::array<double, 3> last_bucket_rock_force{};\n"
        "    std::array<double, 3> last_bucket_rock_torque{};\n"
        "    std::array<std::array<double, 3>, 2> bucket_rock_impulse_by_rock{};\n"
        "    std::array<std::array<double, 3>, 2> bucket_rock_angular_impulse_by_rock{};\n",
        "per_rock_rigid_state")
    source = _replace_once(
        source,
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_force{};\n"
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_torque{};\n"
        "    std::array<double, 3> bucket_interval_reference_point{};\n",
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_force{};\n"
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_torque{};\n"
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_reference_point{};\n"
        "    std::array<double, 3> bucket_interval_reference_point{};\n",
        "per_rock_mpm_reference_state")
    source = _replace_once(
        source,
        "      array(\"last_bucket_rock_torque_about_bucket_origin_world_nm\",\n"
        "            last_bucket_rock_torque.data(), 3);\n",
        "      array(\"last_bucket_rock_torque_about_bucket_origin_world_nm\",\n"
        "            last_bucket_rock_torque.data(), 3);\n"
        "      std::cout << \",\\\"bucket_rigid_by_rock\\\":[\";\n"
        "      for (int index = 0; index < rock_count; ++index) {\n"
        "        if (index) std::cout << ',';\n"
        "        std::cout << \"{\\\"rock_id\\\":\\\"r1-rock-\" << std::setw(3)\n"
        "                  << std::setfill('0') << index << std::setfill(' ') << \"\\\"\";\n"
        "        array(\"bucket_impulse_world_ns\",\n"
        "              bucket_rock_impulse_by_rock[index].data(), 3);\n"
        "        array(\"bucket_angular_impulse_about_interval_reference_world_nms\",\n"
        "              bucket_rock_angular_impulse_by_rock[index].data(), 3);\n"
        "        std::cout << '}';\n"
        "      }\n"
        "      std::cout << ']';\n",
        "emit_per_rock_rigid")
    source = _replace_once(
        source,
        "        array(\"force_world_n\", applied_rock_mpm_force[index].data(), 3);\n"
        "        array(\"torque_world_nm\", applied_rock_mpm_torque[index].data(), 3);\n",
        "        array(\"force_world_n\", applied_rock_mpm_force[index].data(), 3);\n"
        "        array(\"torque_world_nm\", applied_rock_mpm_torque[index].data(), 3);\n"
        "        array(\"reference_point_world_m\",\n"
        "              applied_rock_mpm_reference_point[index].data(), 3);\n",
        "emit_per_rock_mpm_reference")
    source = _replace_once(
        source,
        "        applied_rock_mpm_force[index].fill(0);\n"
        "        applied_rock_mpm_torque[index].fill(0);\n",
        "        applied_rock_mpm_force[index].fill(0);\n"
        "        applied_rock_mpm_torque[index].fill(0);\n"
        "        applied_rock_mpm_reference_point[index].fill(0);\n",
        "reset_per_rock_mpm_reference")
    source = _replace_once(
        source,
        "          applied_rock_mpm_force[rock_index] = wrench.force;\n"
        "          applied_rock_mpm_torque[rock_index] = wrench.torque;\n",
        "          applied_rock_mpm_force[rock_index] = wrench.force;\n"
        "          applied_rock_mpm_torque[rock_index] = wrench.torque;\n"
        "          applied_rock_mpm_reference_point[rock_index] = wrench.point;\n",
        "capture_per_rock_mpm_reference")
    source = _replace_once(
        source,
        "      last_bucket_rock_force.fill(0);\n"
        "      last_bucket_rock_torque.fill(0);\n",
        "      last_bucket_rock_force.fill(0);\n"
        "      last_bucket_rock_torque.fill(0);\n"
        "      for (int index = 0; index < rock_count; ++index) {\n"
        "        bucket_rock_impulse_by_rock[index].fill(0);\n"
        "        bucket_rock_angular_impulse_by_rock[index].fill(0);\n"
        "      }\n",
        "reset_per_rock_rigid")
    source = _replace_once(
        source,
        "        std::array<double, 3> substep_bucket_contact_torque{};\n"
        "        std::array<double, 3> substep_bucket_moment_arm{};\n",
        "        std::array<double, 3> substep_bucket_contact_torque{};\n"
        "        std::array<double, 3> substep_bucket_moment_arm{};\n"
        "        std::array<std::array<double, 3>, 2>\n"
        "            substep_bucket_force_by_rock{};\n"
        "        std::array<std::array<double, 3>, 2>\n"
        "            substep_bucket_torque_by_rock{};\n",
        "substep_per_rock_rigid")
    source = _replace_once(
        source,
        "          const bool rock_is_first = first_rock >= 0;\n"
        "          const int other_geom = rock_is_first ? contact.geom2 : contact.geom1;\n",
        "          const bool rock_is_first = first_rock >= 0;\n"
        "          const int contact_rock_index =\n"
        "              rock_is_first ? first_rock : second_rock;\n"
        "          const int other_geom = rock_is_first ? contact.geom2 : contact.geom1;\n",
        "contact_rock_index")
    source = _replace_once(
        source,
        "              substep_bucket_torque[axis] +=\n"
        "                  sign * (world_torque[axis] + moment[axis]);\n",
        "              substep_bucket_torque[axis] +=\n"
        "                  sign * (world_torque[axis] + moment[axis]);\n"
        "              substep_bucket_force_by_rock[contact_rock_index][axis] +=\n"
        "                  sign * world_force[axis];\n"
        "              substep_bucket_torque_by_rock[contact_rock_index][axis] +=\n"
        "                  sign * (world_torque[axis] + moment[axis]);\n",
        "accumulate_substep_per_rock_rigid")
    source = _replace_once(
        source,
        "          bucket_rock_moment_arm_impulse[axis] +=\n"
        "              substep_bucket_moment_arm[axis] * model->opt.timestep;\n",
        "          bucket_rock_moment_arm_impulse[axis] +=\n"
        "              substep_bucket_moment_arm[axis] * model->opt.timestep;\n"
        "          for (int index = 0; index < rock_count; ++index) {\n"
        "            bucket_rock_impulse_by_rock[index][axis] +=\n"
        "                substep_bucket_force_by_rock[index][axis] * model->opt.timestep;\n"
        "            bucket_rock_angular_impulse_by_rock[index][axis] +=\n"
        "                substep_bucket_torque_by_rock[index][axis] * model->opt.timestep;\n"
        "          }\n",
        "integrate_per_rock_rigid")
    source = _replace_once(
        source, '"mixed_coupled_bridge_r3_c: "',
        '"mixed_coupled_bridge_r3_c_dynamics: "', "error_prefix")
    required = (
        "bucket_rigid_by_rock", "reference_point_world_m",
        "bucket_rock_impulse_by_rock", READY_SCHEMA, STATE_SCHEMA,
    )
    if not all(value in source for value in required):
        raise ValueError("incomplete_r3_c_dynamics_bridge_transform")
    manifest = {
        "schema": "excavation_mixed_coupled_bridge_source_manifest_r3_c_dynamics_v1",
        "status": "r3_c_dynamics_observability_bridge_generated",
        "r3_c_config_sha256": sha256(config_path),
        "r3_c_base_generated_source_sha256": _sha256_text(
            render_r3_c(config_path, repository)[0]),
        "r3_c_base_manifest": base_manifest,
        "ready_schema": READY_SCHEMA,
        "state_schema": STATE_SCHEMA,
        "required_rock_count": 2,
        "per_rock_applied_mpm_reference_point": True,
        "per_rock_bucket_rigid_impulse": True,
        "physics_qualified": False,
        "training_qualified": False,
        "frozen_r1_r2_r3_a_r3_b_r3_c_base_modified": False,
        "generated_source_sha256": _sha256_text(source),
    }
    return source, manifest


def write_bridge(config_path: str | Path, repository_root: str | Path,
                 output: str | Path) -> dict:
    source, manifest = render_bridge(config_path, repository_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    manifest = write_bridge(args.config, args.repository_root, args.output)
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
