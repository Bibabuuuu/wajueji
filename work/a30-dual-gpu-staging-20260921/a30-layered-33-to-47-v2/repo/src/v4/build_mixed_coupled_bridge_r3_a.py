"""Derive an isolated one-rock R3-A bridge from the frozen R1 bridge.

The generated bridge adds only a bounded, named per-body external-wrench
protocol.  The R1 source remains byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONFIG_SCHEMA = "excavation_mixed_material_r3_a_config_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"r3_bridge_transform_anchor:{label}:{count}")
    return source.replace(old, new)


def validate_config(config: dict, repository_root: str | Path) -> None:
    repository = Path(repository_root).resolve()
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_r3_a_config_schema")
    frozen = config["frozen_inputs"]
    for path_key, hash_key in (
        ("r1_config_path", "r1_config_sha256"),
        ("r1_bridge_source_path", "r1_bridge_source_sha256"),
        ("mechanism_model_path", "mechanism_model_sha256"),
        ("rock_asset_path", "rock_asset_sha256"),
    ):
        path = repository / frozen[path_key]
        if not path.is_file() or sha256(path) != frozen[hash_key]:
            raise ValueError("frozen_input_sha256_mismatch:" + path_key)
    bridge = config["bridge"]
    if bridge != {
        "ready_schema": "excavation_mixed_coupled_bridge_r3_a_v1",
        "state_schema": "excavation_mixed_coupled_bridge_r3_a_state_v1",
        "rock_body_name": "r1-rock-000",
        "maximum_external_wrench_count": 2,
        "allowed_external_wrench_bodies": ["bucket", "r1-rock-000"],
        "external_wrench_frame": "world",
        "external_torque_reference": "supplied_fixed_world_point",
        "coupling_schedule": (
            "previous_mpm_interval_wrench_applied_to_next_mujoco_interval"),
    }:
        raise ValueError("invalid_r3_a_bridge_contract")
    scope = config["scope"]
    if (not scope["one_frozen_r0_rock"] or
            not scope["small_mpm_matrix"] or
            not scope["per_body_mpm_wrench"] or
            scope["bucket_mpm_contact_enabled"] or scope["multi_rock"] or
            scope["official_envelope"] or scope["training"] or
            scope["final_rock_count_selected"] or
            scope["frozen_r1_or_r2_modified"]):
        raise ValueError("r3_a_scope_expanded")


def render_bridge(config_path: str | Path,
                  repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    bridge = config["bridge"]
    source_path = repository / frozen["r1_bridge_source_path"]
    source = source_path.read_text(encoding="utf-8")

    source = _replace_once(
        source,
        "// Isolated one-rock MuJoCo bridge.  The legacy 19/18/12 bridge is untouched.",
        "// Generated isolated R3-A one-rock/MPM bridge; frozen R1/R2 bridges are untouched.",
        "header",
    )
    source = _replace_once(
        source,
        '"usage: mixed_rigid_bridge model.xml"',
        '"usage: mixed_coupled_bridge_r3_a model.xml"',
        "usage",
    )
    source = _replace_once(
        source,
        '"@STATE {\\"schema\\":\\"excavation_mixed_rigid_bridge_state_v1\\""',
        '"@STATE {\\"schema\\":\\"excavation_mixed_coupled_bridge_r3_a_state_v1\\""',
        "state_schema",
    )
    source = _replace_once(
        source,
        '"@READY {\\"schema\\":\\"excavation_mixed_rigid_bridge_v1\\""',
        '"@READY {\\"schema\\":\\"excavation_mixed_coupled_bridge_r3_a_v1\\""',
        "ready_schema",
    )
    source = _replace_once(
        source,
        '<< ",\\"rock_count\\":" << rock_count\n'
        '              << ",\\"external_wrench_input\\":\\"reserved_zero_only\\"}"',
        '<< ",\\"rock_count\\":" << rock_count\n'
        '              << ",\\"maximum_external_wrench_count\\":2"\n'
        '              << ",\\"allowed_external_wrench_bodies\\":[\\"bucket\\",\\"r1-rock-000\\"]"\n'
        '              << ",\\"external_wrench_input\\":\\"named_world_wrench_about_fixed_world_point\\"}"',
        "ready_contract",
    )

    state_declarations = """
    struct ExternalWrench {
      int body = -1;
      std::array<double, 3> force{};
      std::array<double, 3> torque{};
      std::array<double, 3> point{};
    };
    std::vector<ExternalWrench> external_wrenches;
    std::array<double, 3> applied_bucket_mpm_force{};
    std::array<double, 3> applied_bucket_mpm_torque{};
    std::array<double, 3> applied_rock_mpm_force{};
    std::array<double, 3> applied_rock_mpm_torque{};
"""
    source = _replace_once(
        source,
        "    std::array<double, 3> last_bucket_rock_torque{};\n",
        "    std::array<double, 3> last_bucket_rock_torque{};\n" + state_declarations,
        "external_state",
    )

    emit_fields = """
      mjtNum rock_velocity[6] = {};
      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, rock_bodies[0],
                        rock_velocity, 0);
      array("rock_linear_velocity_world_m_s", rock_velocity + 3, 3);
      array("rock_angular_velocity_world_rad_s", rock_velocity, 3);
      array("controller_target", controller.target.data(), 4);
      array("controller_target_velocity", controller.target_velocity.data(), 4);
      array("applied_bucket_mpm_force_world_n", applied_bucket_mpm_force.data(), 3);
      array("applied_bucket_mpm_torque_world_nm", applied_bucket_mpm_torque.data(), 3);
      array("applied_rock_mpm_force_world_n", applied_rock_mpm_force.data(), 3);
      array("applied_rock_mpm_torque_world_nm", applied_rock_mpm_torque.data(), 3);
"""
    source = _replace_once(
        source,
        '      array("last_bucket_rock_torque_about_bucket_origin_world_nm",\n'
        '            last_bucket_rock_torque.data(), 3);\n',
        '      array("last_bucket_rock_torque_about_bucket_origin_world_nm",\n'
        '            last_bucket_rock_torque.data(), 3);\n' + emit_fields,
        "emit_external_state",
    )

    parse_wrenches = """
      const double external_wrench_count_value = number();
      if (external_wrench_count_value < 0 || external_wrench_count_value > 2 ||
          external_wrench_count_value != int(external_wrench_count_value)) {
        throw std::runtime_error("R3-A external wrench count contract");
      }
      external_wrenches.clear();
      applied_bucket_mpm_force.fill(0);
      applied_bucket_mpm_torque.fill(0);
      applied_rock_mpm_force.fill(0);
      applied_rock_mpm_torque.fill(0);
      bool seen_bucket = false, seen_rock = false;
      for (int wrench_index = 0;
           wrench_index < int(external_wrench_count_value); ++wrench_index) {
        std::string body_name;
        if (!(input >> body_name)) throw std::runtime_error("missing R3-A wrench body");
        ExternalWrench wrench;
        if (body_name == "bucket") {
          if (seen_bucket) throw std::runtime_error("duplicate bucket wrench");
          seen_bucket = true;
          wrench.body = bucket_body;
        } else if (body_name == "r1-rock-000") {
          if (seen_rock) throw std::runtime_error("duplicate rock wrench");
          seen_rock = true;
          wrench.body = rock_bodies[0];
        } else {
          throw std::runtime_error("unapproved R3-A wrench body");
        }
        for (double& value : wrench.force) value = number();
        for (double& value : wrench.torque) value = number();
        for (double& value : wrench.point) value = number();
        if (body_name == "bucket") {
          applied_bucket_mpm_force = wrench.force;
          applied_bucket_mpm_torque = wrench.torque;
        } else {
          applied_rock_mpm_force = wrench.force;
          applied_rock_mpm_torque = wrench.torque;
        }
        external_wrenches.push_back(wrench);
      }
"""
    source = _replace_once(
        source,
        "      const double external_wrench_count = number();\n"
        "      if (external_wrench_count != 0) {\n"
        "        throw std::runtime_error(\"R1 external wrench input is reserved and must be zero\");\n"
        "      }\n",
        parse_wrenches,
        "parse_external_wrenches",
    )

    apply_wrenches = """
        mju_zero(data->qfrc_applied, model->nv);
        for (const auto& wrench : external_wrenches) {
          mj_applyFT(model.get(), data.get(), wrench.force.data(),
                     wrench.torque.data(), wrench.point.data(), wrench.body,
                     data->qfrc_applied);
        }
"""
    source = _replace_once(
        source,
        "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());\n",
        apply_wrenches + "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());\n",
        "apply_external_wrenches",
    )
    source = _replace_once(
        source,
        '"mixed_rigid_bridge: "',
        '"mixed_coupled_bridge_r3_a: "',
        "error_prefix",
    )
    if "reserved_zero_only" in source or "R1 external wrench input" in source:
        raise ValueError("incomplete_r3_bridge_transform")
    manifest = {
        "schema": "excavation_mixed_coupled_bridge_source_manifest_r3_a_v1",
        "status": "r3_a_bridge_source_generated",
        "base_r1_source_path": frozen["r1_bridge_source_path"],
        "base_r1_source_sha256": sha256(source_path),
        "ready_schema": bridge["ready_schema"],
        "state_schema": bridge["state_schema"],
        "maximum_external_wrench_count": 2,
        "allowed_external_wrench_bodies": bridge["allowed_external_wrench_bodies"],
        "frozen_r1_source_modified": False,
    }
    return source, manifest


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
            encoding="utf-8",
        )
    else:
        print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
