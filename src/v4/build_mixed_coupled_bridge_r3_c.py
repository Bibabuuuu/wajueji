#!/usr/bin/env python3
"""Derive the isolated two-rock R3-C bridge from the frozen R3-B generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_mixed_coupled_bridge_r3_b import render_bridge as render_r3_b
from build_mixed_multirock_candidate_r3_c import sha256, validate_config


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"r3_c_bridge_transform_anchor:{label}:{count}")
    return source.replace(old, new)


def _replace_between_once(source: str, start: str, end: str,
                          replacement: str, label: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"r3_c_bridge_transform_range:{label}")
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[:begin] + replacement + source[finish:]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_bridge(config_path: str | Path,
                  repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    source, r3_b_manifest = render_r3_b(
        repository / frozen["r3_b_config_path"], repository)
    source = _replace_once(
        source,
        "// Generated isolated R3-B bucket/rock MPM bridge; frozen R1/R2/R3-A are untouched.",
        "// Generated isolated R3-C two-rock MPM bridge; frozen R1/R2/R3-A/R3-B are untouched.",
        "header")
    source = _replace_once(
        source, "mixed_coupled_bridge_r3_b model.xml",
        "mixed_coupled_bridge_r3_c model.xml", "usage")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_b_state_v1",
        config["bridge"]["state_schema"], "state_schema")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_b_v1",
        config["bridge"]["ready_schema"], "ready_schema")
    source = _replace_once(
        source, "rock_count < 1 || rock_count > 2 ||", "rock_count != 2 ||",
        "exact_two_rocks")
    source = _replace_once(
        source,
        "    std::array<double, 3> applied_rock_mpm_force{};\n"
        "    std::array<double, 3> applied_rock_mpm_torque{};\n",
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_force{};\n"
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_torque{};\n",
        "per_rock_applied_state")

    old_rock_loop = '''      std::cout << ",\\"rocks\\":[";
      for (int index = 0; index < rock_count; ++index) {
        if (index) std::cout << ',';
        std::cout << "{\\"rock_id\\":\\"r1-rock-" << std::setw(3)
                  << std::setfill('0') << index << std::setfill(' ') << "\\"";
        array("qpos", data->qpos + 19 + 7 * index, 7);
        array("qvel", data->qvel + 18 + 6 * index, 6);
        array("position", data->xpos + 3 * rock_bodies[index], 3);
        std::cout << '}';
      }
      std::cout << ']';
'''
    new_rock_loop = '''      std::cout << ",\\"rocks\\":[";
      for (int index = 0; index < rock_count; ++index) {
        if (index) std::cout << ',';
        std::cout << "{\\"rock_id\\":\\"r1-rock-" << std::setw(3)
                  << std::setfill('0') << index << std::setfill(' ') << "\\"";
        array("qpos", data->qpos + 19 + 7 * index, 7);
        array("qvel", data->qvel + 18 + 6 * index, 6);
        array("position", data->xpos + 3 * rock_bodies[index], 3);
        mjtNum rock_world_velocity[6] = {};
        mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY,
                          rock_bodies[index], rock_world_velocity, 0);
        array("linear_velocity_world_m_s", rock_world_velocity + 3, 3);
        array("angular_velocity_world_rad_s", rock_world_velocity, 3);
        std::cout << '}';
      }
      std::cout << ']';
'''
    source = _replace_once(
        source, old_rock_loop, new_rock_loop, "per_rock_world_velocity")

    source = _replace_once(
        source,
        '      array("applied_rock_mpm_force_world_n", applied_rock_mpm_force.data(), 3);\n'
        '      array("applied_rock_mpm_torque_world_nm", applied_rock_mpm_torque.data(), 3);\n',
        '''      std::cout << ",\\"applied_rocks_mpm\\":[";
      for (int index = 0; index < rock_count; ++index) {
        if (index) std::cout << ',';
        std::cout << "{\\"rock_id\\":\\"r1-rock-" << std::setw(3)
                  << std::setfill('0') << index << std::setfill(' ') << "\\"";
        array("force_world_n", applied_rock_mpm_force[index].data(), 3);
        array("torque_world_nm", applied_rock_mpm_torque[index].data(), 3);
        std::cout << '}';
      }
      std::cout << ']';
''',
        "emit_per_rock_applied")

    ready_old = (
        '              << ",\\"maximum_external_wrench_count\\":2"\n'
        '              << ",\\"allowed_external_wrench_bodies\\":[\\"bucket\\",\\"r1-rock-000\\"]"\n')
    ready_new = (
        '              << ",\\"maximum_external_wrench_count\\":3"\n'
        '              << ",\\"allowed_external_wrench_bodies\\":[\\"bucket\\",\\"r1-rock-000\\",\\"r1-rock-001\\"]"\n')
    source = _replace_once(source, ready_old, ready_new, "ready_wrench_contract")

    parse_start = "      const double external_wrench_count_value = number();\n"
    parse_end = (
        "      std::string extra;\n"
        "      if (input >> extra) throw std::runtime_error(\"extra step fields\");\n")
    parse_replacement = '''      const double external_wrench_count_value = number();
      if (external_wrench_count_value < 0 || external_wrench_count_value > 3 ||
          external_wrench_count_value != int(external_wrench_count_value)) {
        throw std::runtime_error("R3-C external wrench count contract");
      }
      external_wrenches.clear();
      applied_bucket_mpm_force.fill(0);
      applied_bucket_mpm_torque.fill(0);
      for (int index = 0; index < rock_count; ++index) {
        applied_rock_mpm_force[index].fill(0);
        applied_rock_mpm_torque[index].fill(0);
      }
      bool seen_bucket = false;
      std::array<bool, 2> seen_rocks{};
      for (int wrench_index = 0;
           wrench_index < int(external_wrench_count_value); ++wrench_index) {
        std::string body_name;
        if (!(input >> body_name)) throw std::runtime_error("missing R3-C wrench body");
        ExternalWrench wrench;
        int rock_index = -1;
        if (body_name == "bucket") {
          if (seen_bucket) throw std::runtime_error("duplicate bucket wrench");
          seen_bucket = true;
          wrench.body = bucket_body;
        } else if (body_name == "r1-rock-000") {
          rock_index = 0;
        } else if (body_name == "r1-rock-001") {
          rock_index = 1;
        } else {
          throw std::runtime_error("unapproved R3-C wrench body");
        }
        if (rock_index >= 0) {
          if (rock_index >= rock_count || seen_rocks[rock_index]) {
            throw std::runtime_error("duplicate or absent R3-C rock wrench");
          }
          seen_rocks[rock_index] = true;
          wrench.body = rock_bodies[rock_index];
        }
        for (double& value : wrench.force) value = number();
        for (double& value : wrench.torque) value = number();
        for (double& value : wrench.point) value = number();
        if (rock_index < 0) {
          applied_bucket_mpm_force = wrench.force;
          applied_bucket_mpm_torque = wrench.torque;
        } else {
          applied_rock_mpm_force[rock_index] = wrench.force;
          applied_rock_mpm_torque[rock_index] = wrench.torque;
        }
        external_wrenches.push_back(wrench);
      }
'''
    source = _replace_between_once(
        source, parse_start, parse_end, parse_replacement,
        "parse_three_named_wrenches")
    source = _replace_once(
        source, '"mixed_coupled_bridge_r3_b: "',
        '"mixed_coupled_bridge_r3_c: "', "error_prefix")
    if ("maximum_external_wrench_count\\\":2" in source or
            "allowed_external_wrench_bodies\\\":[\\\"bucket\\\",\\\"r1-rock-000\\\"]" in source or
            "applied_rock_mpm_force_world_n" in source):
        raise ValueError("incomplete_r3_c_bridge_transform")
    manifest = {
        "schema": "excavation_mixed_coupled_bridge_source_manifest_r3_c_v1",
        "status": "r3_c_two_rock_bridge_source_generated",
        "r3_c_config_sha256": sha256(config_path),
        "r3_b_config_path": frozen["r3_b_config_path"],
        "r3_b_config_sha256": sha256(repository / frozen["r3_b_config_path"]),
        "r3_b_generated_source_sha256": _sha256_text(
            render_r3_b(repository / frozen["r3_b_config_path"], repository)[0]),
        "r3_b_manifest": r3_b_manifest,
        "ready_schema": config["bridge"]["ready_schema"],
        "state_schema": config["bridge"]["state_schema"],
        "required_rock_count": 2,
        "maximum_external_wrench_count": 3,
        "allowed_external_wrench_bodies":
            config["bridge"]["allowed_external_wrench_bodies"],
        "frozen_r1_r2_r3_a_r3_b_modified": False,
        "physics_qualified": False,
        "training_qualified": False,
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
            encoding="utf-8")
    else:
        print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
