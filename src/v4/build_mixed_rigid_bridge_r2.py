"""Derive the R2 1--20 rock bridge source without modifying the frozen R1 bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixed_material_r2_mechanism import _validate_config, sha256


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"bridge_transform_anchor:{label}:{count}")
    return source.replace(old, new)


def render_bridge(config_path: str | Path,
                  repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    bridge = config["bridge_candidate"]
    base_path = repository / bridge["base_r1_source_path"]
    if sha256(base_path) != bridge["base_r1_source_sha256"]:
        raise ValueError("base_r1_bridge_sha256_mismatch")
    source = base_path.read_text(encoding="utf-8")
    replacements = [
        (
            "// Isolated one-rock MuJoCo bridge.  The legacy 19/18/12 bridge is untouched.",
            "// Generated R2 bridge for one to twenty rocks; R1 source remains untouched.",
            "header",
        ),
        (
            '"usage: mixed_rigid_bridge model.xml"',
            '"usage: mixed_rigid_bridge_r2 model.xml"',
            "usage",
        ),
        ("rock_count > 2", "rock_count > 20", "maximum_count"),
        (
            '"model contract: MuJoCo 238, mechanism 19/18/12 plus one or two rocks"',
            '"model contract: MuJoCo 238, mechanism 19/18/12 plus one to twenty rocks"',
            "model_contract",
        ),
        ('"r1-rock-%03d"', '"r2-rock-%03d"', "body_name"),
        ('"r1_rock_free_%03d"', '"r2_rock_free_%03d"', "joint_name"),
        (
            '"@STATE {\\"schema\\":\\"excavation_mixed_rigid_bridge_state_v1\\""',
            '"@STATE {\\"schema\\":\\"excavation_mixed_rigid_bridge_state_v2\\""',
            "state_schema",
        ),
        (
            '<< ",\\"time_s\\":" << data->time;',
            '<< ",\\"time_s\\":" << data->time\n'
            '                << ",\\"rock_count\\":" << rock_count\n'
            '                << ",\\"maximum_rock_capacity\\":20";',
            "state_count_fields",
        ),
        ('"r1-rock-"', '"r2-rock-"', "state_rock_id"),
        (
            '"@READY {\\"schema\\":\\"excavation_mixed_rigid_bridge_v1\\""',
            '"@READY {\\"schema\\":\\"excavation_mixed_rigid_bridge_v2\\""',
            "ready_schema",
        ),
        (
            '<< ",\\"rock_count\\":" << rock_count\n'
            '              << ",\\"external_wrench_input\\":\\"reserved_zero_only\\"}"',
            '<< ",\\"rock_count\\":" << rock_count\n'
            '              << ",\\"maximum_rock_capacity\\":20"\n'
            '              << ",\\"runtime_add_remove_allowed\\":false"\n'
            '              << ",\\"placeholder_rocks_allowed\\":false"\n'
            '              << ",\\"external_wrench_input\\":\\"reserved_zero_only\\"}"',
            "ready_capacity_fields",
        ),
        (
            '"R1 external wrench input is reserved and must be zero"',
            '"R2 external wrench input is reserved and must be zero"',
            "external_wrench_error",
        ),
        ('"r1_bucket_"', '"r2_bucket_"', "bucket_prefix"),
        ('"r1_ground_"', '"r2_ground_"', "ground_prefix"),
        (
            '} else if (StartsWith(other_name, "r2_ground_")) {\n'
            '            ground_this_substep = true;\n'
            '            ramp_this_substep = true;',
            '} else if (StartsWith(other_name, "r2_surface_")) {\n'
            '            ground_this_substep = true;\n'
            '          } else if (StartsWith(other_name, "r2_ground_")) {\n'
            '            ground_this_substep = true;\n'
            '            ramp_this_substep = true;',
            "surface_contact_role",
        ),
        ('"r1_truck_"', '"r2_truck_"', "truck_prefix"),
        (
            '"mixed_rigid_bridge: "',
            '"mixed_rigid_bridge_r2: "',
            "error_prefix",
        ),
    ]
    rendered = source
    for old, new, label in replacements:
        rendered = _replace_once(rendered, old, new, label)
    if ("r1-rock" in rendered or '"r1_' in rendered or
            "rock_count > 2 ||" in rendered):
        raise ValueError("incomplete_r2_bridge_transform")
    manifest = {
        "schema": "excavation_mixed_rigid_bridge_source_manifest_v2",
        "status": "r2_bridge_source_generated",
        "base_r1_source_path": bridge["base_r1_source_path"],
        "base_r1_source_sha256": sha256(base_path),
        "ready_schema": bridge["ready_schema"],
        "state_schema": bridge["state_schema"],
        "minimum_rock_count": 1,
        "maximum_rock_count": bridge["maximum_rock_count"],
        "actual_count_from_loaded_model": True,
        "runtime_add_remove_allowed": False,
        "placeholder_rocks_allowed": False,
        "replacement_anchor_count": len(replacements),
    }
    return rendered, manifest


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
