#!/usr/bin/env python3
"""Build variable-count MuJoCo models and a named-wrench bridge for capacity preflight."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


CONFIG_SCHEMA = "excavation_mixed_material_capacity_preflight_config_v1"
ROCK_NAME_RE = re.compile(r"^r1-rock-(\d{3,4})$")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"capacity_transform_anchor:{label}:{count}")
    return source.replace(old, new)


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != (
            "capacity_protocol_preflight_only"):
        raise ValueError("capacity_config_schema_or_status")
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, dict) or set(frozen) != {
            "two_rock_model", "two_rock_bridge", "three_cycle_config"}:
        raise ValueError("capacity_frozen_input_set")
    for name, binding in frozen.items():
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                or not isinstance(binding["path"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])):
            raise ValueError(f"capacity_frozen_input_format:{name}")
        path = repository / binding["path"]
        if not path.is_file() or sha256(path) != binding["sha256"]:
            raise ValueError(f"capacity_frozen_input_mismatch:{name}")
    capacity = config.get("capacity")
    if capacity != {
        "preflight_rock_counts": [64, 256, 512],
        "maximum_formal_rock_count": 1024,
        "runtime_count_source": "loaded_model_contiguous_named_freejoints",
        "fixed_after_startup": True,
        "maximum_external_wrench_count": "rock_count_plus_bucket",
    }:
        raise ValueError("capacity_contract_drift")
    layout = config.get("synthetic_layout")
    if layout != {
        "equivalent_diameter_m": 0.16,
        "origin_world_m": [3.8875, -1.0125, 0.0875],
        "spacing_m": 0.175,
        "columns_x": 12,
        "columns_y": 12,
        "maximum_layers_z": 8,
        "purpose": "time_zero_capacity_only_not_formal_scene_geometry",
    }:
        raise ValueError("capacity_layout_drift")
    if config.get("protocol") != {
        "ready_schema": "excavation_mixed_coupled_bridge_capacity_v1",
        "state_schema": "excavation_mixed_coupled_bridge_capacity_state_v1",
        "rock_id_format": "r1-rock-%03d",
        "rock_joint_format": "r1_rock_free_%03d",
        "external_wrench_frame": "world",
        "named_wrench_duplicate_unknown_or_absent_id_rejected": True,
    }:
        raise ValueError("capacity_protocol_drift")
    if config.get("scope") != {
        "model_generation": True,
        "bridge_generation": True,
        "bridge_compile": True,
        "time_zero_protocol": True,
        "one_clear_step": False,
        "mpm_or_gpu_run": False,
        "settling": False,
        "physics_qualified": False,
        "formal_scene_qualified": False,
        "training": False,
        "frozen_inputs_modified": False,
    }:
        raise ValueError("capacity_scope_expanded")
    return config


def _numbers(values: list[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _synthetic_position(index: int, layout: dict) -> list[float]:
    columns_x = int(layout["columns_x"])
    columns_y = int(layout["columns_y"])
    layer_size = columns_x * columns_y
    layer = index // layer_size
    remainder = index % layer_size
    row = remainder // columns_x
    column = remainder % columns_x
    if layer >= int(layout["maximum_layers_z"]):
        raise ValueError("capacity_synthetic_layout_exhausted")
    origin = layout["origin_world_m"]
    spacing = float(layout["spacing_m"])
    return [
        float(origin[0]) + spacing * column,
        float(origin[1]) + spacing * row,
        float(origin[2]) + spacing * layer,
    ]


def build_capacity_model(config_path: str | Path,
                         repository_root: str | Path,
                         rock_count: int,
                         output_path: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_and_validate_config(config_path, repository)
    maximum = int(config["capacity"]["maximum_formal_rock_count"])
    if (isinstance(rock_count, bool) or not isinstance(rock_count, int)
            or rock_count < 1 or rock_count > maximum):
        raise ValueError("capacity_rock_count_out_of_range")
    source_path = repository / config["frozen_inputs"]["two_rock_model"]["path"]
    tree = ET.parse(source_path)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("capacity_source_model_structure")
    rock_meshes = [mesh for mesh in asset.findall("mesh")
                   if (mesh.get("name") or "").startswith("r1_rock_mesh_")]
    rock_bodies = [body for body in worldbody.findall("body")
                   if ROCK_NAME_RE.fullmatch(body.get("name") or "")]
    if len(rock_meshes) != 2 or len(rock_bodies) != 2:
        raise ValueError("capacity_source_two_rock_contract")
    template_mesh = copy.deepcopy(next(
        mesh for mesh in rock_meshes if mesh.get("name") == "r1_rock_mesh_000"))
    template_body = copy.deepcopy(next(
        body for body in rock_bodies if body.get("name") == "r1-rock-000"))
    for mesh in rock_meshes:
        asset.remove(mesh)
    for body in rock_bodies:
        worldbody.remove(body)
    diameter = float(config["synthetic_layout"]["equivalent_diameter_m"])
    source_scale_values = [float(value) for value in
                           (template_mesh.get("scale") or "").split()]
    if (len(source_scale_values) != 3
            or not all(math.isfinite(value) and value > 0.0
                       for value in source_scale_values)
            or max(source_scale_values) - min(source_scale_values) > 1.0e-12):
        raise ValueError("capacity_source_rock_mesh_scale")
    source_diameter = source_scale_values[0]
    template_mesh.set("name", "capacity_rock_mesh")
    template_mesh.set("scale", _numbers([diameter] * 3))
    asset.append(template_mesh)
    mass_scale = (diameter / source_diameter) ** 3
    inertia_scale = (diameter / source_diameter) ** 5
    ids = []
    positions = []
    for index in range(rock_count):
        rock_id = f"r1-rock-{index:03d}"
        body = copy.deepcopy(template_body)
        body.set("name", rock_id)
        position = _synthetic_position(index, config["synthetic_layout"])
        body.set("pos", _numbers(position))
        body.set("quat", "1 0 0 0")
        joint = body.find("freejoint")
        inertial = body.find("inertial")
        geom = body.find("geom")
        if joint is None or inertial is None or geom is None:
            raise ValueError("capacity_source_rock_body_structure")
        joint.set("name", f"r1_rock_free_{index:03d}")
        geom.set("name", f"r1_rock_geom_{index:03d}")
        geom.set("mesh", "capacity_rock_mesh")
        mass = float(inertial.get("mass")) * mass_scale
        inertia = [float(value) * inertia_scale
                   for value in inertial.get("fullinertia").split()]
        if not math.isfinite(mass) or not all(math.isfinite(v) for v in inertia):
            raise ValueError("capacity_nonfinite_mass_or_inertia")
        inertial.set("mass", f"{mass:.17g}")
        inertial.set("fullinertia", _numbers(inertia))
        worldbody.append(body)
        ids.append(rock_id)
        positions.append(position)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    return {
        "schema": "excavation_mixed_material_capacity_model_manifest_v1",
        "status": "capacity_model_generated",
        "config_sha256": sha256(config_path),
        "source_model_sha256": sha256(source_path),
        "model_sha256": sha256(output),
        "rock_count": rock_count,
        "expected_nq": 19 + 7 * rock_count,
        "expected_nv": 18 + 6 * rock_count,
        "expected_nu": 12,
        "stable_rock_ids": ids,
        "synthetic_positions_world_m": positions,
        "geometry_only_capacity_fixture": True,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    }


def render_capacity_bridge(config_path: str | Path,
                           repository_root: str | Path) -> tuple[str, dict]:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_and_validate_config(config_path, repository)
    source_path = repository / config["frozen_inputs"]["two_rock_bridge"]["path"]
    source = source_path.read_text(encoding="utf-8")
    protocol = config["protocol"]
    maximum = int(config["capacity"]["maximum_formal_rock_count"])
    source = _replace_once(
        source, "mixed_coupled_bridge_r3_c model.xml",
        "mixed_coupled_bridge_capacity model.xml", "usage")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_c_state_v1",
        protocol["state_schema"], "state_schema")
    source = _replace_once(
        source, "excavation_mixed_coupled_bridge_r3_c_v1",
        protocol["ready_schema"], "ready_schema")
    source = _replace_once(
        source, "rock_count != 2 ||",
        f"rock_count < 1 || rock_count > {maximum} ||", "rock_count")
    source = _replace_once(
        source,
        "model contract: MuJoCo 238, mechanism 19/18/12 plus one or two rocks",
        "capacity model contract: MuJoCo 238, mechanism 19/18/12 plus dynamic rocks",
        "model_error")
    source = _replace_once(
        source,
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_force{};\n"
        "    std::array<std::array<double, 3>, 2> applied_rock_mpm_torque{};\n",
        "    std::vector<std::array<double, 3>> applied_rock_mpm_force(rock_count);\n"
        "    std::vector<std::array<double, 3>> applied_rock_mpm_torque(rock_count);\n",
        "dynamic_applied_rock_wrenches")
    source = _replace_once(
        source, "#include <vector>\n",
        "#include <unordered_map>\n#include <vector>\n", "unordered_map_include")
    source = _replace_once(
        source,
        "    std::vector<int> rock_bodies;\n"
        "    std::vector<int> rock_joints;\n",
        "    std::vector<int> rock_bodies;\n"
        "    std::vector<int> rock_joints;\n"
        "    std::unordered_map<std::string, int> rock_index_by_name;\n",
        "rock_name_index")
    source = _replace_once(
        source,
        "      rock_bodies.push_back(body);\n"
        "      rock_joints.push_back(joint);\n",
        "      rock_bodies.push_back(body);\n"
        "      rock_joints.push_back(joint);\n"
        "      rock_index_by_name.emplace(body_name, index);\n",
        "populate_rock_name_index")
    ready_old = '''    std::cout << "@READY {\\"schema\\":\\"excavation_mixed_coupled_bridge_capacity_v1\\""
              << ",\\"mechanism_nq\\":19,\\"mechanism_nv\\":18"
              << ",\\"rock_count\\":" << rock_count
              << ",\\"maximum_external_wrench_count\\":3"
              << ",\\"allowed_external_wrench_bodies\\":[\\"bucket\\",\\"r1-rock-000\\",\\"r1-rock-001\\"]"
              << ",\\"external_wrench_input\\":\\"named_world_wrench_about_interval_start_body_origin\\"}"
              << std::endl;
'''
    ready_new = '''    std::cout << "@READY {\\"schema\\":\\"excavation_mixed_coupled_bridge_capacity_v1\\""
              << ",\\"mechanism_nq\\":19,\\"mechanism_nv\\":18"
              << ",\\"rock_count\\":" << rock_count
              << ",\\"maximum_external_wrench_count\\":" << rock_count + 1
              << ",\\"allowed_external_wrench_bodies\\":[\\"bucket\\"";
    for (int index = 0; index < rock_count; ++index) {
      std::cout << ",\\"r1-rock-" << std::setw(3) << std::setfill('0')
                << index << std::setfill(' ') << "\\"";
    }
    std::cout << "]"
              << ",\\"external_wrench_input\\":\\"named_world_wrench_about_interval_start_body_origin\\"}"
              << std::endl;
'''
    source = _replace_once(source, ready_old, ready_new, "dynamic_ready")
    source = _replace_once(
        source,
        "external_wrench_count_value > 3 ||",
        "external_wrench_count_value > rock_count + 1 ||",
        "dynamic_wrench_count")
    source = _replace_once(
        source, "std::array<bool, 2> seen_rocks{};",
        "std::vector<bool> seen_rocks(rock_count, false);",
        "dynamic_seen_rocks")
    parser_old = '''        } else if (body_name == "r1-rock-000") {
          rock_index = 0;
        } else if (body_name == "r1-rock-001") {
          rock_index = 1;
        } else {
          throw std::runtime_error("unapproved R3-C wrench body");
        }
'''
    parser_new = '''        } else if (body_name.rfind("r1-rock-", 0) == 0) {
          const auto found = rock_index_by_name.find(body_name);
          if (found == rock_index_by_name.end()) {
            throw std::runtime_error("unknown or absent capacity rock wrench");
          }
          rock_index = found->second;
        } else {
          throw std::runtime_error("unapproved capacity wrench body");
        }
'''
    source = _replace_once(source, parser_old, parser_new, "dynamic_named_wrench")
    source = source.replace("R3-C external wrench count contract",
                            "capacity external wrench count contract")
    source = source.replace("missing R3-C wrench body",
                            "missing capacity wrench body")
    source = source.replace("duplicate or absent R3-C rock wrench",
                            "duplicate or absent capacity rock wrench")
    source = source.replace("mixed_coupled_bridge_r3_c:",
                            "mixed_coupled_bridge_capacity:")
    forbidden_fragments = (
        "rock_count != 2", "std::array<bool, 2>",
        "std::array<std::array<double, 3>, 2>",
        'body_name == "r1-rock-000"', 'body_name == "r1-rock-001"',
        '"maximum_external_wrench_count\\":3',
    )
    if any(fragment in source for fragment in forbidden_fragments):
        raise ValueError("capacity_bridge_transform_incomplete")
    return source, {
        "schema": "excavation_mixed_material_capacity_bridge_manifest_v1",
        "status": "capacity_bridge_source_generated",
        "config_sha256": sha256(config_path),
        "source_bridge_sha256": sha256(source_path),
        "generated_source_sha256": hashlib.sha256(
            source.encode("utf-8")).hexdigest(),
        "minimum_rock_count": 1,
        "maximum_rock_count": maximum,
        "runtime_rock_count": True,
        "maximum_external_wrench_count": "rock_count_plus_bucket",
        "stable_named_wrench_routing": True,
        "physics_qualified": False,
        "formal_scene_qualified": False,
    }
