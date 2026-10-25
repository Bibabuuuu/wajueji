"""Assemble an isolated R2 multi-rock MJCF from a reviewed assembly plan.

This module only generates a candidate model.  It does not run forward dynamics,
step the model, or change the validated R1 bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

from mixed_material_r1 import build_proxy


MANIFEST_SCHEMA = "excavation_mixed_material_r2_model_candidate_v1"
PLAN_SCHEMA = "excavation_scene_assembly_plan_v1"
STABLE_INITIAL_STATE_SCHEMA = (
    "excavation_mixed_material_r2_stable_initial_state_v1")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [element for element in root.iter(tag) if element.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected_one:{tag}:{name}:{len(matches)}")
    return matches[0]


def _finite_numbers(value) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return False


def _validate_inertia(matrix) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("invalid_body_inertia_shape")
    result = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("invalid_body_inertia_shape")
        result.append([float(value) for value in row])
    if not _finite_numbers(result):
        raise ValueError("non_finite_body_inertia")
    tolerance = 1e-10
    if any(abs(result[row][column] - result[column][row]) > tolerance
           for row in range(3) for column in range(3)):
        raise ValueError("non_symmetric_body_inertia")
    a, b, c = result[0]
    _, d, e = result[1]
    _, _, f = result[2]
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    if a <= 0 or a * d - b * b <= 0 or determinant <= 0:
        raise ValueError("non_positive_definite_body_inertia")
    return result


def _validate_config(config: dict) -> None:
    if config.get("schema") != "excavation_mixed_material_r2_config_v1":
        raise ValueError("invalid_r2_config_schema")
    capacity = config["capacity"]
    if capacity != {
        "policy": "single_bridge_maximum_capacity_actual_count_fixed_at_model_load",
        "maximum_rock_count": 20,
        "placeholder_rocks_allowed": False,
        "runtime_add_remove_allowed": False,
        "qpos_per_rock": 7,
        "qvel_per_rock": 6,
    }:
        raise ValueError("invalid_single_bridge_capacity_policy")
    first = config["first_admission"]
    if first["required_rock_count"] != 20 or len(first["scene_ids"]) != 6:
        raise ValueError("invalid_first_admission_scope")
    if len(set(first["scene_ids"])) != 6:
        raise ValueError("duplicate_first_admission_scene")
    scope = config["scope"]
    if (scope["forward_dynamics"] or scope["time_stepping"] or
            scope["bridge_modification"] or scope["physics_qualification"] or
            scope["training"]):
        raise ValueError("r2_load_only_scope_expanded")
    bits = [int(value) for value in config["contact_bits"].values()]
    if len(bits) != len(set(bits)) or any(value <= 0 or value & (value - 1)
                                            for value in bits):
        raise ValueError("invalid_contact_bits")
    bridge = config["bridge_candidate"]
    if (bridge["maximum_rock_count"] != 20 or
            bridge["ready_schema"] != "excavation_mixed_rigid_bridge_v2" or
            bridge["state_schema"] != "excavation_mixed_rigid_bridge_state_v2" or
            bridge["actual_rock_count_source"] !=
            "loaded_model_dimensions_and_contiguous_named_freejoints" or
            bridge["external_wrench_input"] != "reserved_zero_only"):
        raise ValueError("invalid_r2_bridge_candidate_contract")
    surface = config["surface_support"]
    if (surface["policy"] !=
            "deterministic_static_heightfield_from_reviewed_scene_manifest" or
            int(surface["grid_nrow"]) != 65 or
            int(surface["grid_ncol"]) != 65 or
            float(surface["base_depth_m"]) <= 0 or
            int(surface["interpolation_subsamples_per_cell"]) != 4 or
            float(surface["maximum_interpolation_error_m"]) <= 0 or
            surface["heightfield_asset_name"] != "r2_surface_heightfield" or
            surface["heightfield_geom_name"] != "r2_surface_terrain" or
            surface["heightfield_file_name"] != "surface-heightfield.hfield" or
            surface["source_ground_plane_rock_collision_enabled"] is not False):
        raise ValueError("invalid_r2_surface_support_contract")
    settling = config["settling_probe"]
    if (float(settling["duration_s"]) != 1.0 or
            float(settling["output_dt_s"]) != 1.0 / 120.0 or
            int(settling["internal_substeps_per_output"]) != 20 or
            settling["command"] != [0.0, 0.0, 0.0, 0.0] or
            int(settling["external_wrench_count"]) != 0 or
            float(settling["maximum_closure_m"]) != 0.001 or
            int(settling["required_forbidden_contact_count"]) != 0 or
            config["scope"].get("settling_probe_enabled") is not True):
        raise ValueError("invalid_r2_settling_probe_contract")
    if not _finite_numbers(config):
        raise ValueError("non_finite_r2_config")


def _surface_height(surface: dict, x: float, y: float) -> float:
    origin_x, origin_y = map(float, surface["origin_xy_m"])
    value = (float(surface["base_height_m"])
             + float(surface["slope_dz_dx"]) * (x - origin_x)
             + float(surface["slope_dz_dy"]) * (y - origin_y))
    for bump in surface["bumps"]:
        center_x, center_y = map(float, bump["center_xy_m"])
        sigma_x, sigma_y = map(float, bump["sigma_xy_m"])
        if sigma_x <= 0 or sigma_y <= 0:
            raise ValueError("invalid_surface_bump_sigma")
        exponent = -0.5 * (
            ((x - center_x) / sigma_x) ** 2
            + ((y - center_y) / sigma_y) ** 2)
        value += float(bump["amplitude_m"]) * math.exp(exponent)
    if not math.isfinite(value):
        raise ValueError("non_finite_surface_height")
    return value


def _surface_source(plan: dict, config: dict, repository: Path,
                    scene_manifest_override_path: str | Path | None = None
                    ) -> tuple[Path, dict]:
    support = config["surface_support"]
    if scene_manifest_override_path is None:
        reconstruction_root = repository / support["scene_reconstruction_root"]
        root_report = reconstruction_root / "report.json"
        if (not root_report.is_file() or
                sha256(root_report) != support[
                    "scene_reconstruction_root_report_sha256"]):
            raise ValueError("scene_reconstruction_root_report_sha256_mismatch")
        scene_manifest_path = (
            reconstruction_root / plan["scene_id"] / "scene-manifest.json")
    else:
        scene_manifest_path = Path(scene_manifest_override_path).resolve()
    expected_hash = plan["source_input_hashes"]["scene_manifest_sha256"]
    if (not scene_manifest_path.is_file() or
            sha256(scene_manifest_path) != expected_hash):
        raise ValueError("scene_manifest_sha256_mismatch")
    manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("scene_id") != plan["scene_id"] or
            manifest.get("scene_class") != plan["scene_class"] or
            manifest.get("status") != "geometry_reconstruction_candidate"):
        raise ValueError("scene_manifest_identity_or_status_mismatch")
    if not _finite_numbers(manifest):
        raise ValueError("non_finite_scene_manifest")
    return scene_manifest_path, manifest


def _rectangle_bounds(polygon) -> tuple[float, float, float, float]:
    if not isinstance(polygon, list) or len(polygon) != 5 or polygon[0] != polygon[-1]:
        raise ValueError("surface_region_requires_closed_rectangle")
    points = [(float(point[0]), float(point[1])) for point in polygon[:-1]]
    xs = sorted(set(point[0] for point in points))
    ys = sorted(set(point[1] for point in points))
    if (len(xs) != 2 or len(ys) != 2 or len(set(points)) != 4 or
            set(points) != {(x, y) for x in xs for y in ys} or
            xs[0] >= xs[1] or ys[0] >= ys[1]):
        raise ValueError("surface_region_requires_axis_aligned_rectangle")
    return xs[0], xs[1], ys[0], ys[1]


def _triangle_height(values, row: int, column: int,
                     u: float, v: float) -> float:
    # MuJoCo uses the diagonal from a cell's lower-left to upper-right vertex.
    if u >= v:
        return (values[row][column] * (1.0 - u)
                + values[row][column + 1] * (u - v)
                + values[row + 1][column + 1] * v)
    return (values[row][column] * (1.0 - v)
            + values[row + 1][column] * (v - u)
            + values[row + 1][column + 1] * u)


def _build_heightfield(plan: dict, config: dict, repository: Path,
                       output_model: Path,
                       scene_manifest_override_path: str | Path | None = None) -> dict:
    if sys.byteorder != "little":
        raise RuntimeError("mujoco_heightfield_writer_requires_little_endian_host")
    scene_manifest_path, scene_manifest = _surface_source(
        plan, config, repository, scene_manifest_override_path)
    support = config["surface_support"]
    surface = scene_manifest["surface_model"]
    x_min, x_max, y_min, y_max = _rectangle_bounds(
        scene_manifest["region_polygon_xy_m"])
    nrow = int(support["grid_nrow"])
    ncol = int(support["grid_ncol"])
    values = []
    for row in range(nrow):
        y = y_min + (y_max - y_min) * row / (nrow - 1)
        row_values = []
        for column in range(ncol):
            x = x_min + (x_max - x_min) * column / (ncol - 1)
            raw = _surface_height(surface, x, y)
            row_values.append(struct.unpack("<f", struct.pack("<f", raw))[0])
        values.append(row_values)
    minimum = min(min(row) for row in values)
    maximum = max(max(row) for row in values)
    elevation = maximum - minimum
    if not elevation > 0:
        raise ValueError("heightfield_requires_positive_elevation_range")

    subsamples = int(support["interpolation_subsamples_per_cell"])
    maximum_error = 0.0
    squared_error = 0.0
    sample_count = 0
    for row in range(nrow - 1):
        for column in range(ncol - 1):
            for subrow in range(subsamples + 1):
                v = subrow / subsamples
                y = y_min + (y_max - y_min) * (row + v) / (nrow - 1)
                for subcolumn in range(subsamples + 1):
                    u = subcolumn / subsamples
                    x = x_min + (x_max - x_min) * (column + u) / (ncol - 1)
                    error = abs(
                        _triangle_height(values, row, column, u, v)
                        - _surface_height(surface, x, y))
                    maximum_error = max(maximum_error, error)
                    squared_error += error * error
                    sample_count += 1
    allowed_error = float(support["maximum_interpolation_error_m"])
    if maximum_error > allowed_error:
        raise ValueError("heightfield_interpolation_error_above_limit")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    asset_path = output_model.parent / support["heightfield_file_name"]
    payload = bytearray(struct.pack("<ii", nrow, ncol))
    payload.extend(struct.pack(
        "<" + "f" * (nrow * ncol),
        *(value for row in values for value in row)))
    asset_path.write_bytes(payload)
    return {
        "scene_manifest_path": str(scene_manifest_path.relative_to(repository)),
        "scene_manifest_sha256": sha256(scene_manifest_path),
        "asset_path": asset_path,
        "asset_file_name": asset_path.name,
        "asset_sha256": sha256(asset_path),
        "nrow": nrow,
        "ncol": ncol,
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
        "center_xyz_m": [
            0.5 * (x_min + x_max), 0.5 * (y_min + y_max), minimum],
        "radius_xy_m": [0.5 * (x_max - x_min), 0.5 * (y_max - y_min)],
        "minimum_sampled_height_m": minimum,
        "maximum_sampled_height_m": maximum,
        "elevation_range_m": elevation,
        "base_depth_m": float(support["base_depth_m"]),
        "maximum_interpolation_error_m": maximum_error,
        "rms_interpolation_error_m": math.sqrt(squared_error / sample_count),
        "interpolation_error_sample_count": sample_count,
        "source_surface_model": surface,
        "source_region_polygon_xy_m": scene_manifest["region_polygon_xy_m"],
    }


def _validate_plan(plan: dict, config: dict, repository: Path,
                   allow_unlisted_candidate_scene: bool = False) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid_assembly_plan_schema")
    if plan.get("status") != "scene_assembly_preflight_candidate":
        raise ValueError("invalid_assembly_plan_status")
    first = config["first_admission"]
    if (plan.get("scene_id") not in first["scene_ids"] and
            not allow_unlisted_candidate_scene):
        raise ValueError("scene_not_in_first_admission")
    if plan.get("scene_class") not in first["admitted_scene_classes"]:
        raise ValueError("scene_class_not_in_first_admission")
    rocks = plan.get("rocks")
    if not isinstance(rocks, list) or len(rocks) != plan.get("rock_count"):
        raise ValueError("assembly_plan_rock_count_mismatch")
    if len(rocks) != first["required_rock_count"]:
        raise ValueError("first_admission_requires_twenty_rocks")
    if len(rocks) > config["capacity"]["maximum_rock_count"]:
        raise ValueError("maximum_rock_capacity_exceeded")
    if not _finite_numbers(plan):
        raise ValueError("non_finite_assembly_plan")
    seen_rock_ids = set()
    for slot, rock in enumerate(rocks):
        binding = rock["full_scene_binding"]
        expected = {
            "slot": slot,
            "body_name": f"r2-rock-{slot:03d}",
            "mesh_name": f"r2_rock_mesh_{slot:03d}",
            "geom_name": f"r2_rock_geom_{slot:03d}",
            "freejoint_name": f"r2_rock_free_{slot:03d}",
            "qpos_address": 19 + 7 * slot,
            "qpos_width": 7,
            "qvel_address": 18 + 6 * slot,
            "qvel_width": 6,
            "contact_role": "rock",
            "contact_bits": config["contact_bits"],
        }
        if binding != expected:
            raise ValueError(f"invalid_full_scene_binding:{slot}")
        if rock["rock_id"] in seen_rock_ids:
            raise ValueError("duplicate_rock_id")
        seen_rock_ids.add(rock["rock_id"])
        asset_path = repository / rock["asset_path"]
        if not asset_path.is_file() or sha256(asset_path) != rock["asset_sha256"]:
            raise ValueError(f"rock_asset_hash_mismatch:{slot}")
        if float(rock["mass_kg"]) <= 0 or float(rock["equivalent_diameter_m"]) <= 0:
            raise ValueError(f"invalid_rock_mass_or_scale:{slot}")
        quaternion = [float(value) for value in rock["orientation_quaternion_wxyz"]]
        if len(quaternion) != 4 or not math.isclose(
                sum(value * value for value in quaternion), 1.0,
                rel_tol=0.0, abs_tol=2e-9):
            raise ValueError(f"invalid_rock_quaternion:{slot}")
        if len(rock["position_world_m"]) != 3:
            raise ValueError(f"invalid_rock_position:{slot}")
        _validate_inertia(rock["inertia_about_com_body_kg_m2"])


def _load_pose_override(path: str | Path | None, plan: dict) -> tuple[dict, str | None]:
    if path is None:
        return {}, None
    override_path = Path(path).resolve()
    override = json.loads(override_path.read_text(encoding="utf-8"))
    if (override.get("schema") != STABLE_INITIAL_STATE_SCHEMA or
            override.get("status") != "stable_initial_state_candidate" or
            override.get("scene_id") != plan["scene_id"] or
            override.get("rock_count") != len(plan["rocks"]) or
            not _finite_numbers(override)):
        raise ValueError("invalid_stable_initial_state_contract")
    records = override.get("rocks")
    if not isinstance(records, list) or len(records) != len(plan["rocks"]):
        raise ValueError("stable_initial_state_rock_count_mismatch")
    indexed = {}
    for slot, (record, source) in enumerate(zip(records, plan["rocks"])):
        if (record.get("slot") != slot or
                record.get("source_rock_id") != source["rock_id"] or
                record.get("body_name") != f"r2-rock-{slot:03d}" or
                record.get("initial_qvel") != [0.0] * 6):
            raise ValueError(f"stable_initial_state_identity_mismatch:{slot}")
        position = [float(value) for value in record["position_world_m"]]
        quaternion = [float(value) for value in record["orientation_quaternion_wxyz"]]
        if (len(position) != 3 or len(quaternion) != 4 or
                not math.isclose(sum(value * value for value in quaternion), 1.0,
                                 rel_tol=0.0, abs_tol=2e-9)):
            raise ValueError(f"invalid_stable_initial_pose:{slot}")
        indexed[slot] = {
            "position_world_m": position,
            "orientation_quaternion_wxyz": quaternion,
        }
    return indexed, sha256(override_path)


def build_candidate(config_path: str | Path, assembly_plan_path: str | Path,
                    output_model: str | Path,
                    repository_root: str | Path,
                    initial_pose_override_path: str | Path | None = None,
                    surface_scene_manifest_path: str | Path | None = None) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    plan_path = Path(assembly_plan_path).resolve()
    output = Path(output_model).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _validate_plan(
        plan, config, repository,
        allow_unlisted_candidate_scene=surface_scene_manifest_path is not None)
    pose_overrides, pose_override_sha256 = _load_pose_override(
        initial_pose_override_path, plan)

    source = repository / config["source_model"]["path"]
    r1_config_path = repository / config["bucket_proxy"]["r1_config_path"]
    inventory_path = Path(config["bucket_proxy"]["inventory_path"])
    expected_hashes = (
        (source, config["source_model"]["sha256"], "source_model"),
        (r1_config_path, config["bucket_proxy"]["r1_config_sha256"], "r1_config"),
        (inventory_path, config["bucket_proxy"]["inventory_sha256"], "inventory"),
    )
    for path, expected, label in expected_hashes:
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(label + "_sha256_mismatch")
    r1_config = json.loads(r1_config_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    proxy = build_proxy(inventory, r1_config)
    if proxy["geom_count"] != config["bucket_proxy"]["expected_geom_count"]:
        raise ValueError("bucket_proxy_geom_count_mismatch")

    heightfield = _build_heightfield(
        plan, config, repository, output, surface_scene_manifest_path)

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if compiler is None or asset is None or worldbody is None:
        raise ValueError("missing_model_sections")
    source_meshdir = Path(compiler.get("meshdir", ""))
    if not source_meshdir.is_absolute():
        source_meshdir = source.parent / source_meshdir
    for source_mesh in asset.findall("mesh"):
        source_file = source_mesh.get("file")
        if source_file and not Path(source_file).is_absolute():
            resolved = (source_meshdir / source_file).resolve()
            if not resolved.is_file():
                raise ValueError("source_mesh_file_missing:" + source_file)
            source_mesh.set("file", str(resolved))
    # MuJoCo resolves hfields through meshdir as well.  Existing source meshes
    # are absolute above, so using the model directory makes the adjacent
    # generated hfield portable without embedding an output-specific path.
    compiler.set("meshdir", ".")
    original_bucket = _find_named(root, "geom", "bucket")
    if original_bucket.get("type") != "mesh":
        raise ValueError("unexpected_source_bucket_geom")
    original_bucket.set("contype", "0")
    original_bucket.set("conaffinity", "0")
    original_bucket.set("group", "1")

    bits = config["contact_bits"]
    rock_bit = int(bits["rock"])
    bucket_bit = int(bits["bucket_proxy"])
    ground_bit = int(bits["rock_ground"])
    truck_bit = int(bits["truck_receiver_reserved"])
    bucket_body = _find_named(root, "body", "bucket")
    for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
        ET.SubElement(bucket_body, "geom", {
            "name": "r2_" + box["name"],
            "type": "box",
            "pos": _numbers(box["position_bucket_m"]),
            "quat": _numbers(box["quaternion_wxyz"]),
            "size": _numbers(box["half_size_m"]),
            "density": "0",
            "rgba": "0.2 0.6 0.9 0",
            "group": "3",
            "contype": str(bucket_bit),
            "conaffinity": str(rock_bit),
        })
    for side in proxy["side_plate_meshes"]:
        side_path = repository / "assets" / "v4" / "mixed-material-r1" / f"{side['name']}.obj"
        if not side_path.is_file():
            raise ValueError("missing_side_plate_asset:" + side["name"])
        mesh_name = "r2_" + side["name"]
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(side_path)})
        ET.SubElement(bucket_body, "geom", {
            "name": mesh_name,
            "type": "mesh",
            "mesh": mesh_name,
            "density": "0",
            "rgba": "0.2 0.6 0.9 0",
            "group": "3",
            "contype": str(bucket_bit),
            "conaffinity": str(rock_bit),
        })
    ground = _find_named(root, "geom", "ground_plane")
    ground.set("contype", "1")
    ground.set("conaffinity", "1")
    support = config["surface_support"]
    ET.SubElement(asset, "hfield", {
        "name": support["heightfield_asset_name"],
        "file": heightfield["asset_file_name"],
        "content_type": "image/vnd.mujoco.hfield",
        "size": _numbers([
            *heightfield["radius_xy_m"],
            heightfield["elevation_range_m"],
            heightfield["base_depth_m"],
        ]),
    })
    ET.SubElement(worldbody, "geom", {
        "name": support["heightfield_geom_name"],
        "type": "hfield",
        "hfield": support["heightfield_asset_name"],
        "pos": _numbers(heightfield["center_xyz_m"]),
        "density": "0",
        "rgba": "0.38 0.27 0.16 1",
        "contype": str(ground_bit),
        "conaffinity": str(rock_bit),
    })

    rock_records = []
    for slot, rock in enumerate(plan["rocks"]):
        binding = rock["full_scene_binding"]
        pose = pose_overrides.get(slot, {
            "position_world_m": rock["position_world_m"],
            "orientation_quaternion_wxyz": rock["orientation_quaternion_wxyz"],
        })
        scale = float(rock["equivalent_diameter_m"])
        inertia = _validate_inertia(rock["inertia_about_com_body_kg_m2"])
        mesh_path = repository / rock["asset_path"]
        ET.SubElement(asset, "mesh", {
            "name": binding["mesh_name"],
            "file": str(mesh_path),
            "scale": _numbers([scale, scale, scale]),
        })
        body = ET.SubElement(worldbody, "body", {
            "name": binding["body_name"],
            "pos": _numbers(pose["position_world_m"]),
            "quat": _numbers(pose["orientation_quaternion_wxyz"]),
        })
        ET.SubElement(body, "freejoint", {"name": binding["freejoint_name"]})
        ET.SubElement(body, "inertial", {
            "pos": "0 0 0",
            "mass": f"{float(rock['mass_kg']):.17g}",
            "fullinertia": _numbers([
                inertia[0][0], inertia[1][1], inertia[2][2],
                inertia[0][1], inertia[0][2], inertia[1][2],
            ]),
        })
        ET.SubElement(body, "geom", {
            "name": binding["geom_name"],
            "type": "mesh",
            "mesh": binding["mesh_name"],
            "density": "0",
            "rgba": "0.48 0.31 0.18 1",
            "contype": str(rock_bit),
            "conaffinity": str(rock_bit | bucket_bit | ground_bit | truck_bit),
        })
        rock_records.append({
            "slot": slot,
            "source_rock_id": rock["rock_id"],
            "prototype_id": rock["prototype_id"],
            "body_name": binding["body_name"],
            "geom_name": binding["geom_name"],
            "freejoint_name": binding["freejoint_name"],
            "qpos_address": binding["qpos_address"],
            "qvel_address": binding["qvel_address"],
            "mass_kg": float(rock["mass_kg"]),
            "initial_position_world_m": pose["position_world_m"],
            "initial_orientation_quaternion_wxyz": (
                pose["orientation_quaternion_wxyz"]),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    source_contract = config["source_model"]
    rock_count = len(rock_records)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "r2_model_candidate_generated",
        "scene_id": plan["scene_id"],
        "scene_class": plan["scene_class"],
        "rock_count": rock_count,
        "maximum_rock_capacity": config["capacity"]["maximum_rock_count"],
        "placeholder_rock_count": 0,
        "source_model_sha256": sha256(source),
        "r1_config_sha256": sha256(r1_config_path),
        "inventory_sha256": sha256(inventory_path),
        "assembly_plan_sha256": sha256(plan_path),
        "initial_pose_source": (
            "stable_initial_state_candidate"
            if pose_override_sha256 is not None else "scene_assembly_plan"),
        "initial_pose_override_sha256": pose_override_sha256,
        "surface_support": {
            key: value for key, value in heightfield.items()
            if key != "asset_path"
        },
        "candidate_model_sha256": sha256(output),
        "expected_candidate_contract": {
            "nq": int(source_contract["mechanism_nq"]) + 7 * rock_count,
            "nv": int(source_contract["mechanism_nv"]) + 6 * rock_count,
            "nu": int(source_contract["actuator_count"]),
            "nplugin": int(source_contract["plugin_count"]),
            "rock_count": rock_count,
        },
        "contact_bits": bits,
        "bucket_proxy_geom_count": proxy["geom_count"],
        "total_rock_mass_kg": sum(record["mass_kg"] for record in rock_records),
        "rocks": rock_records,
        "frozen_source_modified": False,
        "physics_qualified": False,
        "scope_limit": (
            "MJCF generation only; requires a separate load-only inspection and "
            "does not qualify settling, excavation, retention, dump or fill rate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assembly-plan", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--initial-pose-override")
    parser.add_argument("--surface-scene-manifest")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    manifest = build_candidate(
        args.config, args.assembly_plan, args.output_model, args.repository_root,
        args.initial_pose_override, args.surface_scene_manifest)
    if args.manifest:
        write_json(args.manifest, manifest)
    else:
        print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
