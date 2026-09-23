#!/usr/bin/env python3
"""Validate a D-063 stable-scene manifest and assemble its actual-N MuJoCo model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from build_mixed_material_capacity_preflight import (
    load_and_validate_config as load_capacity_config,
    sha256,
)
from mixed_material_r0 import normalized_prototypes


CONFIG_SCHEMA = "excavation_formal_scene_model_adapter_config_v1"
MODEL_MANIFEST_SCHEMA = "excavation_formal_scene_model_adapter_manifest_v1"
TEMPLATE_ROCK_RE = re.compile(r"^r1-rock-\d{3,4}$")


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"formal_scene_{label}_shape")
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"formal_scene_{label}_nonfinite")
    return array


def load_and_validate_config(config_path: str | Path,
                             repository_root: str | Path) -> tuple[dict, Path]:
    repository = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "visual_stable_manifest_to_time_zero_model_only"):
        raise ValueError("formal_adapter_config_schema_or_status")
    binding = config.get("frozen_capacity_config")
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}):
        raise ValueError("formal_adapter_capacity_binding")
    capacity_path = repository / binding["path"]
    if not capacity_path.is_file() or sha256(capacity_path) != binding["sha256"]:
        raise ValueError("formal_adapter_capacity_binding_mismatch")
    capacity = load_capacity_config(capacity_path, repository)
    scene = config.get("scene_contract")
    if scene != {
        "schema": "excavation_fullscale_visual_stable_scene_manifest_v1",
        "status": "visual_stable_geometry_candidate",
        "scenario_name": "mixed-autonomous-fullscale-graded-gravel-slope-truck-v1",
        "coordinate_frame": "world",
        "minimum_rock_count": 1,
        "maximum_rock_count": 1024,
        "required_true_checks": [
            "finite_state", "stable_id_count", "rock_rock_no_intersection",
            "stable_rigid_skeleton", "small_surface_dominant",
            "medium_large_lower_bias"],
    }:
        raise ValueError("formal_adapter_scene_contract_drift")
    if scene["maximum_rock_count"] != capacity["capacity"][
            "maximum_formal_rock_count"]:
        raise ValueError("formal_adapter_capacity_limit_mismatch")
    rocks = config.get("rocks")
    if (rocks.get("density_kg_m3") != 2600.0
            or rocks.get("asset_root") != "assets/v4/mixed-material-r0"
            or rocks.get("size_classes") != {
                "small_0p16_0p24": [0.16, 0.24],
                "medium_0p24_0p32": [0.24, 0.32],
                "large_0p32_0p40": [0.32, 0.4],
            }):
        raise ValueError("formal_adapter_rock_contract_drift")
    for prototype, expected_hash in rocks.get("assets", {}).items():
        asset = repository / rocks["asset_root"] / f"{prototype}.obj"
        if not asset.is_file() or sha256(asset) != expected_hash:
            raise ValueError("formal_adapter_asset_mismatch:" + prototype)
    if set(rocks.get("assets", {})) != {
            "angular_tetra", "angular_bipyramid", "angular_octa", "angular_block"}:
        raise ValueError("formal_adapter_asset_set")
    if config.get("runtime_mapping") != {
        "body_id_format": "r1-rock-%03d",
        "joint_id_format": "r1_rock_free_%03d",
        "geom_id_format": "r1_rock_geom_%03d",
        "mesh_id_format": "formal_rock_mesh_%03d",
        "preserve_source_rock_id_in_model_manifest": True,
    }:
        raise ValueError("formal_adapter_runtime_mapping_drift")
    if config.get("scope") != {
        "manifest_validation": True,
        "model_generation": True,
        "time_zero_ready_for_separate_runner": True,
        "physics_step": False,
        "mpm_or_gpu_run": False,
        "settling": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
        "training": False,
    }:
        raise ValueError("formal_adapter_scope_expanded")
    return config, capacity_path


def validate_scene_manifest(config: dict, manifest: dict,
                            repository_root: str | Path) -> list[dict]:
    repository = Path(repository_root).resolve()
    contract = config["scene_contract"]
    for key in ("schema", "status", "scenario_name", "coordinate_frame"):
        if manifest.get(key) != contract[key]:
            raise ValueError("formal_scene_identity_mismatch:" + key)
    checks = manifest.get("checks")
    if not isinstance(checks, dict) or any(
            checks.get(name) is not True for name in contract["required_true_checks"]):
        raise ValueError("formal_scene_required_checks_missing")
    records = manifest.get("rocks")
    count = manifest.get("rock_count")
    if (isinstance(count, bool) or not isinstance(count, int)
            or not isinstance(records, list) or count != len(records)
            or count < contract["minimum_rock_count"]
            or count > contract["maximum_rock_count"]):
        raise ValueError("formal_scene_rock_count_contract")
    prototypes = normalized_prototypes(sorted(config["rocks"]["assets"]))
    seen = set()
    canonical = []
    for index, record in enumerate(records):
        source_id = record.get("rock_id")
        if (not isinstance(source_id, str) or not source_id
                or source_id in seen):
            raise ValueError("formal_scene_duplicate_or_invalid_source_id")
        seen.add(source_id)
        prototype_id = record.get("prototype_id")
        if prototype_id not in prototypes:
            raise ValueError("formal_scene_unknown_prototype")
        expected_path = (f"{config['rocks']['asset_root']}/{prototype_id}.obj")
        if (record.get("asset_path") != expected_path
                or record.get("asset_sha256") !=
                config["rocks"]["assets"][prototype_id]
                or sha256(repository / expected_path) != record["asset_sha256"]):
            raise ValueError("formal_scene_asset_binding")
        size_class = record.get("size_class")
        if size_class not in config["rocks"]["size_classes"]:
            raise ValueError("formal_scene_unknown_size_class")
        diameter = float(record.get("equivalent_diameter_m", math.nan))
        low, high = config["rocks"]["size_classes"][size_class]
        if not math.isfinite(diameter) or diameter < low or diameter > high:
            raise ValueError("formal_scene_diameter_range")
        density = float(record.get("density_kg_m3", math.nan))
        if density != float(config["rocks"]["density_kg_m3"]):
            raise ValueError("formal_scene_density")
        prototype = prototypes[prototype_id]
        expected_volume = float(prototype["unit_volume_m3"]) * diameter ** 3
        expected_mass = density * expected_volume
        expected_inertia = density * prototype["unit_inertia_m5"] * diameter ** 5
        volume = float(record.get("volume_m3", math.nan))
        mass = float(record.get("mass_kg", math.nan))
        inertia = np.asarray(record.get("inertia_about_com_body_kg_m2"), dtype=float)
        if (not math.isclose(volume, expected_volume, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(mass, expected_mass, rel_tol=1e-9, abs_tol=1e-10)
                or inertia.shape != (3, 3) or not np.isfinite(inertia).all()
                or not np.allclose(inertia, expected_inertia, rtol=1e-9, atol=1e-12)
                or not np.allclose(inertia, inertia.T, rtol=0.0, atol=1e-12)
                or np.linalg.eigvalsh(inertia).min() <= 0.0):
            raise ValueError("formal_scene_mass_inertia_recompute")
        position = _finite_vector(record.get("position_world_m"), 3, "position")
        quaternion = _finite_vector(
            record.get("orientation_quaternion_wxyz"), 4, "quaternion")
        if not math.isclose(float(np.linalg.norm(quaternion)), 1.0,
                            rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("formal_scene_quaternion_not_unit")
        linear = _finite_vector(
            record.get("linear_velocity_world_mps"), 3, "linear_velocity")
        angular = _finite_vector(
            record.get("angular_velocity_world_radps"), 3, "angular_velocity")
        if not np.allclose(linear, 0.0, atol=1e-12) or not np.allclose(
                angular, 0.0, atol=1e-12):
            raise ValueError("formal_scene_frozen_velocity_nonzero")
        canonical.append({
            "source_rock_id": source_id,
            "runtime_rock_id": f"r1-rock-{index:03d}",
            "prototype_id": prototype_id,
            "size_class": size_class,
            "asset_path": expected_path,
            "asset_sha256": record["asset_sha256"],
            "equivalent_diameter_m": diameter,
            "density_kg_m3": density,
            "volume_m3": expected_volume,
            "mass_kg": expected_mass,
            "inertia_about_com_body_kg_m2": expected_inertia.tolist(),
            "position_world_m": position.tolist(),
            "orientation_quaternion_wxyz": quaternion.tolist(),
        })
    return canonical


def build_model(config_path: str | Path, scene_manifest_path: str | Path,
                repository_root: str | Path, output_model: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    scene_path = Path(scene_manifest_path).resolve()
    config, capacity_path = load_and_validate_config(config_path, repository)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    rocks = validate_scene_manifest(config, scene, repository)
    capacity = load_capacity_config(capacity_path, repository)
    source = repository / capacity["frozen_inputs"]["two_rock_model"]["path"]
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("formal_adapter_source_model_structure")
    for mesh in list(asset.findall("mesh")):
        if (mesh.get("name") or "").startswith("r1_rock_mesh_"):
            asset.remove(mesh)
    template_bodies = [body for body in worldbody.findall("body")
                       if TEMPLATE_ROCK_RE.fullmatch(body.get("name") or "")]
    if len(template_bodies) != 2:
        raise ValueError("formal_adapter_source_two_rock_contract")
    template = copy.deepcopy(next(
        body for body in template_bodies if body.get("name") == "r1-rock-000"))
    for body in template_bodies:
        worldbody.remove(body)
    mapping = config["runtime_mapping"]
    for index, rock in enumerate(rocks):
        mesh_name = mapping["mesh_id_format"] % index
        ET.SubElement(asset, "mesh", {
            "name": mesh_name,
            "file": str(repository / rock["asset_path"]),
            "scale": _numbers([rock["equivalent_diameter_m"]] * 3),
        })
        body = copy.deepcopy(template)
        body.set("name", rock["runtime_rock_id"])
        body.set("pos", _numbers(rock["position_world_m"]))
        body.set("quat", _numbers(rock["orientation_quaternion_wxyz"]))
        joint, inertial, geom = body.find("freejoint"), body.find("inertial"), body.find("geom")
        if joint is None or inertial is None or geom is None:
            raise ValueError("formal_adapter_source_rock_body_structure")
        joint.set("name", mapping["joint_id_format"] % index)
        inertia = np.asarray(rock["inertia_about_com_body_kg_m2"], dtype=float)
        inertial.set("mass", f"{rock['mass_kg']:.17g}")
        inertial.set("fullinertia", _numbers([
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2]]))
        geom.set("name", mapping["geom_id_format"] % index)
        geom.set("mesh", mesh_name)
        worldbody.append(body)
    output = Path(output_model).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    return {
        "schema": MODEL_MANIFEST_SCHEMA,
        "status": "formal_scene_actual_n_model_generated",
        "config_sha256": sha256(config_path),
        "source_scene_manifest_sha256": sha256(scene_path),
        "source_model_sha256": sha256(source),
        "candidate_model_sha256": sha256(output),
        "rock_count": len(rocks),
        "expected_nq": 19 + 7 * len(rocks),
        "expected_nv": 18 + 6 * len(rocks),
        "expected_nu": 12,
        "rock_id_mapping": [{
            "source_rock_id": rock["source_rock_id"],
            "runtime_rock_id": rock["runtime_rock_id"],
        } for rock in rocks],
        "total_rock_volume_m3": sum(rock["volume_m3"] for rock in rocks),
        "total_rock_mass_kg": sum(rock["mass_kg"] for rock in rocks),
        "time_zero_ready_for_separate_runner": True,
        "time_zero_executed": False,
        "physics_step_executed": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-manifest")
    args = parser.parse_args()
    manifest = build_model(
        args.config, args.scene_manifest, args.repository_root, args.output_model)
    if args.output_manifest:
        Path(args.output_manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
