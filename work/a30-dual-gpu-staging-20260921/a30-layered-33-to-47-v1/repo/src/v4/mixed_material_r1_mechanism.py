"""Build an isolated one-rock mechanism candidate without editing frozen XML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from mixed_material_r1 import build_proxy
from mixed_material_r0 import normalized_prototypes


MANIFEST_SCHEMA = "excavation_mixed_rigid_mechanism_candidate_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [element for element in root.iter(tag)
               if element.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected_one:{tag}:{name}:{len(matches)}")
    return matches[0]


def build_candidate(source_model: str | Path, output_model: str | Path,
                    repository_root: str | Path, inventory_path: str | Path,
                    config_path: str | Path,
                    scenario_name: str = "freefall") -> dict:
    source = Path(source_model).resolve()
    output = Path(output_model).resolve()
    repository = Path(repository_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    mechanism = config["mechanism_candidate"]
    scenarios = mechanism.get("unit_scenarios", {})
    if scenario_name not in scenarios:
        raise ValueError("unknown_unit_scenario:" + scenario_name)
    scenario = scenarios[scenario_name]
    if sha256(source) != mechanism["source_model_sha256"]:
        raise ValueError("source_model_sha256_mismatch")
    proxy = build_proxy(inventory, config)
    tree = ET.parse(source)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("missing_model_sections")

    original_bucket_geom = _find_named(root, "geom", "bucket")
    if original_bucket_geom.get("type") != "mesh":
        raise ValueError("unexpected_source_bucket_geom")
    original_bucket_geom.set("contype", "0")
    original_bucket_geom.set("conaffinity", "0")
    original_bucket_geom.set("group", "1")

    bits = mechanism["contact_bits"]
    rock_bit = int(bits["rock"])
    bucket_bit = int(bits["bucket_proxy"])
    ground_bit = int(bits["rock_ground"])
    truck_bit = int(bits["truck_receiver_reserved"])
    values = [rock_bit, bucket_bit, ground_bit, truck_bit]
    if any(value <= 0 or value & (value - 1) for value in values):
        raise ValueError("contact_bits_must_be_unique_positive_powers_of_two")
    if len(set(values)) != len(values):
        raise ValueError("duplicate_contact_bits")

    bucket_body = _find_named(root, "body", "bucket")
    for box in proxy["shell_boxes"] + proxy["tooth_boxes"]:
        ET.SubElement(bucket_body, "geom", {
            "name": "r1_" + box["name"],
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
        path = (repository / "assets" / "v4" / "mixed-material-r1"
                / f'{side["name"]}.obj')
        if not path.is_file():
            raise ValueError("missing_side_plate_asset:" + side["name"])
        mesh_name = "r1_" + side["name"]
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(path)})
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
    ground.set("contype", str(1 | ground_bit))
    ground.set("conaffinity", str(1 | rock_bit))

    fixtures = scenario.get("fixtures", [])
    if scenario.get("fixture") is not None:
        fixtures = [scenario["fixture"]]
    for fixture in fixtures:
        role = fixture.get("contact_role", "ground")
        if fixture["type"] != "box" or role not in ("ground", "truck_receiver"):
            raise ValueError("unsupported_unit_fixture")
        if role == "ground":
            if not fixture["name"].startswith("r1_ground_"):
                raise ValueError("unsupported_ground_fixture_name")
            contact_bit = ground_bit
            rgba = "0.35 0.35 0.35 1"
        else:
            if not fixture["name"].startswith("r1_truck_"):
                raise ValueError("unsupported_truck_fixture_name")
            contact_bit = truck_bit
            rgba = "0.2 0.45 0.75 1"
        ET.SubElement(worldbody, "geom", {
            "name": fixture["name"],
            "type": "box",
            "pos": _numbers(fixture["position_world_m"]),
            "quat": _numbers(fixture["quaternion_wxyz"]),
            "size": _numbers(fixture["half_size_m"]),
            "friction": _numbers(fixture["friction"]),
            "rgba": rgba,
            "contype": str(contact_bit),
            "conaffinity": str(rock_bit),
        })

    if "rocks" in scenario:
        rocks = [dict(value) for value in scenario["rocks"]]
    else:
        rock = dict(mechanism["first_rock"])
        rock["initial_position_world_m"] = scenario["initial_position_world_m"]
        rock["initial_orientation_quaternion_wxyz"] = (
            scenario["initial_orientation_quaternion_wxyz"])
        rocks = [rock]
    if not rocks or len(rocks) > 2:
        raise ValueError("unit_scenario_rock_count")
    rock_records = []
    seen_ids = set()
    for index, rock in enumerate(rocks):
        expected_id = f"r1-rock-{index:03d}"
        if rock["rock_id"] != expected_id or rock["rock_id"] in seen_ids:
            raise ValueError("rock_id_sequence")
        seen_ids.add(rock["rock_id"])
        rock_path = repository / rock["asset_path"]
        if not rock_path.is_file():
            raise ValueError("missing_rock_asset")
        rock_mesh_name = f"r1_rock_mesh_{index:03d}"
        scale = float(rock["equivalent_diameter_m"])
        density = float(rock["density_kg_m3"])
        prototype = normalized_prototypes(
            [rock["prototype_id"]])[rock["prototype_id"]]
        mass = density * float(prototype["unit_volume_m3"]) * scale ** 3
        inertia = density * prototype["unit_inertia_m5"] * scale ** 5
        ET.SubElement(asset, "mesh", {
            "name": rock_mesh_name,
            "file": str(rock_path),
            "scale": _numbers([scale, scale, scale]),
        })
        rock_body = ET.SubElement(worldbody, "body", {
            "name": rock["rock_id"],
            "pos": _numbers(rock["initial_position_world_m"]),
            "quat": _numbers(rock["initial_orientation_quaternion_wxyz"]),
        })
        ET.SubElement(rock_body, "freejoint", {
            "name": f"r1_rock_free_{index:03d}"})
        ET.SubElement(rock_body, "inertial", {
            "pos": "0 0 0",
            "mass": f"{mass:.17g}",
            "fullinertia": _numbers([
                inertia[0, 0], inertia[1, 1], inertia[2, 2],
                inertia[0, 1], inertia[0, 2], inertia[1, 2],
            ]),
        })
        ET.SubElement(rock_body, "geom", {
            "name": f"r1_rock_geom_{index:03d}",
            "type": "mesh",
            "mesh": rock_mesh_name,
            "density": "0",
            "rgba": "0.48 0.31 0.18 1",
            "contype": str(rock_bit),
            "conaffinity": str(rock_bit | bucket_bit | ground_bit | truck_bit),
        })
        rock_records.append({
            **rock,
            "mass_kg": mass,
            "inertia_about_com_kg_m2": inertia.tolist(),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=False)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    source_contract = mechanism["source_model_contract"]
    return {
        "schema": MANIFEST_SCHEMA,
        "source_model_path": mechanism["source_model_path"],
        "source_model_sha256": mechanism["source_model_sha256"],
        "candidate_model_sha256": sha256(output),
        "source_model_contract": source_contract,
        "expected_candidate_contract": {
            "nq": int(source_contract["mechanism_nq"]) + 7 * len(rocks),
            "nv": int(source_contract["mechanism_nv"]) + 6 * len(rocks),
            "nu": int(source_contract["actuator_count"]),
            "nplugin": int(source_contract["plugin_count"]),
            "rock_count": len(rocks),
        },
        "rock": rock_records[0],
        "rocks": rock_records,
        "total_rock_mass_kg": sum(value["mass_kg"] for value in rock_records),
        "unit_scenario": scenario_name,
        "unit_fixture": fixtures[0] if len(fixtures) == 1 else None,
        "unit_fixtures": fixtures,
        "rock_mass_kg": rock_records[0]["mass_kg"],
        "rock_inertia_about_com_kg_m2": (
            rock_records[0]["inertia_about_com_kg_m2"]),
        "contact_bits": bits,
        "bucket_proxy_geom_count": proxy["geom_count"],
        "mouth_center_bucket_m": proxy["mouth_center_bucket_m"],
        "mouth_inward_direction_bucket": (
            proxy["mouth_inward_direction_bucket"]),
        "source_bucket_visual_collision_disabled": True,
        "frozen_source_modified": False,
        "scope_limit": (
            "model assembly candidate only until a separate bridge loads and "
            "steps it with classified contacts and state logging"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--scenario", default="freefall")
    args = parser.parse_args()
    manifest = build_candidate(
        args.source_model, args.output_model, args.repository_root,
        args.inventory, args.config, args.scenario)
    if args.manifest:
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
