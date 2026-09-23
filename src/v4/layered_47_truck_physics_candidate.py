#!/usr/bin/env python3
"""Scoped S07 truck collision candidate shared by Newton and MuJoCo.

The five boxes are expressed in the frozen receiver's local frame.  The same
world transforms are installed as Newton static particle colliders and MuJoCo
rock colliders.  TruckReceiver remains a read-only inventory observer.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CONFIG_SCHEMA = "excavation_layered_47_truck_physics_config_v1"
CONFIG_STATUS = "cad_aligned_five_box_development_candidate"
FIXTURE_NAMES = (
    "r1_truck_floor",
    "r1_truck_wall_x_positive",
    "r1_truck_wall_x_negative",
    "r1_truck_wall_y_negative",
    "r1_truck_wall_y_positive",
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def _quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    """Convert a proper rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-10)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-10)):
        raise ValueError("truck_fixture_rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1]
                              - matrix[2, 2]) * 2.0
            quaternion = np.asarray([
                (matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0]
                              - matrix[2, 2]) * 2.0
            quaternion = np.asarray([
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0]
                              - matrix[1, 1]) * 2.0
            quaternion = np.asarray([
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _validate_source(binding: dict, root: Path, label: str) -> Path:
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or not isinstance(binding["sha256"], str)
            or len(binding["sha256"]) != 64):
        raise ValueError("truck_source_binding:" + label)
    path = root / binding["path"]
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise ValueError("truck_source_mismatch:" + label)
    return path


def load_candidate(config_path: str | Path,
                   repository_root: str | Path) -> tuple[dict, list[dict]]:
    root = Path(repository_root).resolve()
    path = Path(config_path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("schema") != CONFIG_SCHEMA
            or document.get("status") != CONFIG_STATUS):
        raise ValueError("truck_candidate_schema_or_status")
    frozen = document.get("frozen_inputs", {})
    if set(frozen) != {"receiver_contract", "receiver_mesh",
                       "collision_protocol"}:
        raise ValueError("truck_candidate_frozen_inputs")
    contract_path = _validate_source(
        frozen["receiver_contract"], root, "receiver_contract")
    _validate_source(frozen["receiver_mesh"], root, "receiver_mesh")
    protocol_path = _validate_source(
        frozen["collision_protocol"], root, "collision_protocol")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "truck_receiver_triangles_v1":
        raise ValueError("truck_receiver_contract_schema")
    transform = np.asarray(contract.get("local_to_world"), dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3],
                               np.eye(3), atol=1.0e-10)):
        raise ValueError("truck_receiver_transform")
    bits = protocol.get("mechanism_candidate", {}).get("contact_bits", {})
    if bits != {"rock": 256, "bucket_proxy": 512,
                "rock_ground": 1024, "truck_receiver_reserved": 2048}:
        raise ValueError("truck_contact_bits")
    ownership = document.get("contact_ownership")
    if ownership != {
            "mpm_soil_to_truck": "newton_static_boxes",
            "rigid_rock_to_truck": "mujoco_static_boxes",
            "truck_receiver": "read_only_inventory_observer",
            "duplicate_soil_contact_backend": False,
            "duplicate_rock_contact_backend": False}:
        raise ValueError("truck_contact_ownership")
    raw = document.get("fixtures_receiver_local")
    if (not isinstance(raw, list) or len(raw) != len(FIXTURE_NAMES)
            or tuple(item.get("name") for item in raw) != FIXTURE_NAMES):
        raise ValueError("truck_fixture_names")
    material = document.get("material", {})
    friction = float(material.get("friction", math.nan))
    mujoco_friction = np.asarray(
        material.get("mujoco_friction"), dtype=np.float64)
    if (not math.isfinite(friction) or friction <= 0.0
            or mujoco_friction.shape != (3,)
            or not np.isfinite(mujoco_friction).all()
            or np.any(mujoco_friction < 0.0)):
        raise ValueError("truck_fixture_friction")
    fixtures = []
    for item in raw:
        center = np.asarray(item.get("center_m"), dtype=np.float64)
        half = np.asarray(item.get("half_size_m"), dtype=np.float64)
        pitch = float(item.get("pitch_rad", math.nan))
        if (center.shape != (3,) or half.shape != (3,)
                or not np.isfinite(center).all() or not np.isfinite(half).all()
                or np.any(half <= 0.0) or not math.isfinite(pitch)):
            raise ValueError("truck_fixture_numeric:" + item["name"])
        local_rotation = _rotation_y(pitch)
        world_rotation = transform[:3, :3] @ local_rotation
        fixtures.append({
            "name": item["name"],
            "position_world_m": (
                transform[:3, :3] @ center + transform[:3, 3]).tolist(),
            "quaternion_wxyz": _quaternion_wxyz(world_rotation),
            "rotation_world": world_rotation.tolist(),
            "half_size_m": half.tolist(),
            "newton_friction": friction,
            "mujoco_friction": mujoco_friction.tolist(),
            "contact_role": "truck_receiver",
        })
    return document, fixtures


def clearance(points: np.ndarray, fixtures: list[dict]) -> dict:
    """Return point-centre clearance to each oriented closed box."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("truck_clearance_points")
    by_fixture = {}
    inside_total = np.zeros(len(values), dtype=bool)
    minimum = math.inf
    for fixture in fixtures:
        center = np.asarray(fixture["position_world_m"], dtype=np.float64)
        rotation = np.asarray(fixture["rotation_world"], dtype=np.float64)
        half = np.asarray(fixture["half_size_m"], dtype=np.float64)
        local = (values - center) @ rotation
        exterior = np.maximum(np.abs(local) - half, 0.0)
        distance = np.linalg.norm(exterior, axis=1)
        inside = np.all(np.abs(local) <= half, axis=1)
        inside_total |= inside
        candidate = float(distance.min()) if len(distance) else math.inf
        minimum = min(minimum, candidate)
        by_fixture[fixture["name"]] = {
            "inside_point_count": int(inside.sum()),
            "minimum_point_center_clearance_m": candidate,
        }
    return {
        "inside_any_fixture_count": int(inside_total.sum()),
        "minimum_point_center_clearance_m": minimum,
        "by_fixture": by_fixture,
    }


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def append_mujoco_fixtures(model_path: str | Path, fixtures: list[dict],
                            contact_bits: dict) -> dict:
    path = Path(model_path)
    tree = ET.parse(path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("truck_model_worldbody")
    if any((geom.get("name") or "").startswith("r1_truck_")
           for geom in root.iter("geom")):
        raise ValueError("truck_model_duplicate_fixture")
    rock_bit = int(contact_bits["rock"])
    truck_bit = int(contact_bits["truck_receiver_reserved"])
    rock_geoms = [geom for geom in root.iter("geom")
                  if (geom.get("name") or "").startswith("r1_rock_geom_")]
    if not rock_geoms or any(
            int(geom.get("contype", "0")) != rock_bit
            or not (int(geom.get("conaffinity", "0")) & truck_bit)
            for geom in rock_geoms):
        raise ValueError("truck_model_rock_affinity")
    for fixture in fixtures:
        ET.SubElement(worldbody, "geom", {
            "name": fixture["name"],
            "type": "box",
            "pos": _numbers(fixture["position_world_m"]),
            "quat": _numbers(fixture["quaternion_wxyz"]),
            "size": _numbers(fixture["half_size_m"]),
            "friction": _numbers(fixture["mujoco_friction"]),
            "rgba": "0.2 0.45 0.75 0",
            "group": "3",
            "contype": str(truck_bit),
            "conaffinity": str(rock_bit),
        })
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    return {
        "fixture_count": len(fixtures),
        "fixture_names": [item["name"] for item in fixtures],
        "rock_contact_bit": rock_bit,
        "truck_contact_bit": truck_bit,
        "model_sha256": sha256(path),
    }


@contextmanager
def installed_candidate(config_path: str | Path, repository_root: str | Path):
    """Install the MuJoCo half of the candidate for one scoped model build."""
    import layered_mound_checkpoint_runtime as runtime
    document, fixtures = load_candidate(config_path, repository_root)
    protocol_path = (Path(repository_root).resolve()
                     / document["frozen_inputs"]["collision_protocol"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bits = protocol["mechanism_candidate"]["contact_bits"]
    original = runtime._build_provisional_model

    def build(*args, **kwargs):
        manifest = original(*args, **kwargs)
        output = kwargs.get("output") if "output" in kwargs else args[3]
        model = append_mujoco_fixtures(output, fixtures, bits)
        return {
            **manifest,
            "model_sha256": model["model_sha256"],
            "truck_physics_candidate": {
                **model,
                "config_path": str(Path(config_path).resolve()),
                "config_sha256": sha256(config_path),
                "soil_owner": "newton",
                "rock_owner": "mujoco",
                "inventory_observer": "TruckReceiver",
            },
        }

    runtime._build_provisional_model = build
    try:
        yield copy.deepcopy(fixtures)
    finally:
        runtime._build_provisional_model = original


__all__ = [
    "CONFIG_SCHEMA", "CONFIG_STATUS", "FIXTURE_NAMES", "append_mujoco_fixtures",
    "clearance", "installed_candidate", "load_candidate", "sha256",
]
