#!/usr/bin/env python3
"""Build a read-only rear lower-face prepoint plan; never step physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from layered_excavation_approach_planner import (
    _load_kinematics,
    build_fused_terrain,
)
from planning.cycle import select_cut
from planning.rear_lower_face import select_rear_lower_face
from planning.rear_lower_prepoint import plan_rear_lower_prepoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(
        array, dtype=np.float64).tobytes()).hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build(args) -> dict:
    repository = Path(args.repository_root).resolve()
    checkpoint = Path(args.source_checkpoint).resolve()
    manifest_path = Path(args.scene_manifest).resolve()
    proxy_path = Path(args.bucket_proxy).resolve()
    truck_path = Path(args.truck_mesh).resolve()
    with np.load(checkpoint, allow_pickle=False) as source:
        particles = source["particle_q"].copy()
        rigid_qpos = source["rigid_qpos"].copy()
        rigid_ids = source["rigid_ids"].tolist()
        mechanism_qpos = source["mechanism_qpos"].copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # This diagnostic bundle stores runtime labels (r1-rock-NNN), while the
    # bound manifest retains semantic layer IDs.  The 47-rock construction is
    # the manifest order with the later top layer excluded, as in the audited
    # restore runner.
    records = [record for record in manifest["rocks"]
               if record["layer_id"] != "top"]
    if (len(records) != len(rigid_ids)
            or rigid_ids != [f"r1-rock-{index:03d}"
                             for index in range(len(records))]):
        raise ValueError("rear_prepoint_rigid_order_contract")
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    with np.load(truck_path, allow_pickle=False) as truck:
        truck_vertices = truck["world_vertices"].copy()
    terrain, observation = build_fused_terrain(
        repository, records, particles, rigid_qpos)
    strip = select_cut(terrain, bucket_width=0.60)
    if strip != observation["selection"]:
        raise ValueError("rear_prepoint_strip_selection_drift")
    contact = select_rear_lower_face(terrain, strip)
    assembly, chain = _load_kinematics(
        args.kinematics_python_directory,
        args.kinematics_model_path,
        args.kinematics_mesh_directory)
    path = plan_rear_lower_prepoint(
        terrain=terrain,
        contact_selection=contact,
        operator_qpos=mechanism_qpos,
        open_bucket_proxy=proxy,
        truck_world_vertices=truck_vertices,
        assembly=assembly,
        chain=chain)
    qpos = path.pop("qpos")
    velocity = path.pop("velocity")
    return _clean({
        "schema": "layered_47_rear_lower_prepoint_read_only_plan_v1",
        "status": path["status"],
        "source": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "scene_manifest_path": str(manifest_path),
            "scene_manifest_sha256": _sha256(manifest_path),
            "bucket_proxy_path": str(proxy_path),
            "bucket_proxy_sha256": _sha256(proxy_path),
            "truck_mesh_path": str(truck_path),
            "truck_mesh_sha256": _sha256(truck_path),
        },
        "strip_selection": strip,
        "contact_selection": contact,
        "path": path,
        "path_arrays": {
            "qpos_shape": list(qpos.shape),
            "qpos_float64_sha256": _array_sha256(qpos),
            "velocity_shape": list(velocity.shape),
            "velocity_float64_sha256": _array_sha256(velocity),
            "commands_saved": False,
        },
        "scope": {
            "observation_only": True,
            "physics_stepped": False,
            "checkpoint_mutated": False,
            "digging_executed": False,
            "manual_or_bc_rl_contact_required": True,
            "training_eligible": False,
            "execution_approved": False,
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--bucket-proxy", required=True)
    parser.add_argument("--truck-mesh", required=True)
    parser.add_argument("--kinematics-python-directory", required=True)
    parser.add_argument("--kinematics-model-path", required=True)
    parser.add_argument("--kinematics-mesh-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
