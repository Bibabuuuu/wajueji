"""Opt-in legacy CAD bucket collider for controlled mixed-scene A/B tests."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np


SCHEMA = "legacy_cad_bucket_collision_candidate_v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_geometry(assembly, mechanism_qpos) -> dict:
    """Extract the same body-local CAD bucket mesh used by the legacy pure-soil run."""
    geometries, _ = assembly.assemble(
        np.asarray(mechanism_qpos, dtype=np.float64), include_meshes=True)
    matches = [item for item in geometries
               if item.get("body") == "bucket" and item.get("mesh") == "bucket"]
    if len(matches) != 1:
        raise ValueError("legacy_cad_bucket_mesh_identity")
    source = matches[0]
    vertices = np.asarray(source["body_local_vertices"], dtype=np.float64)
    faces = np.asarray(source["faces"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in "iu" or not faces.size
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("legacy_cad_bucket_mesh_invalid")
    edges = Counter(tuple(sorted((int(first), int(second))))
                    for a, b, c in faces
                    for first, second in ((a, b), (b, c), (c, a)))
    boundary_edges = sum(count == 1 for count in edges.values())
    nonmanifold_edges = sum(count > 2 for count in edges.values())
    if boundary_edges <= 0 or nonmanifold_edges:
        raise ValueError("legacy_cad_bucket_must_remain_open_manifold")
    mesh_path = Path(source["path"]).resolve()
    return {
        "schema": SCHEMA,
        "is_solid": False,
        "vertices_bucket_m": vertices,
        "faces": faces.astype(np.int32, copy=False),
        "source_mesh_path": str(mesh_path),
        "source_mesh_sha256": _sha256(mesh_path),
        "vertices": int(len(vertices)),
        "faces_count": int(len(faces)),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
        "qualification": "isolated collision candidate; default proxy unchanged",
    }


def public_summary(geometry: dict) -> dict:
    return {key: value for key, value in geometry.items()
            if key not in {"vertices_bucket_m", "faces"}}
