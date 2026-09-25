#!/usr/bin/env python3
"""Build self-contained GLB submission models from frozen project geometry.

The export is a static visual deliverable.  It does not alter or advance the
physics state.  Project world coordinates are Z-up; glTF output is converted
to the standard Y-up convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


REPO = Path("/home/qunwang/wajueji")
MECHANISM_ROOT = Path("/home/qunwang/codex-work/small-soil-20260910-v17-performance")
CHECKPOINT = REPO / "runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz"
CHECKPOINT_META = CHECKPOINT.with_suffix(".npz.json")
SCENE_MANIFEST = REPO / "results/layered-mound-physical-preflight-v1/scene-manifest.json"
ROCK_ASSETS = REPO / "assets/v4/mixed-material-r0"
TRUCK_MESH = REPO / "evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz"
TRUCK_CONTRACT = REPO / "evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json"

PROJECT_NAME = "复杂介质工况自主挖掘"
GROUP_NAME = "高校组"

OUTPUT_NAMES = {
    "complete": f"三维模型-{GROUP_NAME}-{PROJECT_NAME}-完整作业场景.glb",
    "excavator": f"三维模型-{GROUP_NAME}-{PROJECT_NAME}-挖掘机机构.glb",
    "workface": f"三维模型-{GROUP_NAME}-{PROJECT_NAME}-混合土石料堆.glb",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def material(name: str, rgb, roughness: float = 0.68):
    rgba = [int(np.clip(float(c), 0.0, 1.0) * 255 + 0.5) for c in rgb] + [255]
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=rgba,
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )


def make_mesh(name: str, vertices, faces, rgb, roughness: float = 0.68):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    source_triangle = vertices[faces].astype(np.float64)
    source_cross = np.cross(
        source_triangle[:, 1] - source_triangle[:, 0],
        source_triangle[:, 2] - source_triangle[:, 0],
    )
    faces = faces[np.linalg.norm(source_cross, axis=1) > 1e-15]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    vertices = vertices[used]
    faces = remap[faces]
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    # Some source CAD parts contain isolated degenerate triangles.  Compute
    # robust area-weighted normals and give vertices supported only by
    # degenerate faces a valid display normal.  Geometry is left unchanged.
    triangle = vertices[faces].astype(np.float64)
    face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    vertex_normal = np.zeros((len(vertices), 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normal, faces[:, corner], face_normal)
    length = np.linalg.norm(vertex_normal, axis=1)
    good = length > 1e-15
    vertex_normal[good] /= length[good, None]
    vertex_normal[~good] = (0.0, 0.0, 1.0)
    mesh.vertex_normals = vertex_normal
    mesh.visual = trimesh.visual.TextureVisuals(material=material(name, rgb, roughness))
    return mesh


def z_up_to_gltf_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    # Rotation of -90 degrees about X: (x, y, z) -> (x, z, -y).
    result.apply_transform(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    return result


def add(scene: trimesh.Scene, name: str, mesh: trimesh.Trimesh):
    scene.add_geometry(
        z_up_to_gltf_y_up(mesh),
        node_name=name,
        geom_name=name,
    )


def excavator_parts(mechanism_qpos):
    sys.path.insert(0, str(MECHANISM_ROOT))
    from scene_assembly import Assembler

    assembler = Assembler(
        MECHANISM_ROOT / "scene/input/model/A.xml",
        MECHANISM_ROOT / "scene/input/mesh",
    )
    geoms, _ = assembler.assemble(mechanism_qpos, include_meshes=True)
    parts = []
    for index, geom in enumerate(geoms):
        body = str(geom["body"])
        mesh_name = str(geom["mesh"])
        name = f"挖掘机/{index:02d}-{body}/{mesh_name}"
        rgb = np.asarray(geom["rgba"], dtype=float)[:3]
        parts.append((name, make_mesh(name, geom["vertices"], geom["faces"], rgb, 0.55)))
    return parts


def soil_surface(points, step_m: float = 0.04):
    points = np.asarray(points, dtype=float)
    low = points[:, :2].min(axis=0)
    high = points[:, :2].max(axis=0)
    shape = np.ceil((high - low) / step_m).astype(int) + 1
    ij = np.floor((points[:, :2] - low) / step_m).astype(int)
    ij = np.minimum(ij, shape - 1)
    height = np.full(tuple(shape), -np.inf, dtype=float)
    np.maximum.at(height, (ij[:, 0], ij[:, 1]), points[:, 2])

    # The particle lattice leaves only a one-cell numerical rim empty.  Keep
    # the measured cells and triangulate quads whose four corners are present.
    valid = np.isfinite(height)
    index = np.full(tuple(shape), -1, dtype=np.int32)
    cells = np.argwhere(valid)
    index[valid] = np.arange(len(cells), dtype=np.int32)
    vertices = np.column_stack(
        (
            low[0] + cells[:, 0] * step_m,
            low[1] + cells[:, 1] * step_m,
            height[valid],
        )
    )
    faces = []
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            ids = [index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]]
            if min(ids) >= 0:
                a, b, c, d = ids
                faces.extend(((a, b, c), (a, c, d)))
    return make_mesh("土体表面", vertices, faces, (0.42, 0.245, 0.105), 0.92)


def quaternion_matrix_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=float)
    norm = w * w + x * x + y * y + z * z
    if norm < 1e-15:
        return np.eye(3)
    scale = 2.0 / norm
    return np.array(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ]
    )


def rock_parts(rigid_ids, rigid_qpos):
    manifest = json.loads(SCENE_MANIFEST.read_text())
    records = manifest["rocks"][: len(rigid_ids)]
    if len(records) != len(rigid_ids):
        raise RuntimeError("rock manifest does not cover frozen checkpoint")
    prototypes = {}
    parts = []
    for runtime_id, qpos, record in zip(rigid_ids, rigid_qpos, records):
        prototype_id = record["prototype_id"]
        if prototype_id not in prototypes:
            prototype = trimesh.load(
                ROCK_ASSETS / f"{prototype_id}.obj",
                force="mesh",
                process=False,
            )
            prototypes[prototype_id] = (
                np.asarray(prototype.vertices, dtype=float),
                np.asarray(prototype.faces, dtype=np.int32),
            )
        source_vertices, faces = prototypes[prototype_id]
        diameter = float(record["equivalent_diameter_m"])
        rotation = quaternion_matrix_wxyz(qpos[3:7])
        vertices = (source_vertices * diameter) @ rotation.T + qpos[:3]
        name = f"岩石/{runtime_id}-{prototype_id}"
        parts.append(
            (
                name,
                make_mesh(name, vertices, faces, record["color_rgb"], 0.88),
            )
        )
    return parts


def truck_parts():
    frozen = np.load(TRUCK_MESH, allow_pickle=False)
    vertices = frozen["world_vertices"]
    faces = frozen["faces"]
    part_ids = frozen["part_ids"]
    # Insertion order and face counts are frozen by the original OBJ/MTL parser.
    names = ["YellowShine", "tyre", "GreyShine", "yaller", "pearl", "onyx", "ecru"]
    colors = [
        (0.972549, 0.639216, 0.0),
        (0.188235, 0.188235, 0.188235),
        (0.584314, 0.584314, 0.584314),
        (0.972549, 0.639216, 0.0),
        (0.584314, 0.584314, 0.584314),
        (0.0, 0.0, 0.0),
        (0.996078, 0.929412, 0.796078),
    ]
    result = []
    for part_id, (part_name, rgb) in enumerate(zip(names, colors)):
        selected = faces[part_ids == part_id]
        used = np.unique(selected)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        name = f"固定装载车/{part_id:02d}-{part_name}"
        result.append((name, make_mesh(name, vertices[used], remap[selected], rgb, 0.55)))
    return result


def ground_mesh():
    mesh = trimesh.creation.box(extents=(11.0, 10.0, 0.06))
    mesh.apply_translation((2.3, 2.0, -0.04))
    mesh.visual = trimesh.visual.TextureVisuals(
        material=material("作业地面", (0.19, 0.22, 0.25), 0.96)
    )
    return mesh


def scene_from(parts, extras):
    scene = trimesh.Scene(base_frame="项目场景")
    for name, mesh in parts:
        add(scene, name, mesh)
    scene.metadata.update(extras)
    return scene


def export_scene(scene: trimesh.Scene, destination: Path):
    destination.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))


def validate_glb(path: Path):
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"invalid GLB magic: {path}")
    loaded = trimesh.load(path, force="scene", process=False)
    if not loaded.geometry:
        raise RuntimeError(f"no geometry in {path}")
    vertices = 0
    faces = 0
    for geom in loaded.geometry.values():
        v = np.asarray(geom.vertices)
        f = np.asarray(geom.faces)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
            raise RuntimeError(f"invalid vertices in {path}")
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise RuntimeError(f"invalid faces in {path}")
        if f.min() < 0 or f.max() >= len(v):
            raise RuntimeError(f"out-of-range indices in {path}")
        vertices += len(v)
        faces += len(f)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "geometry_count": len(loaded.geometry),
        "node_count": len(loaded.graph.nodes_geometry),
        "vertices": vertices,
        "triangles": faces,
        "bounds_gltf_y_up_m": np.asarray(loaded.bounds).tolist(),
        "extents_m": np.asarray(loaded.extents).tolist(),
        "valid": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    meta = json.loads(CHECKPOINT_META.read_text())
    excavator = excavator_parts(checkpoint["mechanism_qpos"])
    surface = ("混合料堆/土体表面", soil_surface(checkpoint["particle_q"]))
    rocks = rock_parts(checkpoint["rigid_ids"], checkpoint["rigid_qpos"])
    truck = truck_parts()
    ground = ("环境/作业地面", ground_mesh())

    provenance = {
        "title": PROJECT_NAME,
        "deliverable": "static_visual_glb",
        "units": "metre",
        "up_axis": "Y",
        "source_checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_cycle_index": meta["cycle_index"],
        "checkpoint_stage": meta["stage"],
        "source_truck_mesh_sha256": sha256(TRUCK_MESH),
        "source_truck_contract_sha256": sha256(TRUCK_CONTRACT),
        "soil_surface_note": "40 mm cell-max visual surface reconstructed from frozen MPM particles",
        "rock_count": len(rocks),
        "truck_note": "fixed project receiver pose; checkpoint itself has truck_enabled=false",
    }

    scenes = {
        "complete": scene_from([ground, surface, *rocks, *truck, *excavator], provenance),
        "excavator": scene_from(excavator, provenance),
        "workface": scene_from([surface, *rocks], provenance),
    }
    reports = []
    for key, scene in scenes.items():
        target = args.output / OUTPUT_NAMES[key]
        export_scene(scene, target)
        reports.append(validate_glb(target))

    build_report = {
        "schema": "submission_glb_build_report_v1",
        "status": "validated",
        "source": provenance,
        "models": reports,
        "portal_limit_bytes_per_file": 200 * 1024 * 1024,
        "all_files_below_limit": all(r["bytes"] <= 200 * 1024 * 1024 for r in reports),
    }
    (args.output / "GLB模型校验报告.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "SHA256SUMS.txt").write_text(
        "".join(f'{r["sha256"]}  {r["file"]}\n' for r in reports)
    )
    print(json.dumps(build_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
