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


def material(
    name: str,
    rgb,
    roughness: float = 0.68,
    metallic: float = 0.0,
    double_sided: bool = False,
):
    rgba = [int(np.clip(float(c), 0.0, 1.0) * 255 + 0.5) for c in rgb] + [255]
    return trimesh.visual.material.PBRMaterial(
        name=name,
        baseColorFactor=rgba,
        metallicFactor=metallic,
        roughnessFactor=roughness,
        doubleSided=double_sided,
    )


def make_mesh(
    name: str,
    vertices,
    faces,
    rgb,
    roughness: float = 0.68,
    metallic: float = 0.0,
    double_sided: bool = False,
):
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
    mesh.visual = trimesh.visual.TextureVisuals(
        material=material(name, rgb, roughness, metallic, double_sided)
    )
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
        if mesh_name == "caterpillar":
            style = ((0.045, 0.055, 0.060), 0.88, 0.04)
        elif mesh_name.endswith("_rod"):
            style = ((0.72, 0.78, 0.82), 0.12, 0.92)
        elif mesh_name == "bucket":
            style = ((0.16, 0.18, 0.20), 0.42, 0.68)
        elif mesh_name.endswith("_tube"):
            style = ((0.94, 0.48, 0.015), 0.25, 0.22)
        else:
            style = ((1.0, 0.56, 0.018), 0.27, 0.18)
        parts.append(
            (
                name,
                make_mesh(name, geom["vertices"], geom["faces"], *style),
            )
        )
    return parts


def soil_surface_parts(points, step_m: float = 0.025):
    points = np.asarray(points, dtype=float)
    low = points[:, :2].min(axis=0)
    high = points[:, :2].max(axis=0)
    shape = np.ceil((high - low) / step_m).astype(int) + 1
    ij = np.floor((points[:, :2] - low) / step_m).astype(int)
    ij = np.minimum(ij, shape - 1)
    height = np.full(tuple(shape), -np.inf, dtype=float)
    np.maximum.at(height, (ij[:, 0], ij[:, 1]), points[:, 2])

    # Crop the one-cell numerical rim and fill only rare internal sampling
    # holes from immediate neighbours.  The actual cell maxima remain intact.
    valid = np.isfinite(height)
    occupied_i = np.flatnonzero(valid.any(axis=1))
    occupied_j = np.flatnonzero(valid.any(axis=0))
    height = height[
        occupied_i[0] : occupied_i[-1] + 1,
        occupied_j[0] : occupied_j[-1] + 1,
    ]
    x0 = low[0] + occupied_i[0] * step_m
    y0 = low[1] + occupied_j[0] * step_m
    for _ in range(4):
        missing = ~np.isfinite(height)
        if not missing.any():
            break
        total = np.zeros_like(height)
        count = np.zeros_like(height, dtype=np.int16)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted = np.roll(height, (di, dj), axis=(0, 1))
            good = np.isfinite(shifted)
            total[good] += shifted[good]
            count[good] += 1
        fill = missing & (count >= 2)
        height[fill] = total[fill] / count[fill]
    if not np.isfinite(height).all():
        raise RuntimeError("soil surface contains unresolved sampling holes")

    ni, nj = height.shape
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    vertices = np.column_stack(
        (
            x0 + ii.ravel() * step_m,
            y0 + jj.ravel() * step_m,
            height.ravel(),
        )
    )
    index = np.arange(ni * nj, dtype=np.int32).reshape(ni, nj)
    faces = []
    for i in range(ni - 1):
        for j in range(nj - 1):
            a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
            faces.extend(((a, b, c), (a, c, d)))
    faces = np.asarray(faces, dtype=np.int32)

    centroids = vertices[faces].mean(axis=1)
    tone = (
        np.sin(centroids[:, 0] * 8.7)
        + np.cos(centroids[:, 1] * 10.3)
        + np.sin((centroids[:, 0] + centroids[:, 1]) * 5.1)
    ) / 3.0
    bands = np.digitize(tone, (-0.24, 0.22))
    styles = [
        ((0.285, 0.125, 0.045), 0.98),
        ((0.390, 0.190, 0.070), 0.96),
        ((0.515, 0.285, 0.120), 0.94),
    ]
    result = []
    for band, (rgb, roughness) in enumerate(styles):
        name = f"混合料堆/土体表面-{band + 1}"
        result.append((name, make_mesh(name, vertices, faces[bands == band], rgb, roughness)))

    perimeter = []
    perimeter.extend(index[:, 0].tolist())
    perimeter.extend(index[-1, 1:].tolist())
    perimeter.extend(index[-2::-1, -1].tolist())
    perimeter.extend(index[0, -2:0:-1].tolist())
    perimeter = np.asarray(perimeter, dtype=np.int32)
    top = vertices[perimeter]
    bottom = top.copy()
    bottom[:, 2] = 0.0
    side_vertices = np.vstack((top, bottom))
    count = len(perimeter)
    side_faces = []
    for i in range(count):
        j = (i + 1) % count
        side_faces.extend(((i, count + i, count + j), (i, count + j, j)))
    result.append(
        (
            "混合料堆/土体侧壁",
            make_mesh(
                "混合料堆/土体侧壁",
                side_vertices,
                side_faces,
                (0.245, 0.105, 0.035),
                0.99,
            ),
        )
    )
    return result


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
                make_mesh(name, vertices, faces, record["color_rgb"], 0.97, 0.02),
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
    styles = [
        ((1.0, 0.56, 0.015), 0.25, 0.18),
        ((0.035, 0.042, 0.048), 0.9, 0.02),
        ((0.58, 0.61, 0.64), 0.23, 0.82),
        ((1.0, 0.54, 0.01), 0.3, 0.14),
        ((0.69, 0.71, 0.73), 0.32, 0.72),
        ((0.018, 0.022, 0.026), 0.2, 0.35),
        ((0.86, 0.88, 0.9), 0.22, 0.62),
    ]
    result = []
    for part_id, (part_name, style) in enumerate(zip(names, styles)):
        selected = faces[part_ids == part_id]
        used = np.unique(selected)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        name = f"固定装载车/{part_id:02d}-{part_name}"
        result.append((name, make_mesh(name, vertices[used], remap[selected], *style)))
    return result


def ground_mesh():
    mesh = trimesh.creation.box(extents=(11.0, 10.0, 0.06))
    mesh.apply_translation((2.3, 2.0, -0.04))
    mesh.visual = trimesh.visual.TextureVisuals(
        material=material("作业地面", (0.115, 0.135, 0.155), 0.86, 0.02)
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
    surface = soil_surface_parts(checkpoint["particle_q"])
    rocks = rock_parts(checkpoint["rigid_ids"], checkpoint["rigid_qpos"])
    truck = truck_parts()
    ground = ("环境/作业地面", ground_mesh())

    provenance = {
        "title": PROJECT_NAME,
        "deliverable": "enhanced_static_visual_glb",
        "units": "metre",
        "up_axis": "Y",
        "source_checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_cycle_index": meta["cycle_index"],
        "checkpoint_stage": meta["stage"],
        "source_truck_mesh_sha256": sha256(TRUCK_MESH),
        "source_truck_contract_sha256": sha256(TRUCK_CONTRACT),
        "soil_surface_note": "25 mm cell-max surface reconstructed from frozen MPM particles; three PBR earth tones plus closed side wall",
        "material_upgrade": "PBR differentiation for painted metal, chrome rods, rubber tracks and tyres, steel bucket, truck metal, soil, and rock",
        "rock_count": len(rocks),
        "truck_note": "fixed project receiver pose; checkpoint itself has truck_enabled=false",
    }

    scenes = {
        "complete": scene_from([ground, *surface, *rocks, *truck, *excavator], provenance),
        "excavator": scene_from(excavator, provenance),
        "workface": scene_from([*surface, *rocks], provenance),
    }
    reports = []
    for key, scene in scenes.items():
        target = args.output / OUTPUT_NAMES[key]
        export_scene(scene, target)
        reports.append(validate_glb(target))

    build_report = {
        "schema": "submission_glb_build_report_v2",
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
