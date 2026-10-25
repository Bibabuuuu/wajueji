#!/usr/bin/env python3
"""Render an exact layered-mound checkpoint in the existing scene viewer.

This entry is deliberately read-only.  It restores particle positions and
rock poses for display, builds the already accepted excavator/truck geometry,
and never advances either the MPM solver or the rigid-body bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


CHECKPOINT_SCHEMA = "excavation_layered_mound_continuation_checkpoint_v2"
EXPECTED_STATUS = "four_layer_continuation_checkpoint_saved"
EXPECTED_PARTICLE_COUNT = 256441
EXPECTED_ROCK_COUNT = 47
EXPECTED_FIELDS = (
    "particle_q",
    "particle_qd",
    "particle_f",
    "particle_flags",
    "particle_elastic_strain",
    "particle_transform",
    "particle_qd_grad",
    "particle_stress",
    "particle_Jp",
    "rigid_ids",
    "rigid_qpos",
    "rigid_qvel",
    "rigid_reported_linear_velocity",
    "rigid_reported_angular_velocity",
    "simulation_time_s",
    "grid_reference_particle_q",
    "solver_grid_bounds_lo",
    "solver_grid_bounds_hi",
    "solver_grid_resolution",
    "solver_ws_impulse",
    "solver_ws_stress",
    "solver_body_q_prev",
)
LAYERS = ("lower", "lower_middle", "middle", "upper")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(target),
    }


def load_checkpoint_readonly(checkpoint_path: str | Path,
                             metadata_path: str | Path,
                             expected_sha256: str) -> tuple[dict, dict, dict]:
    checkpoint = Path(checkpoint_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    before = file_identity(checkpoint)
    if before["sha256"] != expected_sha256:
        raise ValueError(
            f"checkpoint_sha256:{before['sha256']}:{expected_sha256}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    required_metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "status": EXPECTED_STATUS,
        "checkpoint_sha256": expected_sha256,
        "particle_count": EXPECTED_PARTICLE_COUNT,
        "rock_count": EXPECTED_ROCK_COUNT,
        "newton_version": "1.5.1",
        "warp_version": "1.17.0",
        "fixed_grid_reference_saved": True,
        "newton_solver_warmstart_saved": True,
    }
    for name, expected in required_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"checkpoint_metadata_contract:{name}:"
                f"{metadata.get(name)!r}:{expected!r}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        fields = tuple(sorted(archive.files))
        if fields != tuple(sorted(EXPECTED_FIELDS)):
            raise ValueError("checkpoint_field_set")
        arrays = {
            "particle_q": archive["particle_q"].copy(),
            "rigid_ids": archive["rigid_ids"].copy(),
            "rigid_qpos": archive["rigid_qpos"].copy(),
            "simulation_time_s": archive["simulation_time_s"].copy(),
        }
    if arrays["particle_q"].shape != (EXPECTED_PARTICLE_COUNT, 3):
        raise ValueError("checkpoint_particle_q_shape")
    if arrays["rigid_qpos"].shape != (EXPECTED_ROCK_COUNT, 7):
        raise ValueError("checkpoint_rigid_qpos_shape")
    if arrays["simulation_time_s"].shape != (1,):
        raise ValueError("checkpoint_simulation_time_shape")
    expected_ids = np.asarray(
        [f"r1-rock-{index:03d}" for index in range(EXPECTED_ROCK_COUNT)])
    if not np.array_equal(arrays["rigid_ids"].astype(str), expected_ids):
        raise ValueError("checkpoint_rigid_ids")
    if (not all(np.isfinite(value).all() for key, value in arrays.items()
                if key != "rigid_ids")
            or float(arrays["simulation_time_s"][0]) <= 0.0):
        raise ValueError("checkpoint_nonfinite_or_time")
    if float(metadata.get("simulation_time_s", -1.0)) != float(
            arrays["simulation_time_s"][0]):
        raise ValueError("checkpoint_metadata_simulation_time")
    after = file_identity(checkpoint)
    if after != before:
        raise RuntimeError("checkpoint_changed_while_loading")
    return arrays, metadata, before


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            for index in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _spherical_uv(vertices: np.ndarray) -> np.ndarray:
    centered = vertices.astype(float) - vertices.mean(axis=0)
    radius = np.maximum(np.linalg.norm(centered, axis=1), 1e-9)
    u = 0.5 + np.arctan2(centered[:, 1], centered[:, 0]) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(np.clip(centered[:, 2] / radius, -1.0, 1.0)) / np.pi
    return np.column_stack((u, v)).astype(np.float32)


def _stone_texture(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    coarse = (0.070 * np.sin(xx * 0.105 + yy * 0.037 + seed)
              + 0.045 * np.sin(xx * 0.031 - yy * 0.121 + seed * 0.31)
              + 0.028 * np.cos(xx * 0.247 + yy * 0.193))
    fine = rng.normal(0.0, 0.028, size=(size, size))
    grain = np.clip(0.86 + coarse + fine, 0.62, 1.0)
    rgb = np.stack((grain * 1.02, grain, grain * 0.94), axis=-1)
    return np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _truck_parts(root: Path):
    colors, current = {}, "default"
    for raw in (root / "EarthMover.mtl").read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "newmtl":
            current = fields[1]
        elif fields[0] == "Kd":
            colors[current] = tuple(map(float, fields[1:4]))
    vertices, parts, current = [], {}, "default"
    for raw in (root / "EarthMover.obj").read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(list(map(float, fields[1:4])))
        elif fields[0] == "usemtl":
            current = fields[1]
        elif fields[0] == "f":
            ids = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            for index in range(1, len(ids) - 1):
                parts.setdefault(current, []).append([ids[0], ids[index], ids[index + 1]])
    source = np.asarray(vertices)
    vertices_world = source[:, [2, 0, 1]] * 0.036
    vertices_world[:, 0] -= (-84.802605 + 35.112) / 2.0 * 0.036
    vertices_world[:, 1] += 4.5
    vertices_world[:, 2] -= vertices_world[:, 2].min()
    xy = vertices_world[:, :2].copy()
    vertices_world[:, 0] = 4.5 - xy[:, 1]
    vertices_world[:, 1] = 4.5 + xy[:, 0]
    vertices_world = ((vertices_world - np.asarray([0.0, 4.5, 0.0])) * 0.87
                      + np.asarray([0.0, 4.5, 0.0]))
    angle = -np.pi / 4.0
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    vertices_world = vertices_world @ rotation.T
    for name, faces in parts.items():
        faces = np.asarray(faces, dtype=np.int32)
        ids = np.unique(faces)
        remap = np.full(len(vertices_world), -1, dtype=np.int32)
        remap[ids] = np.arange(len(ids))
        yield name, vertices_world[ids], remap[faces], colors.get(name, (0.6, 0.6, 0.6))


def _write_ppm(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        stream.write(np.ascontiguousarray(pixels).tobytes())


def _records(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [item for item in manifest.get("rocks", [])
               if item.get("layer_id") in LAYERS]
    if (manifest.get("rock_count") != 65 or len(records) != EXPECTED_ROCK_COUNT
            or [item["rock_id"] for item in records] != [
                f"{layer}-{index:03d}"
                for layer, count in (("lower", 9), ("lower_middle", 11),
                                     ("middle", 13), ("upper", 14))
                for index in range(count)]):
        raise ValueError("physical_manifest_four_layer_identity")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rock-assets", type=Path, required=True)
    parser.add_argument("--viewer-runtime", type=Path, required=True)
    parser.add_argument("--excavator-scene", type=Path, required=True)
    parser.add_argument("--operator-start", type=Path, required=True)
    parser.add_argument("--truck-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--headless-output", type=Path)
    args = parser.parse_args()

    arrays, metadata, identity = load_checkpoint_readonly(
        args.checkpoint, args.checkpoint_metadata, args.expected_sha256)
    records = _records(args.manifest.resolve())
    runtime = args.viewer_runtime.resolve()
    sys.path.insert(0, str(runtime))

    # The A30 SSH snapshot path has no X server.  Configure pyglet before
    # importing Newton so ViewerGL creates an EGL-backed invisible window
    # instead of attempting an Xlib display connection.
    import pyglet
    if args.headless_output is not None:
        pyglet.options["headless"] = True

    import newton
    import warp as wp
    from pyglet.window import key
    from scene_assembly import Assembler, matrix_quat
    from fast_kinematics import install
    from soil_surface import SoilSurface

    if args.headless_output is None:
        import pyglet.display.xlib as pyglet_xlib
        if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
            pyglet_xlib.XlibScreen.is_primary = False
    wp.init()
    wp.set_device(args.device)
    newton.use_coord_layout_targets = True

    scene = args.excavator_scene.resolve()
    assembler = Assembler(scene / "input/model/A.xml", scene / "input/mesh")
    install(assembler)
    initial = np.asarray(json.loads(args.operator_start.read_text(encoding="utf-8"))["qpos"])
    geometries, bodies = assembler.assemble(initial)

    def to_pose(transform):
        quaternion = matrix_quat(transform[:3, :3])
        return wp.transform(wp.vec3(*transform[:3, 3]),
                            wp.quat(*quaternion[1:], quaternion[0]))

    builder = newton.ModelBuilder()
    shape_config = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.45)
    for geometry in geometries:
        body = builder.add_body(
            xform=to_pose(bodies[geometry["body"]]), mass=0.0,
            is_kinematic=True, label=geometry["body"])
        mesh = newton.Mesh(
            geometry["body_local_vertices"].astype(np.float32),
            geometry["faces"].astype(np.int32).ravel(),
            compute_inertia=False, is_solid=False)
        builder.add_shape_mesh(
            body, mesh=mesh, cfg=shape_config,
            color=tuple(geometry["rgba"][:3]), label=geometry["mesh"])
    truck_triangles = 0
    for name, vertices, faces, color in _truck_parts(args.truck_root.resolve()):
        mesh = newton.Mesh(
            vertices.astype(np.float32), faces.astype(np.int32).ravel(),
            compute_inertia=False, is_solid=False)
        builder.add_shape_mesh(
            -1, mesh=mesh, cfg=shape_config, color=color, label="truck_" + name)
        truck_triangles += len(faces)
    if truck_triangles != 3298:
        raise ValueError(f"truck_triangle_count:{truck_triangles}")
    builder.add_ground_plane(color=(0.22, 0.24, 0.26))
    model = builder.finalize(device=args.device)
    model_state = model.state()

    viewer = newton.viewer.ViewerGL(
        width=1440, height=900, vsync=False, paused=True,
        headless=args.headless_output is not None)
    viewer.set_model(model)
    viewer.picking_enabled = False
    viewer.renderer.set_title("AUDITED 47-ROCK CHECKPOINT - READ ONLY - ZERO PHYSICS STEPS")
    viewer.set_camera(wp.vec3(9.0, -8.0, 5.5), pitch=-20.0, yaw=128.0)

    prototypes = sorted({item["prototype_id"] for item in records})
    for index, prototype in enumerate(prototypes):
        vertices, faces = _load_obj(args.rock_assets / f"{prototype}.obj")
        viewer.log_mesh(
            f"/checkpoint/prototype/{prototype}",
            wp.array(vertices, dtype=wp.vec3, device=args.device),
            wp.array(faces.reshape(-1), dtype=wp.int32, device=args.device),
            uvs=wp.array(_spherical_uv(vertices), dtype=wp.vec2, device=args.device),
            texture=_stone_texture(128, 4700 + index),
            color=(1.0, 1.0, 1.0), roughness=0.92, metallic=0.0)

    rock_groups = {}
    for prototype in prototypes:
        selected = [(record, pose) for record, pose in zip(
            records, arrays["rigid_qpos"], strict=True)
                    if record["prototype_id"] == prototype]
        rock_groups[prototype] = {
            "xforms": wp.array([
                wp.transform(
                    wp.vec3(*pose[:3]),
                    wp.quat(float(pose[4]), float(pose[5]),
                            float(pose[6]), float(pose[3])))
                for _, pose in selected], dtype=wp.transform, device=args.device),
            "scales": wp.array([
                wp.vec3(float(record["equivalent_diameter_m"]))
                for record, _ in selected], dtype=wp.vec3, device=args.device),
            "colors": wp.array([
                wp.vec3(*record["color_rgb"])
                for record, _ in selected], dtype=wp.vec3, device=args.device),
            "materials": wp.array([
                wp.vec4(0.92, 0.0, 0.0, 1.0)
                for _ in selected], dtype=wp.vec4, device=args.device),
        }

    particles = wp.array(arrays["particle_q"].astype(np.float32),
                         dtype=wp.vec3, device=args.device)
    surface = SoilSurface(particles, spacing=0.025)
    surface.update(particles, 0)
    surface_colors = wp.full(
        len(arrays["particle_q"]), wp.vec3(0.66, 0.46, 0.27),
        dtype=wp.vec3, device=args.device)

    def draw_scene():
        surface.draw(viewer, particles, surface_colors)
        for prototype, group in rock_groups.items():
            viewer.log_instances(
                f"/checkpoint/rocks/{prototype}",
                f"/checkpoint/prototype/{prototype}",
                group["xforms"], group["scales"], group["colors"],
                group["materials"])

    def close(symbol, _modifiers):
        if symbol in (key.X, key.ESCAPE):
            viewer.close()

    viewer.renderer.window.push_handlers(on_key_press=close)

    def ui(imgui):
        imgui.text("AUDITED 47-ROCK CHECKPOINT")
        imgui.text("Read only | zero physics steps | X/Esc: close")
        imgui.text(f"Particles: {EXPECTED_PARTICLE_COUNT} | rocks: {EXPECTED_ROCK_COUNT}")
        imgui.text(f"Simulation time in checkpoint: {float(arrays['simulation_time_s'][0]):.6f} s")
        imgui.text("Four layers: 9 + 11 + 13 + 14")
        imgui.text("Excavator/truck are display geometry; no contact advanced")
        imgui.text("Checkpoint SHA256: " + identity["sha256"][:16] + "...")

    viewer.register_ui_callback(ui, position="side")
    if args.headless_output is not None:
        for _ in range(4):
            viewer.begin_frame(0.0)
            viewer.log_state(model_state)
            draw_scene()
            viewer.end_frame()
        _write_ppm(args.headless_output.resolve(),
                   viewer.get_frame(render_ui=True).numpy())
        viewer.close()
    else:
        while viewer.is_running():
            viewer.begin_frame(0.0)
            viewer.log_state(model_state)
            draw_scene()
            viewer.end_frame()
            time.sleep(0.01)

    after = file_identity(args.checkpoint)
    if after != identity:
        raise RuntimeError("checkpoint_changed_during_view")
    print("VIEW_RESULT_BEGIN")
    print(json.dumps({
        "status": "audited_47_checkpoint_displayed_readonly",
        "checkpoint": identity["path"],
        "checkpoint_sha256": identity["sha256"],
        "checkpoint_unchanged": True,
        "particle_count": EXPECTED_PARTICLE_COUNT,
        "rock_count": EXPECTED_ROCK_COUNT,
        "physics_steps": 0,
        "device": args.device,
        "metadata_status": metadata["status"],
        "headless_output": (str(args.headless_output.resolve())
                            if args.headless_output is not None else None),
    }, indent=2, sort_keys=True))
    print("VIEW_RESULT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
