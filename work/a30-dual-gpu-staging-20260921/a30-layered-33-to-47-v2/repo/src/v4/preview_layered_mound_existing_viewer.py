#!/usr/bin/env python3
"""Preview the confirmed layered mound in the existing excavator ViewerGL.

The entry reuses the already working excavator/truck/ground model and camera
stack.  It only swaps the visible soil/rock initialization.  No physics step
is executed and the preview makes no stability claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


SCHEMA = "excavation_layered_mound_viewer_preview_v1"
BACKEND = Path("/home/qunwang/codex-work/autonomous-multiscoop-speed-opt-2026-09-14-v1")
ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets/v4/mixed-material-r0"

SOIL_COLORS = {
    "foundation": (0.42, 0.30, 0.19),
    "lower_cover": (0.56, 0.39, 0.23),
    "lower_middle_cover": (0.61, 0.43, 0.25),
    "middle_cover": (0.68, 0.49, 0.29),
    "upper_cover": (0.73, 0.54, 0.33),
    "top_exposed": (0.78, 0.60, 0.37),
}
ROCK_BASE_COLORS = {
    "lower": (0.32, 0.30, 0.27),
    "lower_middle": (0.36, 0.34, 0.30),
    "middle": (0.40, 0.37, 0.32),
    "upper": (0.44, 0.40, 0.34),
    "top": (0.48, 0.43, 0.36),
}


def _load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != SCHEMA
            or value.get("status") != "geometry_only_existing_viewer_preview"):
        raise ValueError("layered_mound_preview_config")
    stages = value.get("construction_stages", [])
    if ([item.get("id") for item in stages] != [
            "foundation", "lower_cover", "lower_middle_cover",
            "middle_cover", "upper_cover", "top_exposed"]
            or [item.get("rock_count") for item in stages] != [0, 9, 11, 13, 14, 18]):
        raise ValueError("layered_mound_preview_stage_contract")
    scope = value.get("scope", {})
    if (scope != {
            "uses_existing_excavator_viewer": True,
            "geometry_only": True,
            "physics_step": False,
            "stability_claimed": False,
            "fullscale_qualification_claimed": False}):
        raise ValueError("layered_mound_preview_scope")
    return value


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
    radius = np.linalg.norm(centered, axis=1)
    radius = np.maximum(radius, 1e-9)
    u = 0.5 + np.arctan2(centered[:, 1], centered[:, 0]) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(np.clip(centered[:, 2] / radius, -1.0, 1.0)) / np.pi
    return np.column_stack((u, v)).astype(np.float32)


def _stone_texture(size: int, seed: int) -> np.ndarray:
    """Generate deterministic low-contrast rock grain without external assets."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    coarse = (0.070 * np.sin(xx * 0.105 + yy * 0.037 + seed)
              + 0.045 * np.sin(xx * 0.031 - yy * 0.121 + seed * 0.31)
              + 0.028 * np.cos(xx * 0.247 + yy * 0.193))
    fine = rng.normal(0.0, 0.028, size=(size, size))
    grain = np.clip(0.86 + coarse + fine, 0.62, 1.0)
    specks = rng.random((size, size)) < 0.018
    grain[specks] *= rng.uniform(0.55, 0.78, size=int(specks.sum()))
    rgb = np.stack((grain * 1.02, grain, grain * 0.94), axis=-1)
    return np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)


def _mound_mask(q: np.ndarray, config: dict) -> np.ndarray:
    envelope = config["mound_envelope"]
    outer = np.asarray(envelope["outer_xy_bounds_m"], dtype=float)
    plateau = np.asarray(envelope["plateau_xy_bounds_m"], dtype=float)
    x, y = q[:, 0], q[:, 1]
    distance = np.maximum.reduce([
        np.zeros(len(q)),
        (plateau[0, 0] - x) / (plateau[0, 0] - outer[0, 0]),
        (x - plateau[1, 0]) / (outer[1, 0] - plateau[1, 0]),
        (plateau[0, 1] - y) / (plateau[0, 1] - outer[0, 1]),
        (y - plateau[1, 1]) / (outer[1, 1] - plateau[1, 1]),
    ])
    normalized = np.clip(distance, 0.0, 1.0)
    base = float(envelope["base_height_m"])
    top = float(envelope["plateau_height_m"])
    height = base + (top - base) * 0.5 * (1.0 + np.cos(np.pi * normalized))
    return q[:, 2] <= height + 1e-9


def _euler_quaternion_and_matrix(roll: float, pitch: float, yaw: float):
    """Return xyzw quaternion and rotation matrix for intrinsic XYZ angles."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quaternion = np.asarray([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    x, y, z, w = quaternion
    matrix = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return quaternion, matrix


def _clustered_positions(layer: dict, bounds: np.ndarray,
                         diameters: list[float], rng: np.random.Generator) -> list[np.ndarray]:
    centers = np.asarray(layer["cluster_centers_uv"], dtype=float)
    jitter = np.asarray(layer["cluster_jitter_uv"], dtype=float)
    minimum_factor = float(layer["minimum_separation_factor"])
    span = bounds[1] - bounds[0]
    positions: list[np.ndarray] = []
    placed_diameters: list[float] = []
    cluster_order = np.arange(len(centers))
    rng.shuffle(cluster_order)
    for index, diameter in enumerate(diameters):
        center = centers[cluster_order[index % len(cluster_order)]]
        best = None
        best_clearance = -np.inf
        for attempt in range(180):
            uv = center + rng.normal(0.0, jitter)
            uv = np.clip(uv, [0.055, 0.070], [0.945, 0.930])
            xy = bounds[0] + uv * span
            if not positions:
                best = xy
                break
            clearances = [
                np.linalg.norm(xy - other)
                - minimum_factor * 0.5 * (diameter + other_diameter)
                for other, other_diameter in zip(positions, placed_diameters)]
            clearance = min(clearances)
            if clearance > best_clearance:
                best, best_clearance = xy, clearance
            if clearance >= 0.0:
                break
            if attempt in (59, 119):
                center = centers[rng.integers(0, len(centers))]
        if best is None:
            raise ValueError("layered_mound_preview_cluster_layout")
        positions.append(best)
        placed_diameters.append(diameter)
    return positions


def build_rock_records(config: dict) -> list[dict]:
    bounds = np.asarray(config["work_zone_xy_m"], dtype=float)
    records = []
    stage_by_layer = {
        item["rock_layer"]: index for index, item in enumerate(
            config["construction_stages"]) if item["rock_layer"] is not None}
    stage_specs = {item["rock_layer"]: item for item in config["construction_stages"]
                   if item["rock_layer"] is not None}
    for layer_name in stage_specs:
        layer = config["rock_layers"][layer_name]
        if layer.get("layout") != "clustered_jitter":
            raise ValueError("layered_mound_preview_layout:" + layer_name)
        count = int(layer["count"])
        rng = np.random.default_rng(int(layer["layout_seed"]))
        diameters = [float(layer["diameter_cycle_m"][index % len(layer["diameter_cycle_m"])])
                     for index in range(count)]
        prototypes = [layer["prototype_cycle"][index % len(layer["prototype_cycle"])]
                      for index in range(count)]
        order = rng.permutation(count)
        diameters = [diameters[index] for index in order]
        prototypes = [prototypes[index] for index in order]
        positions = _clustered_positions(layer, bounds, diameters, rng)
        spec = stage_specs[layer_name]
        for index, (xy, diameter, prototype) in enumerate(
                zip(positions, diameters, prototypes)):
            tilt = np.deg2rad(float(layer["tilt_limit_deg"]))
            roll, pitch = rng.uniform(-tilt, tilt, size=2)
            yaw = rng.uniform(-np.pi, np.pi)
            quaternion, rotation = _euler_quaternion_and_matrix(roll, pitch, yaw)
            vertices, _ = _load_obj(ASSET_ROOT / f"{prototype}.obj")
            rotated_z = (vertices.astype(float) * diameter @ rotation.T)[:, 2]
            local_min, local_max = float(rotated_z.min()), float(rotated_z.max())
            vertical_span = local_max - local_min
            surface = float(spec["placement_surface_height_m"])
            embed = np.clip(float(spec["embed_ratio"]) + rng.uniform(-0.055, 0.055),
                            0.15, 0.75)
            z = (surface - local_min - embed * vertical_span
                 + rng.uniform(-float(layer["height_jitter_m"]),
                               float(layer["height_jitter_m"])))
            brightness = rng.uniform(0.88, 1.08)
            records.append({
                "id": f"{layer_name}-{index:03d}",
                "layer": layer_name,
                "stage_index": stage_by_layer[layer_name],
                "prototype": prototype,
                "diameter_m": diameter,
                "position_m": [float(xy[0]), float(xy[1]), float(z)],
                "quaternion_xyzw": [float(value) for value in quaternion],
                "color_rgb": [float(np.clip(value * brightness, 0.0, 1.0))
                              for value in ROCK_BASE_COLORS[layer_name]],
            })
        if count != int(stage_specs[layer_name]["rock_count"]):
            raise ValueError("layered_mound_preview_rock_count:" + layer_name)
    return records


def _write_ppm(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        stream.write(np.ascontiguousarray(pixels).tobytes())


def _apply_physical_manifest(records: list[dict], config_path: Path,
                             manifest_path: Path) -> list[dict]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if (value.get("schema") != "excavation_layered_mound_physical_manifest_v1"
            or value.get("status") != "nonintersecting_geometry_admitted_for_time_zero"
            or value.get("visual_config_sha256") != config_digest
            or value.get("rock_count") != 65
            or not all(value.get("checks", {}).values())
            or value.get("physics_steps_executed") != 0
            or value.get("stable_scene_claimed") is not False):
        raise ValueError("layered_mound_physical_manifest_contract")
    baseline = {item["id"]: item for item in records}
    if {item.get("rock_id") for item in value.get("rocks", [])} != set(baseline):
        raise ValueError("layered_mound_physical_manifest_identity")
    result = []
    for item in value["rocks"]:
        source = baseline[item["rock_id"]]
        if (item["layer_id"] != source["layer"]
                or item["prototype_id"] != source["prototype"]
                or not np.isclose(item["equivalent_diameter_m"], source["diameter_m"],
                                  rtol=0.0, atol=1e-12)):
            raise ValueError("layered_mound_physical_manifest_rock_drift")
        w, x, y, z = item["orientation_quaternion_wxyz"]
        result.append({
            **source,
            "position_m": item["position_world_m"],
            "quaternion_xyzw": [x, y, z, w],
            "color_rgb": item["color_rgb"],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("foundation", "lower_cover", "lower_middle_cover",
                                             "middle_cover", "upper_cover", "top_exposed"),
                        default="top_exposed")
    parser.add_argument("--headless-output", type=Path)
    parser.add_argument("--rock-manifest", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    stages = config["construction_stages"]
    stage_index = next(index for index, item in enumerate(stages)
                       if item["id"] == args.stage)
    records = build_rock_records(config)
    if args.rock_manifest is not None:
        records = _apply_physical_manifest(
            records, args.config.resolve(), args.rock_manifest.resolve())

    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    import common  # noqa: F401  # Initializes the frozen backend import paths.
    import bank_runtime
    import newton
    import warp as wp
    import pyglet.display.xlib as pyglet_xlib
    from pyglet.window import key

    if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
        pyglet_xlib.XlibScreen.is_primary = False
    from soil_surface import SoilSurface
    wp.init()
    wp.set_device("cuda:0")
    model, _, _, _, _ = bank_runtime.build(
        BACKEND.parent / "small-soil-20260910-v17-performance" / "scene",
        BACKEND.parent / "manual-comparison-20260908-v1",
        particle_subdivision=2)
    model_state = model.state()
    with np.load(config["source_bank"]["path"], allow_pickle=False) as archive:
        source_q = archive["q"].astype(np.float32)
    if len(source_q) != int(config["source_bank"]["particle_count"]):
        raise ValueError("layered_mound_preview_particle_count")
    q = source_q[_mound_mask(source_q, config)]
    if len(q) != int(config["mound_envelope"]["expected_retained_particle_count"]):
        raise ValueError("layered_mound_preview_envelope_count")

    viewer_cfg = config["viewer"]
    viewer = newton.viewer.ViewerGL(
        width=int(viewer_cfg["width"]), height=int(viewer_cfg["height"]),
        vsync=False, paused=True, headless=args.headless_output is not None)
    viewer.set_model(model)
    viewer.picking_enabled = False
    viewer.renderer.set_title("Layered mound construction preview - existing excavator viewer")
    viewer.set_camera(wp.vec3(*viewer_cfg["camera_position_m"]),
                      pitch=float(viewer_cfg["camera_pitch_deg"]),
                      yaw=float(viewer_cfg["camera_yaw_deg"]))

    prototypes = sorted({item["prototype"] for item in records})
    rendering = config["rock_rendering"]
    texture_size = int(rendering["texture_size_px"])
    for prototype_index, prototype in enumerate(prototypes):
        vertices, faces = _load_obj(ASSET_ROOT / f"{prototype}.obj")
        viewer.log_mesh(f"/preview/prototype/{prototype}",
                        wp.array(vertices, dtype=wp.vec3, device="cuda:0"),
                        wp.array(faces.reshape(-1), dtype=wp.int32, device="cuda:0"),
                        uvs=wp.array(_spherical_uv(vertices), dtype=wp.vec2, device="cuda:0"),
                        texture=_stone_texture(texture_size, 9100 + prototype_index),
                        color=(1.0, 1.0, 1.0),
                        roughness=float(rendering["roughness"]),
                        metallic=float(rendering["metallic"]))

    selected_stage = {"value": stage_index}
    stage_names = [item["id"] for item in stages]

    def press(symbol, _modifiers):
        mapping = {key._1: 0, key._2: 1, key._3: 2,
                   key._4: 3, key._5: 4, key._6: 5}
        if symbol in mapping:
            selected_stage["value"] = mapping[symbol]
        elif symbol in (key.X, key.ESCAPE):
            viewer.close()

    viewer.renderer.window.push_handlers(on_key_press=press)

    def ui(imgui):
        index = selected_stage["value"]
        imgui.text("Existing excavator ViewerGL; geometry-only preview")
        imgui.text("1 foundation | 2-5 covered rock layers | 6 top exposed")
        imgui.text("Current: " + stage_names[index])
        imgui.text("Soil: smooth mound envelope + continuous surface renderer")
        imgui.text(f"Particles: {len(q)} retained from {len(source_q)} source")
        imgui.text("Rocks: 9 + 11 + 13 + 14 + 18 = 65; five clustered layers")
        imgui.text("Layout: " + ("SAT-clean physical preflight" if args.rock_manifest
                                  else "approved visual baseline"))
        imgui.text("Rock material: procedural grain, rough matte earth-tone variation")
        imgui.text("No physics step; no stability claim")

    viewer.register_ui_callback(ui, position="side")

    stride = int(viewer_cfg["particle_display_stride"])
    soil_points = {}
    soil_colors = {}
    for index, spec in enumerate(stages):
        selected = q[q[:, 2] <= float(spec["soil_height_m"]) + 1e-9][::stride]
        soil_points[index] = wp.array(selected, dtype=wp.vec3, device="cuda:0")
        soil_colors[index] = wp.full(len(selected), wp.vec3(*SOIL_COLORS[spec["id"]]),
                                     dtype=wp.vec3, device="cuda:0")
    surfaces = {}
    displayed_stage = {"value": None}

    def draw_scene(index: int):
        if index not in surfaces:
            surfaces[index] = SoilSurface(soil_points[index], spacing=0.025)
            surfaces[index].update(soil_points[index], index)
        if displayed_stage["value"] != index:
            surfaces[index].drawn_step = -2
            displayed_stage["value"] = index
        surfaces[index].draw(viewer, soil_points[index], soil_colors[index])
        for prototype in prototypes:
            selected = [item for item in records
                        if item["prototype"] == prototype and item["stage_index"] <= index]
            xforms = [wp.transform(wp.vec3(*item["position_m"]),
                                   wp.quat(*item["quaternion_xyzw"])) for item in selected]
            scales = [wp.vec3(item["diameter_m"]) for item in selected]
            colors = [wp.vec3(*item["color_rgb"]) for item in selected]
            materials = [wp.vec4(float(rendering["roughness"]),
                                 float(rendering["metallic"]), 0.0, 1.0)
                         for _ in selected]
            viewer.log_instances(
                f"/preview/rocks/{prototype}", f"/preview/prototype/{prototype}",
                wp.array(xforms, dtype=wp.transform, device="cuda:0") if xforms else None,
                wp.array(scales, dtype=wp.vec3, device="cuda:0") if scales else None,
                wp.array(colors, dtype=wp.vec3, device="cuda:0") if colors else None,
                wp.array(materials, dtype=wp.vec4, device="cuda:0") if materials else None,
                hidden=not selected)

    if args.headless_output is not None:
        for _ in range(4):
            viewer.begin_frame(0.0)
            viewer.log_state(model_state)
            draw_scene(selected_stage["value"])
            viewer.end_frame()
        _write_ppm(args.headless_output.resolve(), viewer.get_frame(render_ui=True).numpy())
        viewer.close()
        print(json.dumps({
            "result": "PASS", "stage": args.stage,
            "source_particle_count": len(source_q), "particle_count": len(q),
            "display_particle_count": len(soil_points[stage_index]),
            "rock_count": sum(item["stage_index"] <= stage_index for item in records),
            "physics_steps": 0,
            "rock_manifest": (str(args.rock_manifest.resolve())
                              if args.rock_manifest is not None else None),
            "output": str(args.headless_output.resolve())},
            sort_keys=True))
        return 0

    while viewer.is_running():
        viewer.begin_frame(0.0)
        viewer.log_state(model_state)
        draw_scene(selected_stage["value"])
        viewer.end_frame()
        time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
