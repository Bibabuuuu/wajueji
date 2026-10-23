"""R1 open-bucket collision proxy and short rigid probe fixtures.

The source bucket mesh remains the visual truth.  This module builds a
separate, deterministic rigid-contact representation from the qualified
interior contract: a segmented curved shell, two thin side plates and five
teeth.  It deliberately does not modify or import the legacy 19/18/12 bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CONFIG_SCHEMA = "excavation_mixed_material_r1_config_v1"
PROXY_SCHEMA = "excavation_open_bucket_collision_proxy_v1"
REPORT_SCHEMA = "excavation_mixed_material_r1_proxy_report_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rdp_polyline(points, tolerance_m: float) -> np.ndarray:
    """Simplify an open polyline while retaining both endpoints."""
    values = np.asarray(points, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 2
            or not np.isfinite(values).all() or tolerance_m <= 0):
        raise ValueError("invalid_polyline")
    if len(values) == 2:
        return values.copy()
    chord = values[-1] - values[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        raise ValueError("degenerate_polyline_chord")
    offsets = values - values[0]
    distances = np.abs(
        chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
    ) / chord_length
    split = int(np.argmax(distances))
    if float(distances[split]) <= tolerance_m:
        return values[[0, -1]].copy()
    first = rdp_polyline(values[:split + 1], tolerance_m)
    second = rdp_polyline(values[split:], tolerance_m)
    return np.vstack((first[:-1], second))


def _box_along_xz(first, second, half_width_y: float,
                  half_thickness: float, overlap_m: float,
                  name: str) -> dict:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    delta = second - first
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("degenerate_box_segment")
    angle = math.atan2(-delta[1], delta[0])
    return {
        "name": name,
        "type": "box",
        "position_bucket_m": [
            float((first[0] + second[0]) / 2.0),
            0.0,
            float((first[1] + second[1]) / 2.0),
        ],
        "quaternion_wxyz": [
            math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0,
        ],
        "half_size_m": [
            length / 2.0 + overlap_m,
            half_width_y,
            half_thickness,
        ],
        "source_segment_xz_m": [first.tolist(), second.tolist()],
    }


def _convex_prism(polygon_xz, y_inner: float, y_outer: float,
                  name: str) -> dict:
    polygon = np.asarray(polygon_xz, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("invalid_prism_polygon")
    turns = []
    for index in range(len(polygon)):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        c = polygon[(index + 2) % len(polygon)]
        first = b - a
        second = c - b
        turns.append(float(first[0] * second[1] - first[1] * second[0]))
    if not (max(turns) < 1e-12 or min(turns) > -1e-12):
        raise ValueError("side_plate_polygon_not_convex")
    vertices = np.asarray(
        [[point[0], y, point[1]] for y in (y_inner, y_outer)
         for point in polygon], dtype=float)
    count = len(polygon)
    faces: list[list[int]] = []
    for index in range(1, count - 1):
        faces.append([0, index + 1, index])
        faces.append([count, count + index, count + index + 1])
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, following, count + following])
        faces.append([index, count + following, count + index])
    return {
        "name": name,
        "type": "convex_mesh",
        "vertices_bucket_m": vertices.tolist(),
        "faces": faces,
        "inner_y_m": y_inner,
        "outer_y_m": y_outer,
    }


def _point_box_clearance(point, box: dict) -> float:
    value = np.asarray(point, dtype=float)
    center = np.asarray(box["position_bucket_m"], dtype=float)
    w, _, y, _ = box["quaternion_wxyz"]
    angle = 2.0 * math.atan2(y, w)
    rotation = np.array([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    local = rotation.T @ (value - center)
    outside = np.maximum(np.abs(local) - np.asarray(box["half_size_m"]), 0.0)
    return float(np.linalg.norm(outside))


def build_proxy(inventory: dict, config: dict) -> dict:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected_r1_config_schema")
    bucket = inventory["bucket"]
    proxy_config = config["collision_proxy"]
    curve = rdp_polyline(
        bucket["polygon_xz_m"],
        float(proxy_config["shell_rdp_tolerance_m"]),
    )
    half_width = float(proxy_config["shell_outer_half_width_m"])
    half_thickness = float(proxy_config["shell_half_thickness_m"])
    overlap = float(proxy_config["shell_segment_overlap_m"])
    shell = [
        _box_along_xz(curve[index], curve[index + 1], half_width,
                      half_thickness, overlap, f"bucket_shell_{index:02d}")
        for index in range(len(curve) - 1)
    ]

    inner_negative, inner_positive = map(float, bucket["y_limits_m"])
    outer_negative, outer_positive = map(
        float, proxy_config["side_plate_outer_y_limits_m"])
    side_plates = [
        _convex_prism(curve, inner_negative, outer_negative,
                      "bucket_side_negative"),
        _convex_prism(curve, inner_positive, outer_positive,
                      "bucket_side_positive"),
    ]

    root = np.asarray(proxy_config["tooth_root_xz_m"], dtype=float)
    tip = np.asarray(proxy_config["tooth_tip_xz_m"], dtype=float)
    teeth = []
    for index, span in enumerate(proxy_config["tooth_tip_spans_y_m"]):
        low, high = map(float, span)
        tooth = _box_along_xz(
            root, tip, (high - low) / 2.0,
            float(proxy_config["tooth_half_thickness_m"]), overlap,
            f"bucket_tooth_{index:02d}",
        )
        tooth["position_bucket_m"][1] = (low + high) / 2.0
        tooth["tip_span_y_m"] = [low, high]
        teeth.append(tooth)

    mouth = np.asarray([curve[0], curve[-1]], dtype=float)
    chord = mouth[1] - mouth[0]
    chord_length = float(np.linalg.norm(chord))
    inward_xz = np.array([-chord[1], chord[0]]) / chord_length
    curve_center = curve.mean(axis=0)
    if float(np.dot(curve_center - mouth.mean(axis=0), inward_xz)) < 0:
        inward_xz *= -1
    mouth_center = np.array([mouth[:, 0].mean(), 0.0, mouth[:, 1].mean()])
    central_clearance = min(
        [_point_box_clearance(mouth_center, item) for item in shell + teeth]
        + [abs(inner_negative), abs(inner_positive)]
    )
    return {
        "schema": PROXY_SCHEMA,
        "coordinate_frame": "bucket_body",
        "source_bucket_sha256": config["bucket"]["source_sha256"],
        "source_inventory_sha256": config["bucket"]["inventory_contract_sha256"],
        "construction": "semantic_open_cavity_segmented_convex_proxy",
        "visual_mesh_policy": "retain_qualified_source_bucket_obj",
        "collision_mesh_policy": "do_not_use_whole_bucket_convex_hull",
        "shell_curve_original_point_count": len(bucket["polygon_xz_m"]),
        "shell_curve_simplified_xz_m": curve.tolist(),
        "shell_rdp_tolerance_m": proxy_config["shell_rdp_tolerance_m"],
        "mouth_endpoints_xz_m": mouth.tolist(),
        "mouth_chord_m": chord_length,
        "mouth_center_bucket_m": mouth_center.tolist(),
        "mouth_inward_direction_bucket": [
            float(inward_xz[0]), 0.0, float(inward_xz[1])],
        "central_mouth_point_proxy_clearance_m": central_clearance,
        "inner_clear_width_m": inner_positive - inner_negative,
        "shell_boxes": shell,
        "side_plate_meshes": side_plates,
        "tooth_boxes": teeth,
        "geom_count": len(shell) + len(side_plates) + len(teeth),
        "has_mouth_cap": False,
        "scope_limit": (
            "geometry candidate only until the generated MuJoCo probes and "
            "mechanism-attached rock entry/hold/dump tests pass"
        ),
    }


def write_obj(path: str | Path, mesh: dict) -> None:
    lines = ["# deterministic R1 convex side-plate proxy", "o " + mesh["name"]]
    lines += ["v " + " ".join(f"{value:.17g}" for value in vertex)
              for vertex in mesh["vertices_bucket_m"]]
    lines += ["f " + " ".join(str(index + 1) for index in face)
              for face in mesh["faces"]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _geom_xml(box: dict) -> str:
    pos = " ".join(f"{value:.17g}" for value in box["position_bucket_m"])
    quat = " ".join(f"{value:.17g}" for value in box["quaternion_wxyz"])
    size = " ".join(f"{value:.17g}" for value in box["half_size_m"])
    return (f'    <geom name="{box["name"]}" type="box" pos="{pos}" '
            f'quat="{quat}" size="{size}" rgba="0.2 0.6 0.9 0.25"/>')


def build_probe_xml(proxy: dict, config: dict, diameter_m: float,
                    meshdir_from_xml: str) -> str:
    probe = config["probe"]
    radius = diameter_m / 2.0
    density = float(probe["density_kg_m3"])
    mass = density * 4.0 * math.pi * radius ** 3 / 3.0
    center = np.asarray(proxy["mouth_center_bucket_m"], dtype=float)
    inward = np.asarray(proxy["mouth_inward_direction_bucket"], dtype=float)
    start = center - inward * float(probe["start_outside_m"])
    gravity = inward * 9.81
    timestep = float(probe["timestep_s"])
    lines = [
        '<mujoco model="mixed_material_r1_open_bucket_probe">',
        f'  <compiler angle="radian" meshdir="{meshdir_from_xml}"/>',
        (f'  <option timestep="{timestep:.17g}" gravity="'
         + " ".join(f"{value:.17g}" for value in gravity) +
         '" integrator="implicitfast" iterations="80"/>'),
        '  <default><geom contype="1" conaffinity="1" friction="0.8 0.02 0.002" '
        'solref="0.01 1" solimp="0.95 0.99 0.001"/></default>',
        '  <custom>',
        ('    <numeric name="mouth_center" data="' +
         " ".join(f"{value:.17g}" for value in center) + '"/>'),
        ('    <numeric name="mouth_inward" data="' +
         " ".join(f"{value:.17g}" for value in inward) + '"/>'),
        f'    <numeric name="probe_duration_s" data="{float(probe["duration_s"]):.17g}"/>',
        f'    <numeric name="probe_diameter_m" data="{diameter_m:.17g}"/>',
        '  </custom>',
        '  <asset>',
    ]
    for mesh in proxy["side_plate_meshes"]:
        lines.append(f'    <mesh name="{mesh["name"]}" file="{mesh["name"]}.obj"/>')
    lines += ['  </asset>', '  <worldbody>']
    lines += [_geom_xml(box) for box in proxy["shell_boxes"]]
    lines += [_geom_xml(box) for box in proxy["tooth_boxes"]]
    for mesh in proxy["side_plate_meshes"]:
        lines.append(f'    <geom name="{mesh["name"]}" type="mesh" '
                     f'mesh="{mesh["name"]}" rgba="0.2 0.6 0.9 0.25"/>')
    start_text = " ".join(f"{value:.17g}" for value in start)
    lines += [
        f'    <body name="rock" pos="{start_text}">',
        '      <freejoint name="rock_free"/>',
        (f'      <geom name="rock_geom" type="sphere" size="{radius:.17g}" '
         f'mass="{mass:.17g}" rgba="0.55 0.32 0.18 1"/>'),
        '    </body>',
        '  </worldbody>',
        '</mujoco>',
    ]
    return "\n".join(lines) + "\n"


def build_outputs(config_path: str | Path, inventory_path: str | Path,
                  bucket_obj: str | Path, asset_dir: str | Path,
                  report_dir: str | Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if sha256(bucket_obj) != config["bucket"]["source_sha256"]:
        raise ValueError("bucket_mesh_sha256_mismatch")
    if sha256(inventory_path) != config["bucket"]["inventory_contract_sha256"]:
        raise ValueError("inventory_contract_sha256_mismatch")
    assets = Path(asset_dir)
    reports = Path(report_dir)
    assets.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    proxy = build_proxy(inventory, config)
    for mesh in proxy["side_plate_meshes"]:
        write_obj(assets / f'{mesh["name"]}.obj', mesh)
        mesh["asset_path"] = f'assets/v4/mixed-material-r1/{mesh["name"]}.obj'
        mesh["sha256"] = sha256(assets / f'{mesh["name"]}.obj')
    write_json(reports / "proxy-manifest.json", proxy)

    fixture_records = []
    for diameter in config["probe"]["equivalent_diameters_m"]:
        name = f"open-bucket-probe-{float(diameter):.2f}m.xml"
        xml = build_probe_xml(proxy, config, float(diameter),
                              "../../../assets/v4/mixed-material-r1")
        (reports / name).write_text(xml, encoding="utf-8", newline="\n")
        fixture_records.append({
            "equivalent_diameter_m": float(diameter),
            "path": f"reports/v4/mixed-material-r1-2026-09-19-v1/{name}",
            "sha256": sha256(reports / name),
        })
    report = {
        "schema": REPORT_SCHEMA,
        "status": "geometry_candidate_generated_physics_probe_pending",
        "config_sha256": sha256(config_path),
        "proxy_manifest_sha256": sha256(reports / "proxy-manifest.json"),
        "source_bucket_sha256": config["bucket"]["source_sha256"],
        "source_inventory_sha256": config["bucket"]["inventory_contract_sha256"],
        "proxy_summary": {
            "geom_count": proxy["geom_count"],
            "shell_segment_count": len(proxy["shell_boxes"]),
            "side_plate_count": len(proxy["side_plate_meshes"]),
            "tooth_count": len(proxy["tooth_boxes"]),
            "has_mouth_cap": proxy["has_mouth_cap"],
            "central_mouth_point_proxy_clearance_m": (
                proxy["central_mouth_point_proxy_clearance_m"]),
        },
        "main_scene_distribution": config["main_scene_distribution"],
        "probe_fixtures": fixture_records,
        "admission": {
            "geometry_candidate": True,
            "standalone_mujoco_probe_passed": False,
            "rigid_bridge_mechanism_entry_passed": False,
            "rigid_bridge_mechanism_hold_passed": False,
            "rigid_bridge_mechanism_dump_passed": False,
            "mechanism_attached_entry_hold_dump_passed": False,
            "r1_complete": False,
        },
    }
    write_json(reports / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--bucket-obj", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    report = build_outputs(args.config, args.inventory, args.bucket_obj,
                           args.asset_dir, args.report_dir)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
