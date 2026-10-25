"""Validate and render the official-LiDAR-derived local envelope candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "excavation_official_lidar_envelope_candidate_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_and_validate(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA:
        raise ValueError("invalid_official_lidar_envelope_schema")
    if config.get("status") != "geometry_parameter_candidate_not_physical_scene":
        raise ValueError("invalid_candidate_status")
    source = config["source"]
    if source["source_status"] != "official_platform_output_human_reviewed":
        raise ValueError("untrusted_official_source_status")
    if source["camera_lidar_extrinsic_available"] is not False:
        raise ValueError("unexpected_camera_lidar_extrinsic_claim")
    roi = config["source_roi"]
    if not (roi["selected_component_cell_count"] <= roi["occupied_cell_count"]
            and roi["selected_component_support_point_count"] <= roi["raw_point_count"]):
        raise ValueError("invalid_component_support_counts")
    source_span = float(roi["x_m"][1]) - float(roi["x_m"][0])
    scaling = config["uniform_scaling"]
    expected_scale = float(scaling["target_reference_span_m"]) / source_span
    if not math.isclose(float(scaling["source_reference_span_m"]), source_span,
                        abs_tol=1e-12):
        raise ValueError("source_span_mismatch")
    if not math.isclose(float(scaling["scale"]), expected_scale, abs_tol=1e-12):
        raise ValueError("uniform_scale_mismatch")
    if scaling["per_object_rescaling_forbidden"] is not True:
        raise ValueError("per_object_rescaling_not_forbidden")
    target = config["target_envelope"]
    angle = float(target["slope_angle_deg"])
    gradient = math.tan(math.radians(angle))
    if not math.isclose(float(target["slope_dz_dx"]), gradient, abs_tol=1e-12):
        raise ValueError("slope_gradient_mismatch")
    rise = gradient * float(target["length_x_m"])
    if not math.isclose(float(target["nominal_plane_rise_m"]), rise, abs_tol=1e-12):
        raise ValueError("nominal_plane_rise_mismatch")
    source_residual = roi["reference_plane_residual_m"]
    target_residual = target["scaled_plane_residual_m"]
    for key in ("p05", "p50", "p95"):
        expected = float(source_residual[key]) * expected_scale
        if not math.isclose(float(target_residual[key]), expected, abs_tol=1e-12):
            raise ValueError("scaled_residual_mismatch:" + key)
    if not (target_residual["p05"] <= target_residual["p50"]
            <= target_residual["p95"]):
        raise ValueError("residual_quantiles_not_ordered")
    toe = float(target["toe_reference_height_m"])
    for key in ("p05", "p50", "p95"):
        if not math.isclose(float(target["toe_height_band_m"][key]),
                            toe + float(target_residual[key]), abs_tol=1e-12):
            raise ValueError("toe_height_band_mismatch:" + key)
        if not math.isclose(float(target["crest_height_band_m"][key]),
                            toe + rise + float(target_residual[key]), abs_tol=1e-12):
            raise ValueError("crest_height_band_mismatch:" + key)
    if target["toe_height_band_m"]["p05"] < 0.0:
        raise ValueError("target_reference_band_below_ground")
    if config["material_layout_intent"]["exact_rigid_rock_count"] is not None:
        raise ValueError("rigid_rock_count_prematurely_fixed")
    return config


def make_svg(config: dict) -> str:
    target = config["target_envelope"]
    length = float(target["length_x_m"])
    toe = float(target["toe_reference_height_m"])
    rise = float(target["nominal_plane_rise_m"])
    residual = target["scaled_plane_residual_m"]
    plot_left, plot_top, plot_width, plot_height = 100.0, 105.0, 1000.0, 540.0
    y_max = max(float(target["crest_height_band_m"]["p95"]) * 1.12, 1.7)

    def px(x: float) -> float:
        return plot_left + x / length * plot_width

    def py(z: float) -> float:
        return plot_top + plot_height - z / y_max * plot_height

    points = []
    lower = []
    upper = []
    median = []
    for index in range(101):
        x = length * index / 100.0
        plane = toe + rise * x / length
        lower.append((px(x), py(plane + float(residual["p05"]))))
        median.append((px(x), py(plane + float(residual["p50"]))))
        upper.append((px(x), py(plane + float(residual["p95"]))))
    points = lower + list(reversed(upper))
    polygon = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    median_path = " ".join(("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
                           for index, (x, y) in enumerate(median))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" viewBox="0 0 1280 760">
<rect width="100%" height="100%" fill="#f6f2e9"/>
<style>text{{font-family:DejaVu Sans,Arial,sans-serif;fill:#20252b}} .title{{font-size:27px;font-weight:700}} .head{{font-size:18px;font-weight:700}} .small{{font-size:14px}}</style>
<text x="70" y="45" class="title">Scaled official LiDAR A-envelope candidate</text>
<text x="70" y="75" class="small">2.2 m compact work patch; one uniform XYZ scale; not a physical scene</text>
<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#fff" stroke="#667085"/>
<polygon points="{polygon}" fill="#8ec9a3" opacity="0.55"/>
<path d="{median_path}" fill="none" stroke="#1f6f50" stroke-width="4"/>
<line x1="{plot_left}" y1="{py(0):.2f}" x2="{plot_left + plot_width}" y2="{py(0):.2f}" stroke="#333"/>
<text x="{plot_left + plot_width / 2}" y="700" class="head" text-anchor="middle">simulation X: toe → crest (m)</text>
<text x="38" y="{plot_top + plot_height / 2}" class="head" text-anchor="middle" transform="rotate(-90 38 {plot_top + plot_height / 2})">height (m)</text>
<text x="1130" y="135" class="small">slope: {float(target['slope_angle_deg']):.2f}°</text>
<text x="1130" y="165" class="small">plane rise: {rise:.3f} m</text>
<text x="1130" y="195" class="small">toe band: {float(target['toe_height_band_m']['p05']):.3f}…{float(target['toe_height_band_m']['p95']):.3f} m</text>
<text x="1130" y="225" class="small">crest band: {float(target['crest_height_band_m']['p05']):.3f}…{float(target['crest_height_band_m']['p95']):.3f} m</text>
<text x="70" y="735" class="small">Shaded band is scaled p05–p95 residual uncertainty, not an exact height field or pile extrema.</text>
</svg>
'''


def build(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_and_validate(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "preview.svg"
    preview_path.write_text(make_svg(config), encoding="utf-8")
    report = {
        "schema": "excavation_official_lidar_envelope_report_v1",
        "status": config["status"],
        "config_sha256": sha256(config_path),
        "preview_sha256": sha256(preview_path),
        "source_binding": config["source"],
        "target_envelope": config["target_envelope"],
        "material_layout_intent": config["material_layout_intent"],
        "physical_scene_modified": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    readme = (
        "# Official LiDAR A-envelope candidate\n\n"
        "Status: `geometry_parameter_candidate_not_physical_scene`.\n\n"
        "The official near-field A envelope is uniformly scaled to a 2.2 m compact "
        "work patch. The preview shows the reference slope and the scaled p05–p95 "
        "residual band. It does not reconstruct exact cells, infer material mechanics, "
        "modify R2, or authorize training.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.config, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
