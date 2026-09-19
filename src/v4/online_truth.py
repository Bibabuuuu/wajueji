"""Read-only soil-body separation and calibrated online fill truth.

The detector anchors the main bank to a fixed cohort from the initial bottom
layers, then labels the current particle topology.  Material travelling in the
bucket is therefore not treated as terrain merely because it still loads the
bucket.  The fill wrapper imports the already-qualified 2 cm measurement by an
explicit path and verifies its calibration and held-out reports before use.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.ndimage import generate_binary_structure, label


SEPARATION_SCHEMA = "soil_body_separation_truth_v1"
FILL_TRUTH_SCHEMA = "online_bucket_fill_truth_v1"
FILL_METHOD_VERSION = "bucket-local-density-v2-weighted"
FAST_PROFILE = "fast-5cm"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("invalid_" + name)
    return points


def _cavity_mask(points: np.ndarray, polygon_xz: np.ndarray,
                 y_limits: np.ndarray, front_rim_xz: np.ndarray) -> np.ndarray:
    q = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    front = np.asarray(front_rim_xz, dtype=np.float64)
    x = q[:, 0]
    z = q[:, 2]
    inside = np.zeros(len(q), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[1] - a[1]) > 1e-14:
            inside ^= ((a[1] > z) != (b[1] > z)) & (
                x < (b[0] - a[0]) * (z - a[1]) / (b[1] - a[1]) + a[0])
    order = np.argsort(front[:, 1])
    front_x = np.interp(z, front[order, 1], front[order, 0],
                        left=-10.0, right=-10.0)
    return (inside & (q[:, 1] >= y_limits[0]) & (q[:, 1] <= y_limits[1])
            & (x >= front_x))


def _distance_to_polygon_xz(points_xz: np.ndarray,
                            polygon_xz: np.ndarray) -> np.ndarray:
    if not len(points_xz):
        return np.empty(0, dtype=np.float64)
    best = np.full(len(points_xz), np.inf, dtype=np.float64)
    for a, b in zip(polygon_xz, np.roll(polygon_xz, -1, axis=0)):
        segment = b - a
        denominator = float(segment @ segment)
        if denominator <= 1e-18:
            distance = np.linalg.norm(points_xz - a, axis=1)
        else:
            along = np.clip(((points_xz - a) @ segment) / denominator, 0.0, 1.0)
            distance = np.linalg.norm(points_xz - (a + along[:, None] * segment), axis=1)
        best = np.minimum(best, distance)
    return best


class SoilBodySeparationDetector:
    """Classify current bucket/main-bank contact without advancing physics."""

    def __init__(self, initial_positions: np.ndarray, contract: dict[str, Any],
                 front_rim_xz: np.ndarray, *, particle_spacing_m: float,
                 support_depth_layers: float = 1.5,
                 contact_margin_factor: float = 1.25,
                 minimum_support_fraction: float = 0.5,
                 maximum_grid_cells: int = 8_000_000):
        initial = _finite_points(initial_positions, "initial_particle_positions")
        spacing = float(particle_spacing_m)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("invalid_particle_spacing")
        bucket = contract.get("bucket") if isinstance(contract, dict) else None
        if not isinstance(bucket, dict):
            raise ValueError("missing_bucket_contract")
        self.polygon = np.asarray(bucket.get("polygon_xz_m"), dtype=np.float64)
        self.y_limits = np.asarray(bucket.get("y_limits_m"), dtype=np.float64)
        self.front = np.asarray(front_rim_xz, dtype=np.float64)
        if (self.polygon.ndim != 2 or self.polygon.shape[1:] != (2,)
                or len(self.polygon) < 3 or not np.isfinite(self.polygon).all()
                or self.y_limits.shape != (2,) or not np.isfinite(self.y_limits).all()
                or self.y_limits[1] <= self.y_limits[0]
                or self.front.ndim != 2 or self.front.shape[1:] != (2,)
                or len(self.front) < 2 or not np.isfinite(self.front).all()):
            raise ValueError("invalid_bucket_geometry_contract")
        self.initial = initial.copy()
        self.count = len(initial)
        self.spacing = spacing
        self.cell_m = spacing
        self.contact_margin_m = contact_margin_factor * spacing
        self.minimum_support_fraction = float(minimum_support_fraction)
        self.maximum_grid_cells = int(maximum_grid_cells)
        bottom = float(initial[:, 2].min())
        self.support_mask = initial[:, 2] <= bottom + support_depth_layers * spacing + 1e-12
        self.support_count = int(self.support_mask.sum())
        if self.support_count < 4:
            raise ValueError("insufficient_fixed_support_cohort")

    def analyze(self, positions_world: np.ndarray, bucket_pose: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        q = _finite_points(positions_world, "particle_positions")
        pose = np.asarray(bucket_pose, dtype=np.float64)
        if len(q) != self.count:
            raise ValueError("particle_identity_count_changed")
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("invalid_bucket_pose")

        low = q.min(axis=0) - self.cell_m
        index = np.floor((q - low) / self.cell_m + 1e-9).astype(np.int64)
        shape = index.max(axis=0) + 2
        grid_cells = int(np.prod(shape, dtype=np.int64))
        if grid_cells <= 0 or grid_cells > self.maximum_grid_cells:
            raise ValueError("separation_grid_too_large")
        occupied = np.zeros(tuple(shape), dtype=bool)
        occupied[tuple(index.T)] = True
        components, component_count = label(
            occupied, structure=generate_binary_structure(3, 3))
        particle_components = components[tuple(index.T)]
        support_by_component = np.bincount(
            particle_components[self.support_mask], minlength=component_count + 1)
        if component_count == 0 or len(support_by_component) <= 1:
            raise ValueError("no_particle_components")
        main_component = int(np.argmax(support_by_component[1:]) + 1)
        support_fraction = float(support_by_component[main_component] / self.support_count)
        component_sizes = np.bincount(particle_components, minlength=component_count + 1)
        main_mask = particle_components == main_component

        local = (q - pose[:3, 3]) @ pose[:3, :3]
        cavity = _cavity_mask(local, self.polygon, self.y_limits, self.front)
        load_components = np.unique(particle_components[cavity])
        load_components = load_components[load_components != 0]
        main_connected_to_cavity = bool(np.any(load_components == main_component))

        margin = self.contact_margin_m
        main_local = local[main_mask]
        xz_low = self.polygon.min(axis=0) - margin
        xz_high = self.polygon.max(axis=0) + margin
        candidate = ((main_local[:, 0] >= xz_low[0])
                     & (main_local[:, 0] <= xz_high[0])
                     & (main_local[:, 2] >= xz_low[1])
                     & (main_local[:, 2] <= xz_high[1])
                     & (main_local[:, 1] >= self.y_limits[0] - margin)
                     & (main_local[:, 1] <= self.y_limits[1] + margin))
        candidate_xz = main_local[candidate][:, [0, 2]]
        boundary_distance = _distance_to_polygon_xz(candidate_xz, self.polygon)
        main_near_bucket = int(np.count_nonzero(boundary_distance <= margin + 1e-12))

        valid = support_fraction >= self.minimum_support_fraction
        terrain_contact = bool(main_connected_to_cavity or main_near_bucket > 0)
        if not valid:
            reason = "fixed support cohort no longer identifies one main soil body"
        elif terrain_contact:
            reason = "bucket or cavity-anchored load remains connected to main soil body"
        else:
            reason = "bucket and cavity-anchored load are separated from main soil body"
        load_component_particles = int(sum(
            int(component_sizes[value]) for value in load_components))
        return {
            "schema": SEPARATION_SCHEMA,
            "valid": valid,
            "terrain_contact": terrain_contact if valid else None,
            "separated_from_main_body": (not terrain_contact) if valid else None,
            "reason": reason,
            "particle_count": self.count,
            "component_count": int(component_count),
            "main_body_component": main_component,
            "main_body_particles": int(component_sizes[main_component]),
            "fixed_support_particles": self.support_count,
            "fixed_support_fraction_in_main_body": support_fraction,
            "cavity_particles": int(cavity.sum()),
            "cavity_component_count": int(len(load_components)),
            "cavity_component_particles": load_component_particles,
            "cavity_connected_to_main_body": main_connected_to_cavity,
            "main_body_particles_near_bucket_shell": main_near_bucket,
            "particle_spacing_m": self.spacing,
            "connectivity_cell_m": self.cell_m,
            "bucket_contact_margin_m": self.contact_margin_m,
            "identity_contract": "fixed particle row index; no resampling",
            "compute_wall_s": time.perf_counter() - started,
        }


class OnlineBucketFill:
    """Qualified read-only adapter around the existing 2 cm fill module."""

    def __init__(self, module: Any, calibration: Any, contract: dict[str, Any],
                 front_rim_xz: np.ndarray, source_hashes: dict[str, str]):
        self.module = module
        self.calibration = calibration
        self.contract = contract
        self.front = np.asarray(front_rim_xz, dtype=np.float64)
        self.source_hashes = dict(source_hashes)

    @classmethod
    def from_paths(cls, *, fill_module: str | Path,
                   calibration_report: str | Path,
                   held_out_report: str | Path,
                   contract: dict[str, Any], front_rim_xz: np.ndarray) -> "OnlineBucketFill":
        module_path = Path(fill_module).resolve()
        calibration_path = Path(calibration_report).resolve()
        held_out_path = Path(held_out_report).resolve()
        calibration_report_value = _read(calibration_path)
        held_out_value = _read(held_out_path)
        try:
            selected = calibration_report_value["profiles"][FAST_PROFILE]["selected"]
            held_profile = held_out_value["profiles"][FAST_PROFILE]
            calibration_value = selected["calibration"]
        except (KeyError, TypeError) as error:
            raise ValueError("missing_fast_fill_qualification") from error
        if not (calibration_report_value.get("schema") == "bucket-local-density-calibration-v1"
                and calibration_report_value.get("pass") is True
                and selected.get("profile") == FAST_PROFILE
                and selected.get("passes_three_point_contract") is True
                and held_out_value.get("schema") == "bucket-local-density-held-out-phase-v1"
                and held_out_value.get("pass") is True
                and held_profile.get("passes_three_point_contract") is True
                and held_profile.get("calibration") == calibration_value):
            raise ValueError("fast_fill_qualification_failed")
        spec = importlib.util.spec_from_file_location("qualified_fill_volume_runtime", module_path)
        if spec is None or spec.loader is None:
            raise ValueError("cannot_load_fill_module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if getattr(module, "METHOD_VERSION", None) != FILL_METHOD_VERSION:
            raise ValueError("fill_method_version_mismatch")
        calibration = module.Calibration.from_dict(calibration_value)
        if abs(float(calibration.particle_spacing_m) - 0.05) > 1e-12:
            raise ValueError("online_truth_requires_fast_5cm_profile")
        return cls(module, calibration, contract, front_rim_xz, {
            "fill_module_sha256": sha256(module_path),
            "calibration_report_sha256": sha256(calibration_path),
            "held_out_report_sha256": sha256(held_out_path),
        })

    def measure(self, positions_world: np.ndarray, bucket_pose: np.ndarray) -> dict[str, Any]:
        result = self.module.bucket_volume(
            positions_world, bucket_pose, self.contract, self.front,
            float(self.calibration.particle_spacing_m) ** 3, self.calibration)
        fill_percent = float(result["fill_percent"])
        valid = bool(result["valid_estimate"] and math.isfinite(fill_percent)
                     and fill_percent >= 0.0)
        return {
            "schema": FILL_TRUTH_SCHEMA,
            "valid": valid,
            "fill_fraction": fill_percent / 100.0 if valid else None,
            "method_version": result["method_version"],
            "profile": FAST_PROFILE,
            "qualification_scope": (
                "fast-tier training label; balanced/reference upgrade required near 95 percent"),
            "measurement": result,
            "source_hashes": dict(self.source_hashes),
        }
