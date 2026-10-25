"""Read-only post-hoc geometry and stability audit for the D-128 manual scoop."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, "/home/qunwang/codex-work/fill-volume-2026-09-15-v1")
from fill_volume import Calibration, bucket_volume, cavity_mask
from layered_excavation_approach_planner import (
    _load_kinematics, quaternion_matrix_wxyz)
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices


ROOT = Path("/home/qunwang/wajueji")
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-204445-912095"
INITIAL = ROOT / "runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz"
FINAL = RUN / "final-001212.diagnostic.npz"
REPO = Path("/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo")
CAPACITY_LITERS = 255.164729950989

contract_path = Path("/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json")
front_path = contract_path.with_name("front-rim.npz")
calibration_path = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1/calibration-known-geometry-v1/report.json")
fill_volume_path = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1/fill_volume.py")
contract = json.loads(contract_path.read_text())
front = np.load(front_path)["front"]
calibration_report = json.loads(calibration_path.read_text())
assert calibration_report["pass"]
calibration = Calibration.from_dict(
    calibration_report["profiles"]["reference-2p5cm"]["selected"]["calibration"])

kinematics = Path("/home/qunwang/codex-work/small-soil-20260910-v17-performance")
assembly, _ = _load_kinematics(
    kinematics, kinematics / "scene/input/model/A.xml",
    kinematics / "scene/input/mesh")
records = json.loads((REPO / "results/layered-mound-physical-preflight-v1/scene-manifest.json").read_text())["rocks"]
records = [r for r in records if r["layer_id"] in (
    "lower", "lower_middle", "middle", "upper")]
vertices = _cache_rock_vertices(REPO, records)
polygon = np.asarray(contract["bucket"]["polygon_xz_m"])
y_limits = np.asarray(contract["bucket"]["y_limits_m"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rock_cloud(data, pose, spacing):
    points, weights, details = [], [], []
    for index, (record, local_vertices, qpos) in enumerate(zip(
            records, vertices, data["rigid_qpos"], strict=True)):
        rotation = quaternion_matrix_wxyz(qpos[3:])
        world_vertices = local_vertices @ rotation.T + qpos[:3]
        local_center = (
            world_vertices.mean(axis=0) - pose[:3, 3]) @ pose[:3, :3]
        if np.linalg.norm(local_center - np.array([0.4, 0.0, -0.5])) > 1.6:
            continue
        hull = ConvexHull(local_vertices)
        axes = [np.arange(
            local_vertices[:, axis].min() + spacing / 2,
            local_vertices[:, axis].max(), spacing) for axis in range(3)]
        samples = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        samples = samples[np.all(
            samples @ hull.equations[:, :3].T + hull.equations[:, 3]
            <= 1.0e-10, axis=1)]
        if not len(samples):
            continue
        world = samples @ rotation.T + qpos[:3]
        points.append(world)
        weights.extend([record["volume_m3"] / len(samples)] * len(samples))
        inside = cavity_mask(
            (world - pose[:3, 3]) @ pose[:3, :3],
            polygon, y_limits, front)
        details.append({
            "rock_id": record["rock_id"],
            "index": index,
            "volume_liters": record["volume_m3"] * 1000.0,
            "fraction_in_cavity": float(inside.mean()),
            "center_bucket_m": local_center.tolist(),
        })
    cloud = np.concatenate(points) if points else np.empty((0, 3))
    return cloud, np.asarray(weights), details


report = json.loads((RUN / "report.json").read_text())
rows = [json.loads(line) for line in (RUN / "physics.ndjson").read_text().splitlines()]
initial = np.load(INITIAL)
final = np.load(FINAL)
assert len(rows) == 1212
assert report["status"] == "manual_contact_pilot_recorded"
assert all(report["checks"].values())
assert all(row["hard_guard"] for row in rows)

pose = assembly.assemble(
    final["mechanism_qpos"], include_meshes=False)[1]["bucket"]
soil = final["particle_q"]
soil_local = (soil - pose[:3, 3]) @ pose[:3, :3]
cavity_soil = cavity_mask(soil_local, polygon, y_limits, front)
rock_points, rock_weights, rock_details = rock_cloud(final, pose, 0.0125)
soil_geometry = bucket_volume(
    soil, pose, contract, front, 0.025 ** 3, calibration)
mixed_geometry = bucket_volume(
    np.concatenate((soil, rock_points)), pose, contract, front,
    np.r_[np.full(len(soil), 0.025 ** 3), rock_weights], calibration)

bucket_velocity = np.asarray(rows[-1]["after_state"][
    "bucket_linear_velocity_world_m_s"])
bucket_omega = np.asarray(rows[-1]["after_state"][
    "bucket_angular_velocity_world_rad_s"])
relative_speed = np.linalg.norm(
    final["particle_qd"] - bucket_velocity
    - np.cross(bucket_omega, soil - pose[:3, 3]), axis=1)

initial_q = initial["particle_q"]
displacement = np.linalg.norm(soil - initial_q, axis=1)
initial_bottom = initial_q[:, 2] < 0.2
expanded_footprint = (
    (soil[:, 0] >= initial_q[:, 0].min() - 0.1)
    & (soil[:, 0] <= initial_q[:, 0].max() + 0.1)
    & (soil[:, 1] >= initial_q[:, 1].min() - 0.1)
    & (soil[:, 1] <= initial_q[:, 1].max() + 0.1))

materials = [row["hencky_material_guard"] for row in rows]
phase_events = report["phase_events"]
payload_initial = initial["particle_q"][cavity_soil]
payload_final = final["particle_q"][cavity_soil]
payload_displacement = payload_final - payload_initial
initial_bucket_origin = np.asarray(rows[0]["before_state"][
    "bucket_origin_world_m"])
final_bucket_origin = np.asarray(rows[-1]["after_state"][
    "bucket_origin_world_m"])
result = {
    "schema": "hencky_manual_single_scoop_posthoc_v1",
    "status": "posthoc_geometry_and_stability_audit_completed",
    "run": str(RUN),
    "snapshot": str(FINAL),
    "snapshot_sha256": sha256(FINAL),
    "frames": len(rows),
    "simulation_seconds": len(rows) / 60.0,
    "terminal_zero_hold_frames": 30,
    "capacity_liters": CAPACITY_LITERS,
    "load_geometry": {
        "soil_only": soil_geometry,
        "mixed_soil_and_rock_0p0125m_samples": mixed_geometry,
        "reported_mixed_fill_percent": mixed_geometry["fill_percent"],
        "cavity_soil_particles": int(cavity_soil.sum()),
        "cavity_soil_relative_speed_quantiles_m_s": np.quantile(
            relative_speed[cavity_soil], [0.0, 0.5, 0.9, 1.0]).tolist(),
        "rock_candidates": rock_details,
    },
    "soil_stability": {
        "all_final_positions_finite": bool(np.isfinite(soil).all()),
        "final_minimum_z_m": float(soil[:, 2].min()),
        "final_particles_below_minus_5mm": int(np.count_nonzero(
            soil[:, 2] < -0.005)),
        "initial_bottom_particle_count": int(initial_bottom.sum()),
        "initial_bottom_moved_over_0p2m": int(np.count_nonzero(
            initial_bottom & (displacement > 0.2))),
        "initial_bottom_maximum_displacement_m": float(
            displacement[initial_bottom].max()),
        "initial_bottom_outside_expanded_footprint": int(np.count_nonzero(
            initial_bottom & ~expanded_footprint)),
    },
    "material_guard": {
        "all_frames_safe": all(m["safe"] for m in materials),
        "maximum_rejection_count": max(m["rejection_count"] for m in materials),
        "total_nonfinite_values": sum(
            sum(m["nonfinite_values"].values()) for m in materials),
        "maximum_F_frobenius": max(m["maximum_F_frobenius"] for m in materials),
        "minimum_Jp": min(m["minimum_Jp"] for m in materials),
        "maximum_Jp": max(m["maximum_Jp"] for m in materials),
    },
    "effective_payload": {
        "bucket_origin_displacement_m": (
            final_bucket_origin - initial_bucket_origin).tolist(),
        "cavity_soil_particle_count": int(cavity_soil.sum()),
        "cavity_soil_vertical_displacement_quantiles_m": np.quantile(
            payload_displacement[:, 2], [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        "cavity_soil_total_displacement_quantiles_m": np.quantile(
            np.linalg.norm(payload_displacement, axis=1),
            [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        "cavity_soil_final_z_quantiles_m": np.quantile(
            payload_final[:, 2], [0.0, 0.1, 0.5, 0.9, 1.0]).tolist(),
        "fully_contained_rock_count": sum(
            detail["fraction_in_cavity"] >= 0.99 for detail in rock_details),
        "partially_contained_rock_count": sum(
            0.01 < detail["fraction_in_cavity"] < 0.99
            for detail in rock_details),
        "terminal_hold_relative_speed_max_m_s": float(
            relative_speed[cavity_soil].max()),
        "assessment": (
            "payload_lifted_from_initial_bank_and_retained_through_30_frame_hold"),
    },
    "phase_annotations": {
        "operator_markers": phase_events,
        "operator_reported_timing_inaccurate": True,
        "used_for_fill_measurement": False,
        "posthoc_reconstruction_required_before_training_review": True,
    },
    "scope": (
        "Final-snapshot fixed-CAD geometric diagnostic after a 30-frame zero "
        "hold. Mixed rock sampling is geometric, not a calibrated mixed mass. "
        "This does not qualify truck delivery or training data."),
    "source_bindings": {
        str(path): sha256(path) for path in (
            FINAL, RUN / "report.json", RUN / "physics.ndjson", INITIAL,
            contract_path, front_path, calibration_path, fill_volume_path,
            Path(__file__))
    },
}
(OUT / "report.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, indent=2, ensure_ascii=False))
