"""Read-only same-method geometry diagnostic for original and D-128 finals."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, "/home/qunwang/codex-work/fill-volume-2026-09-15-v1")
from fill_volume import Calibration, bucket_volume, cavity_mask
from layered_excavation_approach_planner import _load_kinematics, quaternion_matrix_wxyz
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices

ROOT = Path("/home/qunwang/wajueji")
OUT = Path(__file__).resolve().parent
REPO = Path("/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo")
CAPACITY_LITERS = 255.164729950989

contract_path = Path("/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json")
contract = json.loads(contract_path.read_text())
front = np.load(contract_path.with_name("front-rim.npz"))["front"]
calpath = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1/calibration-known-geometry-v1/report.json")
calreport = json.loads(calpath.read_text())
assert calreport["pass"]
cal = Calibration.from_dict(calreport["profiles"]["reference-2p5cm"]["selected"]["calibration"])
kin = Path("/home/qunwang/codex-work/small-soil-20260910-v17-performance")
assembly, _ = _load_kinematics(kin, kin / "scene/input/model/A.xml", kin / "scene/input/mesh")
records = json.loads((REPO / "results/layered-mound-physical-preflight-v1/scene-manifest.json").read_text())["rocks"]
records = [r for r in records if r["layer_id"] in ("lower", "lower_middle", "middle", "upper")]
vertices = _cache_rock_vertices(REPO, records)
polygon = np.asarray(contract["bucket"]["polygon_xz_m"])
ylim = np.asarray(contract["bucket"]["y_limits_m"])


def rock_cloud(data, pose, spacing=0.0125):
    points, weights = [], []
    for record, vertices_local, q in zip(records, vertices, data["rigid_qpos"], strict=True):
        rotation = quaternion_matrix_wxyz(q[3:])
        world_vertices = vertices_local @ rotation.T + q[:3]
        bucket_center = (world_vertices.mean(axis=0) - pose[:3, 3]) @ pose[:3, :3]
        if np.linalg.norm(bucket_center - np.array([0.4, 0.0, -0.5])) > 1.6:
            continue
        hull = ConvexHull(vertices_local)
        axes = [np.arange(vertices_local[:, j].min() + spacing / 2, vertices_local[:, j].max(), spacing) for j in range(3)]
        samples = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        samples = samples[np.all(samples @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-10, axis=1)]
        if not len(samples):
            continue
        world = samples @ rotation.T + q[:3]
        points.append(world)
        weights.extend([record["volume_m3"] / len(samples)] * len(samples))
    return (np.concatenate(points) if points else np.empty((0, 3))), np.asarray(weights)


def measure(label, snapshot_path, state):
    data = np.load(snapshot_path)
    pose = assembly.assemble(np.asarray(state["mechanism_qpos"]), include_meshes=False)[1]["bucket"]
    assert np.max(np.abs(pose[:3, 3] - np.asarray(state["bucket_origin_world_m"]))) < 1e-6
    soil = data["particle_q"]
    local = (soil - pose[:3, 3]) @ pose[:3, :3]
    inside = cavity_mask(local, polygon, ylim, front)
    rocks, weights = rock_cloud(data, pose)
    soil_result = bucket_volume(soil, pose, contract, front, 0.025 ** 3, cal)
    mixed_result = bucket_volume(
        np.concatenate((soil, rocks)), pose, contract, front,
        np.r_[np.full(len(soil), 0.025 ** 3), weights], cal)
    return {
        "label": label,
        "snapshot": str(snapshot_path),
        "capacity_liters": CAPACITY_LITERS,
        "cavity_soil_particles": int(inside.sum()),
        "soil_geometry": soil_result,
        "mixed_geometry_rock_samples_0p0125m": mixed_result,
        "mixed_fill_percent": mixed_result["fill_percent"],
    }


original_rows = [json.loads(line) for line in (ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279/physics.ndjson").read_text().splitlines()]
candidate_rows = [json.loads(line) for line in (OUT / "trace.ndjson").read_text().splitlines()]
result = {
    "method": "Same fixed CAD cavity/front-rim, calibration and 1.25cm rock sampling as the prior manual fill diagnostic.",
    "original": measure("original", OUT.parent / "manual-soil-loss-replay-20260923-v1/diagnostic-001329.npz", original_rows[-1]["after_state"]),
    "candidate": measure("candidate", OUT / "candidate-001329.npz", candidate_rows[-1]["after_state"]),
    "scope": "Single-snapshot geometry diagnostic. It is not stable retained load, calibrated mixed-material mass, delivered truck volume, or full-scoop qualification.",
}
(OUT / "load-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
