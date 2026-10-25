"""D136 paired geometry audit; same frozen D135 calibration and rock sampling."""
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
RUN = ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-235749-1040071"
INITIAL = ROOT / "runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz"
FINAL = RUN / "failure-001498.diagnostic.npz"
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



results = {}
for case, frame in [('recorded',1160), ('loaded',180)]:
    for mode in ['baseline','candidate']:
        directory = OUT / (case+'-'+mode)
        snapshot = directory / f'diagnostic-{frame:06d}.npz'
        if not snapshot.exists(): continue
        data = np.load(snapshot)
        pose = assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket']
        soil = data['particle_q']
        rocks, weights, details = rock_cloud(data,pose,.0125)
        geometry = bucket_volume(np.concatenate((soil,rocks)),pose,contract,front,
            np.r_[np.full(len(soil),.025**3),weights],calibration)
        results[case+'-'+mode] = {'snapshot':str(snapshot),'sha256':sha256(snapshot),
            'geometry':geometry,'rocks':details,'cavity_soil_count':int(cavity_mask(
                (soil-pose[:3,3])@pose[:3,:3],polygon,y_limits,front).sum()),
            'qualification':'geometric occupancy at short hold; not stable delivered volume'}
        print(case,mode,geometry['fill_percent'],flush=True)
(OUT/'geometry-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
