"""CPU-only working fill at an immutable first-scoop transport checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from live_bucket_fill import MixedFillEstimator, load_config, RESOURCE_KEYS
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    meta = json.loads(Path(str(source)+'.json').read_text())
    digest = sha(source)
    assert digest == meta['checkpoint_sha256']
    assert meta['cycle_index'] == 1 and meta['stage'] in ('transport', 'failure')
    assert meta['schema'] == 'layered_47_full_response_carry_transport_candidate_v2'
    scene_path = source.parent/'display/scene.json'
    scene = json.loads(scene_path.read_text())
    records = scene['records']
    vertices = _cache_rock_vertices(Path(scene['root']), records)
    config_path = ROOT/'configs/v4/live-bucket-fill-local-v1.json'
    config = load_config(config_path)
    estimator = MixedFillEstimator(config, [r['volume_m3'] for r in records], vertices)
    k = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
    assembly, _ = _load_kinematics(k, k/'scene/input/model/A.xml', k/'scene/input/mesh')
    with np.load(source, allow_pickle=False) as f:
        pose = assembly.assemble(f['mechanism_qpos'], include_meshes=False)[1]['bucket']
        result = estimator.measure_snapshot(dict(frame=0, pose=pose,
            soil=f['particle_q'], rocks=f['rigid_qpos']))
    result.pop('frame'); result.pop('sim_time_s')
    inputs = [source, Path(str(source)+'.json'), scene_path, config_path,
              Path(__file__), ROOT/'src/v4/live_bucket_fill.py',
              *[Path(config[key]) for key in RESOURCE_KEYS]]
    result.update(source=str(source), checkpoint_sha256=digest, physics_steps_added=0,
        formal_fill_qualification=False, stable_delivered_volume_qualification=False,
        scope='Single snapshot working cavity plus heap estimate; not spill attribution',
        inputs={str(p): sha(p) for p in inputs})
    assert sha(source) == digest
    with Path(args.output).open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
