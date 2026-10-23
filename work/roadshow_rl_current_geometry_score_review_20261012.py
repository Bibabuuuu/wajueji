"""Read-only, sparse mixed-fill validity check on saved first-scoop geometry."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
sys.path[:0] = [str(ROOT), str(ROOT/'src/v4')]
RUN = ROOT/'runs/roadshow-measurements-20261011/geometry-independent-20261012-v4/physics'
OUT = ROOT/'runs/roadshow-measurements-20261011/rl-current-geometry-score-gate-20261012-v1'
CONFIG = ROOT/'configs/v4/live-bucket-fill-local-v1.json'
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
FRAMES = (0, 900, 1800, 2400)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise ValueError('immutable_output_already_exists')
    from layered_excavation_approach_planner import _load_kinematics
    from live_bucket_fill import MixedFillEstimator, load_config
    from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices

    metadata=json.loads((RUN/'display/reconstruction.json').read_text())
    for name,expected in metadata['cache_sha256'].items():
        if sha(RUN/'display'/name)!=expected:
            raise ValueError('cache_sha_mismatch:'+name)
    scene=json.loads((RUN/'display/scene.json').read_text())
    states=json.loads((RUN/'display/states.json').read_text())
    particles=np.load(RUN/'display/particles.npy',mmap_mode='r')
    phase_rows={}
    with (RUN/'phase-trace.ndjson').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['frame'] in FRAMES:
                phase_rows[row['frame']]=row['phase']['phase_for_next_decision']
    assembly,_=_load_kinematics(KIN,KIN/'scene/input/model/A.xml',KIN/'scene/input/mesh')
    records=scene['records']
    estimator=MixedFillEstimator(load_config(CONFIG),
        [x['volume_m3'] for x in records],
        _cache_rock_vertices(Path(scene['root']),records))
    results=[]
    for frame in FRAMES:
        state=states[frame]
        pose=assembly.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
        measurement=estimator.measure_snapshot(dict(frame=frame,pose=pose,
            soil=particles[frame],rocks=[x['qpos'] for x in state['rocks']]))
        phase='approach_contact' if frame==0 else phase_rows[frame]
        results.append(dict(frame=frame,phase=phase,
            valid_estimate=measurement['valid_estimate'],
            fill_percent=measurement['fill_percent'],
            quality_reasons=measurement['quality_reasons'],
            compute_wall_s=measurement['compute_wall_s'],
            reward_eligible=phase=='off_soil' and measurement['valid_estimate'],
            display_only=measurement['display_only']))
    report=dict(schema='roadshow_rl_current_geometry_score_gate_v1',
        status='saved_first_scoop_sparse_score_checked',
        physics_steps_added=0,training_started=False,
        current_first_scoop_reward_qualified=any(x['reward_eligible'] for x in results),
        results=results,source_sha256={str(p):sha(p) for p in
            (RUN/'report.json',RUN/'phase-trace.ndjson',RUN/'display/reconstruction.json',
             CONFIG,Path(__file__))},
        cache_sha256=metadata['cache_sha256'],
        limitation='Four saved frames only; no claim about unsampled frames or stable truck delivery.')
    OUT.mkdir(parents=True)
    (OUT/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
