"""Audit two bounded resets and benchmark saved-state mixed fill, no new physics."""

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path('/home/qunwang/wajueji')
BATCH = ROOT/'runs/roadshow-measurements-20261011/rl-env-reset-gate-20261012-v1'
CACHE = ROOT/'runs/contact-preview-lift-20260924-v1/continuation'
FROZEN_FILL = ROOT/'work/contact-preview-lift-20260924-v1/fill-review.json'
CONFIG = ROOT/'configs/v4/live-bucket-fill-local-v1.json'
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')

sys.path[:0] = [str(ROOT), str(ROOT/'src/v4')]
from layered_excavation_approach_planner import _load_kinematics
from live_bucket_fill import MixedFillEstimator, load_config
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def without(row, keys):
    return {k: v for k, v in row.items() if k not in keys}


def main():
    out = BATCH/'offline-analysis'
    if out.exists():
        raise ValueError('immutable_review_exists')
    paths = [BATCH/f'repeat-{suffix}/physics' for suffix in ('a', 'b')]
    reports = [json.loads((p/'report.json').read_text()) for p in paths]
    checks = {
        'both_120_finished': all(r['status']=='bounded_probe_completed' and
            r['frames_completed']==120 for r in reports),
        'both_guards_and_gate_pass': all(r['all_executed_frames_safe'] and
            r['same_instance_gates']['120']['passed'] for r in reports),
        'same_unchanged_frozen_source': all(r['source_unchanged'] and
            r['source']['sha256']=='700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'
            for r in reports),
        'no_learned_policy': all(not r['policy_used_for_control'] for r in reports),
    }
    arrays = {}
    for name in ('gate-120.npz','final.npz'):
        with np.load(paths[0]/name) as first, np.load(paths[1]/name) as second:
            arrays[name] = {key: bool(np.array_equal(first[key], second[key]))
                            for key in first.files}
            checks[name+'_24_arrays_exact'] = (len(arrays[name]) == 24 and
                first.files == second.files and all(arrays[name].values()))
    logs = {
        'actions.ndjson': {'timing'},
        'phase-trace.ndjson': {'observation_seconds'},
        'terrain.ndjson': set(),
        'physics/frames.ndjson': {'assignment_wall_seconds',
            'mpm_step_components_wall_seconds','mpm_wall_seconds',
            'mujoco_wall_seconds','wall_seconds'},
    }
    log_counts = {}
    for name, ignore in logs.items():
        first, second = (read_lines(p/name) for p in paths)
        log_counts[name] = [len(first), len(second)]
        checks[name+'_nonclock_exact'] = (len(first)==len(second) and
            all(without(a,ignore)==without(b,ignore)
                for a,b in zip(first,second)))
    checks['full_source_still_sha_bound'] = sha(
        ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
    ) == reports[0]['source']['sha256']
    if not all(checks.values()):
        raise ValueError('repeat_reset_gate_failed:'+str(checks))

    scene = json.loads((CACHE/'display/scene.json').read_text())
    states = json.loads((CACHE/'display/states.json').read_text())
    particles = np.load(CACHE/'display/particles.npy', mmap_mode='r')
    config = load_config(CONFIG)
    records = scene['records']
    estimator = MixedFillEstimator(config,[r['volume_m3'] for r in records],
        _cache_rock_vertices(Path(scene['root']),records))
    assembly,_ = _load_kinematics(KIN,KIN/'scene/input/model/A.xml',
                                  KIN/'scene/input/mesh')
    frozen = json.loads(FROZEN_FILL.read_text())
    frozen_by_frame = {r['frame']: r for r in
        [*frozen['sampled_history'],frozen]}
    snapshots = []
    for local in (0,240,424):
        frame = 2400+local
        state = states[local]
        pose = assembly.assemble(state['mechanism_qpos'],
                                 include_meshes=False)[1]['bucket']
        snapshot = dict(frame=frame, pose=pose, soil=particles[local],
            rocks=[r['qpos'] for r in state['rocks']])
        measured = [estimator.measure_snapshot(snapshot) for _ in range(3)]
        first, reference = measured[0], frozen_by_frame[frame]
        same_result = all((r['valid_estimate'],r['fill_percent'],r['liters'],
            r['quality_reasons']) == (first['valid_estimate'],first['fill_percent'],
            first['liters'],first['quality_reasons']) for r in measured)
        matches_frozen = (first['valid_estimate']==reference['valid_estimate']
            and first['quality_reasons']==reference['quality_reasons']
            and (first['fill_percent'] is None and reference['fill_percent'] is None
                 or first['fill_percent'] is not None and abs(
                     first['fill_percent']-reference['fill_percent'])<1e-9))
        if not same_result or not matches_frozen:
            raise ValueError('fill_snapshot_mismatch:'+str(frame))
        times = [r['compute_wall_s'] for r in measured]
        snapshots.append(dict(frame=frame,valid_estimate=first['valid_estimate'],
            fill_percent=first['fill_percent'],quality_reasons=first['quality_reasons'],
            wall_seconds=times,median_wall_seconds=statistics.median(times),
            repeated_results_equal=same_result,matches_frozen_result=matches_frozen,
            display_only=True))
    if snapshots[0]['valid_estimate'] or snapshots[1]['valid_estimate'] or not snapshots[2]['valid_estimate']:
        raise ValueError('unexpected_volume_validity_sequence')
    physics_wall_per_frame = sum(r['wall_seconds'] for r in reports)/(2*120)
    estimator_median = statistics.median(
        t for snap in snapshots for t in snap['wall_seconds'])
    result = dict(schema='roadshow_rl_environment_small_gate_v1',status='fixed_start_repeat_and_saved_fill_probe_passed',
        scope='two independent 120-frame geometry runs, then offline measurement of historical third-scoop cached states',
        checks=checks,array_equal=arrays,log_counts=log_counts,
        runtime=dict(repeat_wall_seconds=[r['wall_seconds'] for r in reports],
            measured_wall_seconds_per_frame=physics_wall_per_frame,
            physics_steps_added_by_this_review=0),
        fill_probe=dict(snapshots=snapshots,all_snapshot_median_wall_seconds=estimator_median,
            serial_overhead_estimate_every_frame=estimator_median/physics_wall_per_frame,
            serial_overhead_estimate_every_60_frames=estimator_median/(60*physics_wall_per_frame),
            estimate_not_integrated_benchmark=True,calibration_status='user_selected_threshold_not_recalibrated',
            reward_qualified=False),
        limits=[
            'Only repeat-from-frozen-first-prepoint was checked; no in-process reset, arbitrary mid-scoop reset, or arbitrary residual step API was tested.',
            'The positive volume snapshot is a historical third-scoop continuation, not the current first-scoop geometry result.',
            'Bucket display estimate after off-soil is not the formal stable truck-delivery reward.',
            'Volume cost was measured offline on cached snapshots; integrated runtime overhead and CPU/GPU contention remain unmeasured.',
            'No residual policy, training reward, training episode, or RL improvement was evaluated.'
        ],training_environment_qualified=False,training_started=False,
        sha256={str(p):sha(p) for p in (
            *[q for run in paths for q in (run/'report.json',run/'actions.ndjson',
                run/'phase-trace.ndjson',run/'terrain.ndjson',run/'physics/frames.ndjson',
                run/'gate-120.npz',run/'final.npz')],
            CACHE/'display/scene.json',CACHE/'display/states.json',
            CACHE/'display/particles.npy',FROZEN_FILL,CONFIG,
            Path(config['measurement_source']),Path(config['contract']),
            Path(config['calibration']),Path(__file__))})
    out.mkdir(parents=True,exist_ok=False)
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,
        allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],checks_passed=sum(checks.values()),
        checks_total=len(checks),wall_seconds_per_frame=physics_wall_per_frame,
        fill_snapshots=snapshots,output=str(out/'review.json')),indent=2))


if __name__=='__main__':
    main()
