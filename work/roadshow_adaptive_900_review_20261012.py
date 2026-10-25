"""Audit bounded roadshow A v2 geometry run and software latency."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
BASE = ROOT / 'runs/roadshow-measurements-20261011'
RUN = BASE / 'geometry-adaptive-v2-900'
OLD = BASE / 'geometry-independent-20261012-v4'
V1 = BASE / 'geometry-adaptive-900'
SOURCE = ROOT / 'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
CANDIDATE = ROOT / 'src/v4/roadshow_adaptive_lift_feedback_v2.py'
RUNNER = ROOT / 'src/v4/roadshow_contact_comparison_adaptive_v2.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def quantile(items, fraction):
    return sorted(items)[math.ceil(fraction * len(items)) - 1]


def latency(directory):
    with (directory / 'metrics/latency.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))[:900]
    assert len(rows) == 900
    result = {}
    for label, start, end in [('warm_51_900', 51, 900), ('no_down_478_900', 478, 900)]:
        values = [float(row['e2e_ms']) for row in rows[start-1:end]]
        assert len(values) == end - start + 1 and all(math.isfinite(x) for x in values)
        result[label] = dict(count=len(values), p50_ms=statistics.median(values),
                             p95_ms=quantile(values, .95), max_ms=max(values),
                             over_100_ms=sum(value > 100 for value in values),
                             boundary='current_capture_to_software_command_send')
    return result


def main():
    out = RUN / 'offline-analysis'
    if out.exists():
        raise ValueError('immutable_output_exists')
    report = json.loads((RUN / 'physics/report.json').read_text())
    assert report['status'] == 'bounded_probe_completed' and report['frames_completed'] == 900
    assert report['all_executed_frames_safe'] and report['same_instance_gates']['120']['passed']
    assert report['source_unchanged'] and sha(SOURCE) == report['source']['sha256']
    assert not report['policy_used_for_control'] and not report['worker_ready']['neural_inference']
    binding = report['comparison_controller']
    assert binding['adaptive_preview_sha256'] == sha(CANDIDATE)
    assert binding['runner_sha256'] == sha(RUNNER)
    assert binding['source_sha256'] == sha(SOURCE)
    actions_path = RUN / 'physics/actions.ndjson'
    actions = [json.loads(line) for line in actions_path.open()]
    assert len(actions) == 900 and all(action['frame'] == i for i, action in enumerate(actions, 1))
    changed = []
    for action in actions:
        detail = action['feedback']
        search = detail['preview_search']
        assert detail['preview_search_version'] == 'roadshow_A_direct_proportional_probe_v2'
        assert 1 <= len(search) <= 6 and search[-1]['passed']
        assert search[-1]['factor'] == detail['preview_scale']
        assert search[-1]['min_body_delta_m'] >= -1e-6
        assert np.allclose(np.asarray(action['applied_command_si']),
                           np.asarray(action['requested_command_si']) * detail['preview_scale'],
                           rtol=0, atol=1e-12)
        if detail['preview_scale'] not in (1., .5, .25, .125, 0.):
            changed.append(action['frame'])
    assert actions[477]['feedback']['no_down_latched'] and changed
    before = [json.loads(line) for line in (OLD / 'physics/actions.ndjson').open()][:900]
    assert len(before) == 900
    with np.load(RUN / 'physics/gate-120.npz') as new, np.load(OLD / 'physics/gate-120.npz') as old:
        prefix_exact = {key: bool(np.array_equal(new[key], old[key])) for key in new.files}
    assert len(prefix_exact) == 24 and all(prefix_exact.values())
    results = dict(status='passed', scope='bounded_geometry_900_only',
                   source_sha256=sha(SOURCE), candidate_sha256=sha(CANDIDATE), runner_sha256=sha(RUNNER),
                   actions_sha256=sha(actions_path), final_sha256=sha(RUN / 'physics/final.npz'),
                   frames=900, all_executed_frames_safe=True, learned_weights_used=False,
                   prefix_120_arrays_exact=prefix_exact, no_down_first_frame=478,
                   fine_factor_frames=len(changed), first_fine_factor_frame=changed[0],
                   final_phase=report['final_phase'],
                   frame_900_closure_before=actions[-1]['feedback']['closure_fraction'],
                   old_frame_900_closure_before=before[-1]['feedback']['closure_fraction'],
                   frame_478_preview_attempts=len(actions[477]['feedback']['preview_search']),
                   latency=dict(adaptive_v2=latency(RUN), adaptive_v1=latency(V1), old_geometry=latency(OLD)),
                   limitation='900 frames is not off-soil or full-scoop proof; software send only')
    out.mkdir()
    (out / 'review.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', frames=900, fine_factor_frames=len(changed),
                          frame_900_closure_before=results['frame_900_closure_before'],
                          latency_post478=results['latency']['adaptive_v2']['no_down_478_900'])))


if __name__ == '__main__':
    main()
