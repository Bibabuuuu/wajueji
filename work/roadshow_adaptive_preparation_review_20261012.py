"""Review roadshow A zero/120 gates against the frozen geometry prefix."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path('/home/qunwang/wajueji/runs/roadshow-measurements-20261011')
SOURCE = Path('/home/qunwang/wajueji/runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz')
REFERENCE = ROOT / 'geometry-independent-20261012-v4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    zero = ROOT / 'geometry-adaptive-zero'
    zero_report = json.loads((zero / 'physics/report.json').read_text())
    assert zero_report['status'] == 'zero_step_passed' and zero_report['frames_completed'] == 0
    assert all(zero_report['zero_step_arrays_exact'].values()) and zero_report['source_unchanged']
    assert not zero_report['policy_used_for_control'] and not zero_report['worker_ready']['neural_inference']
    assert sha(SOURCE) == zero_report['source']['sha256']
    zero_review = dict(status='passed', physics_steps=0, initial_source_sha256=sha(SOURCE),
                       final_sha256=sha(zero / 'physics/final.npz'),
                       adaptive_preview_sha256=zero_report['comparison_controller']['adaptive_preview_sha256'])
    zero_review_path = zero / 'integration-review.json'
    if zero_review_path.exists():
        assert json.loads(zero_review_path.read_text()) == zero_review
    else:
        with zero_review_path.open('x') as stream:
            json.dump(zero_review, stream, indent=2)

    short = ROOT / 'geometry-adaptive-120'
    report = json.loads((short / 'physics/report.json').read_text())
    ref_report = json.loads((REFERENCE / 'physics/report.json').read_text())
    assert report['status'] == 'bounded_probe_completed' and report['frames_completed'] == 120
    assert report['all_executed_frames_safe'] and report['same_instance_gates']['120']['passed']
    assert not report['policy_used_for_control'] and not report['worker_ready']['neural_inference']
    assert report['runtime_bindings'] == ref_report['runtime_bindings']
    assert report['source_unchanged'] and sha(SOURCE) == report['source']['sha256']
    new = rows(short / 'physics/actions.ndjson')
    old = rows(REFERENCE / 'physics/actions.ndjson')[:120]
    assert len(new) == len(old) == 120
    fields = ('frame', 'before_state', 'requested_command_si', 'applied_command_si', 'phase')
    for a, b in zip(new, old):
        assert all(a[key] == b[key] for key in fields)
        assert {key: value for key, value in a['feedback'].items()
                if key not in ('preview_search', 'preview_search_version')} == b['feedback']
        assert len(a['feedback']['preview_search']) <= 6
        assert a['feedback']['preview_search_version'] == 'roadshow_A_one_fine_probe_v1'
    with np.load(short / 'physics/gate-120.npz') as a, np.load(REFERENCE / 'physics/gate-120.npz') as b:
        assert set(a.files) == set(b.files)
        exact = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
    assert len(exact) == 24 and all(exact.values())
    result = dict(status='passed', frames_completed=120,
                  initial_restoration='passed', all_executed_frames_safe=True,
                  shared_prefix_actual_controls_equal=True,
                  shared_prefix_fields=list(fields), feedback_common_fields_equal=True,
                  endpoint_arrays_exact=exact, endpoint_sha256=sha(short / 'physics/gate-120.npz'),
                  physics_runtime_bindings_equal=True, neural_inference=False,
                  adaptive_preview_sha256=report['comparison_controller']['adaptive_preview_sha256'],
                  source_unchanged=True, scope='zero_and_120_preparation_only')
    with (short / 'integration-review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], frames=120, physics_arrays=len(exact),
                          adaptive_preview_sha256=result['adaptive_preview_sha256'])))


if __name__ == '__main__':
    main()
