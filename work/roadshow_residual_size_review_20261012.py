"""Quantify whether screened historical targets fit a small residual head."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/home/qunwang/wajueji')
SOURCE = ROOT/'runs/roadshow-measurements-20261011/historical-residual-screen-20261012-v1'
OUT = ROOT/'runs/roadshow-measurements-20261011/historical-residual-size-audit-20261012-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError('immutable_review_already_exists')
    source = json.loads((SOURCE/'review.json').read_text())
    if (source['status'] != 'offline_candidates_for_review_not_training_admitted'
            or sha(SOURCE/'samples.ndjson') != source['samples_ndjson_sha256']):
        raise ValueError('screen_source_identity')
    rows = [json.loads(line) for line in (SOURCE/'samples.ndjson').read_text().splitlines()]
    candidates = [row for row in rows if row['offline_screen_pass']]
    if len(rows) != 599 or len(candidates) != 130:
        raise ValueError('screen_counts')
    human = np.asarray([r['human_normalized_intent'] for r in candidates], float)
    geometry = np.asarray([r['geometry_normalized_same_state_intent'] for r in candidates], float)
    residual = np.asarray([r['human_minus_geometry_normalized_residual'] for r in candidates], float)
    if not np.allclose(human-geometry, residual, atol=1e-12, rtol=0):
        raise ValueError('residual_arithmetic')
    broad = np.max(np.abs(residual), axis=1) > 1
    early = np.asarray([207 <= r['decision_frame'] <= 300 for r in candidates])
    late = ~early
    result = {
        'schema': 'roadshow_historical_residual_size_audit_v1',
        'status': 'historical_manual_path_diverges_from_small_geometry_residual',
        'candidate_count': len(candidates),
        'independent_episodes': 1,
        'decision_frame_range': [candidates[0]['decision_frame'], candidates[-1]['decision_frame']],
        'mean_abs_residual_by_axis': np.mean(np.abs(residual), axis=0).tolist(),
        'maximum_abs_residual_by_axis': np.max(np.abs(residual), axis=0).tolist(),
        'any_axis_residual_abs_gt_1_count': int(broad.sum()),
        'early_207_through_300': {
            'count': int(early.sum()),
            'any_axis_residual_abs_gt_1_count': int(broad[early].sum()),
            'mean_abs_residual_by_axis': np.mean(np.abs(residual[early]), axis=0).tolist(),
        },
        'later_301_through_639': {
            'count': int(late.sum()),
            'any_axis_residual_abs_gt_1_count': int(broad[late].sum()),
        },
        'interpretation': 'The saved human path was operated without geometric feedback. A same-state geometry controller increasingly commands different boom and arm corrections. These targets do not constitute a small residual imitation dataset or demonstrate closed-loop improvement.',
        'training_eligible': False,
        'source_sha256': {str(p): sha(p) for p in (SOURCE/'review.json', SOURCE/'samples.ndjson', Path(__file__))},
    }
    if result['any_axis_residual_abs_gt_1_count'] != 100 or int(early.sum()) != 24:
        raise ValueError('unexpected_residual_stats')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'review.json').write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                        allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status','candidate_count',
        'any_axis_residual_abs_gt_1_count','early_207_through_300')}, indent=2))


if __name__ == '__main__':
    main()
