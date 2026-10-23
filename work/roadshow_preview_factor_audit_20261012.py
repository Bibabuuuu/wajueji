"""Read-only kinematic replay of recorded geometry preview factors; no physics steps."""
from pathlib import Path
from itertools import product
import hashlib
import json
import sys

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/roadshow-measurements-20261011/geometry-independent-20261012-v4'
OUT = ROOT / 'runs/roadshow-measurements-20261011/geometry-preview-audit-20261012-v2'
SOURCE = ROOT / 'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
sys.path[:0] = [str(ROOT), str(ROOT / 'src/v4'),
                str(ROOT / 'work/contact-multisource-lift-20260924-v1'),
                str(ROOT / 'work/contact-preview-lift-20260924-v1'),
                str(ROOT / 'work/night-dual-start-preparation-20260925-v2')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from candidate_multisource import MultisourceRuntime
    from contact_bc_ground_guard import BucketGroundGuard
    from continuous_preview import preview as continuous_preview
    from contact_geometric_feedback import FeedbackConfig

    if OUT.exists():
        raise ValueError('immutable_output_exists')
    review = json.loads((RUN / 'offline-analysis/review.json').read_text())
    assert review['status'] == 'passed' and sha(SOURCE) == review['initial_state_sha256']
    actions_path = RUN / 'physics/actions.ndjson'
    assert sha(actions_path) == next(x['sha256'] for x in review['bindings'] if x['path'] == str(actions_path))
    indices = {900, 901, 1831, 2400}
    actions = {}
    for line in actions_path.open():
        row = json.loads(line)
        if row['frame'] in indices:
            actions[row['frame']] = row
    assert set(actions) == indices
    config = FeedbackConfig(**json.loads((ROOT / 'work/contact-multisource-lift-20260924-v1/first-protocol.json').read_text())['feedback_parameters'])
    OUT.mkdir()
    rows = []
    with MultisourceRuntime(SOURCE, OUT / 'runtime', allow_training_reset=False) as runtime:
        guard = BucketGroundGuard(runtime.assembly, runtime.bridge.state['mechanism_qpos'])
        corners = np.asarray(list(product(*zip(guard.vertices.min(axis=0), guard.vertices.max(axis=0)))))

        def heights(q):
            pose = runtime.assembly.assemble(q, include_meshes=False)[1]['bucket']
            return corners @ pose[2, :3] + pose[2, 3]

        for frame in sorted(actions):
            action = actions[frame]
            q = np.asarray(action['before_state']['mechanism_qpos'], float)
            requested = np.asarray(action['requested_command_si'], float)
            runtime.chain.q = q.copy()
            projected = runtime.projector.project(q, requested)
            projection_reason = runtime.projector.reason
            assert np.allclose(projected, requested, rtol=0, atol=1e-12)
            clearance = guard.measure(q)['clearance_m']
            start_heights = heights(q)
            factors = []
            for factor in (1., .5, .25, .125, 0., .40, .44, .45, .46):
                candidate = projected * factor
                runtime.chain.q = q.copy()
                remaining = config.preview_seconds
                speed_scale = max(1., float(np.max(np.abs(candidate) / np.array([.12, .0225, .0225, .0225]))))
                min_clearance = clearance
                min_body_delta = 0.
                error = None
                try:
                    while remaining > 1e-12:
                        dt = min(1/60, remaining, .05/speed_scale)
                        predicted, _ = continuous_preview(runtime.projector, candidate/speed_scale,
                                                          min(.05, dt*speed_scale), runtime.chain.residual(q))
                        min_clearance = min(min_clearance, guard.measure(predicted)['clearance_m'])
                        min_body_delta = min(min_body_delta, float(np.min(heights(predicted) - start_heights)))
                        remaining -= dt
                except ValueError as exc:
                    error = str(exc)
                finally:
                    runtime.chain.q = q.copy()
                factors.append(dict(factor=factor, min_clearance_m=min_clearance,
                                    min_body_delta_m=min_body_delta, error=error,
                                    clearance_ok=min_clearance >= config.ground_clearance_m - 1e-6,
                                    no_down_ok=min_body_delta >= -1e-6))
            expected = action['feedback']['preview_scale']
            selected = next((x['factor'] for x in factors[:5] if x['error'] is None and x['clearance_ok'] and x['no_down_ok']), None)
            assert selected == expected, (frame, selected, expected)
            assert np.allclose(projected * selected, action['applied_command_si'], rtol=0, atol=1e-12)
            rows.append(dict(frame=frame, logged_factor=expected, reproduced_factor=selected,
                             projection_reason=projection_reason, factors=factors,
                             fine_factor_44_safe=next(x for x in factors if x['factor']==.44)['error'] is None
                             and next(x for x in factors if x['factor']==.44)['clearance_ok']
                             and next(x for x in factors if x['factor']==.44)['no_down_ok']))
    report = dict(status='passed', scope='four_saved_decision_states_kinematic_preview_only_with_fine_factor_probe',
                  physics_steps=0, source_run=str(RUN), source_actions_sha256=sha(actions_path),
                  source_initial_sha256=sha(SOURCE), rows=rows)
    (OUT / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='passed', frames=sorted(indices), physics_steps=0,
                          rejected=[dict(frame=x['frame'], factors=[(v['factor'], v['error'], v['min_body_delta_m']) for v in x['factors']]) for x in rows])))


if __name__ == '__main__':
    main()
