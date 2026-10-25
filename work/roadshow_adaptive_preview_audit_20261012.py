"""Saved-state kinematic review of roadshow A; zero physical steps."""
from pathlib import Path
from itertools import product
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/roadshow-measurements-20261011/geometry-independent-20261012-v4'
OUT = ROOT / 'runs/roadshow-measurements-20261011/geometry-adaptive-preview-audit-20261012-v2'
SOURCE = ROOT / 'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
CANDIDATE = ROOT / 'src/v4/roadshow_adaptive_lift_feedback.py'
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
    from roadshow_adaptive_lift_feedback import choose_preview_factor

    if OUT.exists():
        raise ValueError('immutable_output_exists')
    review = json.loads((RUN / 'offline-analysis/review.json').read_text())
    assert review['status'] == 'passed' and sha(SOURCE) == review['initial_state_sha256']
    actions_path = RUN / 'physics/actions.ndjson'
    assert sha(actions_path) == next(x['sha256'] for x in review['bindings'] if x['path'] == str(actions_path))
    frames = {120, 394, 478, 600, 750, 900, 901, 1050, 1200, 1350,
              1500, 1650, 1800, 1831, 1950, 2100, 2250, 2400}
    actions = {}
    for line in actions_path.open():
        row = json.loads(line)
        if row['frame'] in frames:
            actions[row['frame']] = row
    assert set(actions) == frames
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
            assert np.allclose(projected, requested, rtol=0, atol=1e-12)
            clearance = guard.measure(q)['clearance_m']
            latched = action['feedback']['no_down_latched']
            start_heights = heights(q) if latched else None
            calls = []

            def inspect(factor):
                t0 = time.perf_counter()
                candidate = projected * factor
                runtime.chain.q = q.copy()
                remaining = config.preview_seconds
                speed_scale = max(1., float(np.max(np.abs(candidate) / np.array([.12, .0225, .0225, .0225]))))
                minimum = clearance
                min_body_delta = 0.
                error = None
                try:
                    while remaining > 1e-12:
                        dt = min(1/60, remaining, .05/speed_scale)
                        predicted, _ = continuous_preview(runtime.projector, candidate/speed_scale,
                                                          min(.05, dt*speed_scale), runtime.chain.residual(q))
                        minimum = min(minimum, guard.measure(predicted)['clearance_m'])
                        if start_heights is not None:
                            min_body_delta = min(min_body_delta, float(np.min(heights(predicted) - start_heights)))
                        remaining -= dt
                except ValueError as exc:
                    error = str(exc)
                finally:
                    runtime.chain.q = q.copy()
                result = dict(factor=float(factor), passed=error is None and
                              minimum >= config.ground_clearance_m - 1e-6 and
                              min_body_delta >= -1e-6, error=error,
                              min_clearance_m=minimum, min_body_delta_m=min_body_delta,
                              elapsed_ms=1000 * (time.perf_counter() - t0))
                calls.append(result)
                return candidate, result

            accepted, safe, selected, search = choose_preview_factor(inspect, config.ground_clearance_m)
            assert calls == search and len(search) <= 6 and safe['passed']
            old_factor = action['feedback']['preview_scale']
            old_probe_count = (1., .5, .25, .125, 0.).index(old_factor) + 1
            assert selected >= old_factor and np.allclose(accepted, projected * selected, rtol=0, atol=1e-12)
            assert safe['min_clearance_m'] >= config.ground_clearance_m - 1e-6
            assert safe['min_body_delta_m'] >= -1e-6
            rows.append(dict(frame=frame, old_factor=old_factor, new_factor=selected,
                             no_down_latched=latched, attempts=search,
                             old_preview_count=old_probe_count,new_preview_count=len(search),
                             preview_count_delta=len(search)-old_probe_count,
                             candidate_preview_ms=sum(x['elapsed_ms'] for x in search)))
    assert all(x['new_factor'] > x['old_factor'] for x in rows if x['frame'] in (900, 901, 1831, 2400))
    report = dict(status='passed', scope='18_saved_decision_states_kinematic_only', physics_steps=0,
                  source_run=str(RUN), source_actions_sha256=sha(actions_path),
                  source_initial_sha256=sha(SOURCE), candidate_sha256=sha(CANDIDATE), rows=rows)
    (OUT / 'review.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', physics_steps=0, frames=len(rows),
                          improved=sum(x['new_factor'] > x['old_factor'] for x in rows),
                          key_frames=[(x['frame'], x['old_factor'], x['new_factor'])
                                      for x in rows if x['frame'] in (900, 901, 1831, 2400)])))


if __name__ == '__main__':
    main()
