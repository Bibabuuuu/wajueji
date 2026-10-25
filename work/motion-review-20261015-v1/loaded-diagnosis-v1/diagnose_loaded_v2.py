"""Saved-log diagnosis only: kinematics and NumPy, no physical runtime import."""
import ast
import csv
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path('/home/qunwang/wajueji')
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
RUN = ROOT / 'runs/motion-physical-short-20261015-v2'
OLD = ROOT / 'runs/edge-first-two-scoop-20261015-continuation'
AXES = np.array([7, 9, 12, 15])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def angle_delta(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def summarize(values):
    a = np.asarray(values, float)
    return dict(minimum=float(a.min()), maximum=float(a.max()),
                rms=float(np.sqrt(np.mean(a * a))), start=float(a[0]), end=float(a[-1]))


def main():
    started = time.monotonic()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    inputs = [RUN / 'report.json', RUN / 'read-only-audit.json',
              RUN / 'loaded/actions.ndjson', RUN / 'loaded/dispatch.ndjson',
              RUN / 'loaded/physics/frames.ndjson', RUN / 'loaded/display/states.json',
              RUN / 'loaded/executed-reference.npz', OLD / 'geometry-actions.ndjson',
              ROOT / 'src/v4/bucket_integral_candidate.h',
              ROOT / 'work/contact-carry-transport-20260924-v2/load_tracking.py',
              ROOT / 'work/contact-second-finish-unload-20260925-v4/transport_feedback.py',
              ROOT / 'work/contact-second-fixed-route-20260925-v1/plan.py',
              ROOT / 'planning/cycle_geometry.py',
              RUN / 'loaded/physics/integral/capacity-bridge.cc',
              ROOT / 'work/motion-physical-short-20261015-v2/run.py']
    before_hashes = {str(p): sha(p) for p in inputs}
    sys.path[:0] = [str(KIN), str(ROOT)]
    assembly = importlib.import_module('scene_assembly').Assembler(
        KIN / 'scene/input/model/A.xml', KIN / 'scene/input/mesh')
    importlib.import_module('fast_kinematics').install(assembly)
    chain = importlib.import_module('closed_chain').ClosedChain(assembly)
    from planning.cycle_geometry import tool_pose, MOUTH
    # Reuse exactly the existing pure function without importing plan.py's
    # transport runtime dependencies or any native simulation module.
    plan = inputs[11]
    tree = ast.parse(plan.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'project_passive')
    scope = dict(np=np, least_squares=least_squares)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(plan), 'exec'), scope)
    project = lambda q: scope['project_passive'](chain, np.array(q, float, copy=True))
    def project_axes(seed, target_axes):
        # Large axis substitutions can exceed the original helper's 0.05rad
        # passive-branch step bound. Continue in 16 small mathematical steps;
        # every call still enforces the unchanged residual and branch bounds.
        q = seed.copy()
        origin = q[AXES].copy()
        for fraction in np.arange(1, 17) / 16:
            q[AXES] = origin + fraction * (target_axes - origin)
            q = project(q)
        return q
    pitch = lambda q: tool_pose(assembly, np.asarray(q, float), MOUTH)[1]
    states = read(RUN / 'loaded/display/states.json')
    actions = list(rows(RUN / 'loaded/actions.ndjson'))
    physics = list(rows(RUN / 'loaded/physics/frames.ndjson'))
    dispatch = list(rows(RUN / 'loaded/dispatch.ndjson'))
    assert len(states) == len(physics) + 1 == 888
    assert [x['frame'] for x in actions] == list(range(4, 888))
    assert all(x['before_state'] == states[x['frame'] - 1] for x in actions)
    assert all(x['after_state'] == states[x['frame']] for x in physics)
    assert all(x['applied_command_si'] == d['command'] for x, d in zip(physics, dispatch, strict=True))
    with np.load(RUN / 'loaded/executed-reference.npz') as archive:
        for phase in ('CLEAR_PILE', 'ALIGN'):
            selected = [x for x in actions if x['phase'] == phase]
            q = archive[phase + '_qpos']
            v = archive[phase + '_velocity']
            for i, x in enumerate(selected):
                assert np.array_equal(x['desired'], q[min(i + 1, len(q) - 1)])
                assert np.array_equal(x['velocity'], v[i] if i < len(v) else np.zeros(4))
    flat = []
    for x in actions:
        actual = np.array(x['before_state']['mechanism_qpos'])
        desired = np.array(x['desired'])
        actual_pitch = pitch(actual)
        assert abs(angle_delta(actual_pitch, x['detail']['pitch_diagnostic_rad'])) < 1e-12
        exact = project(actual)
        residual = float(np.max(abs(chain.residual(actual))))
        row = dict(frame=x['frame'], phase=x['phase'], simulation_time_s=x['before_state']['time_s'],
                   actual_pitch_deg=float(np.rad2deg(actual_pitch)),
                   reference_pitch_deg=float(np.rad2deg(pitch(desired))),
                   pitch_error_deg=float(np.rad2deg(angle_delta(actual_pitch, pitch(desired)))),
                   passive_projection_pitch_delta_deg=float(np.rad2deg(angle_delta(actual_pitch, pitch(exact)))),
                   closure_residual_m=residual,
                   bucket_integral_N=x['before_state']['bucket_integral_effort'][3])
        for i, name in enumerate(('swing', 'boom', 'arm', 'bucket')):
            target = x['before_state']['controller_target'][i]
            row.update({name + '_reference_error_si': float(actual[AXES[i]] - desired[AXES[i]]),
                        name + '_actual_minus_native_target_si': float(actual[AXES[i]] - target),
                        name + '_command_si': x['command'][i],
                        name + '_load_bias_si': x['detail']['load_bias'][i]})
        flat.append(row)
    with (out / 'frame-alignment.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    metrics = {}
    peaks = []
    for phase in ('CLEAR_PILE', 'ALIGN'):
        group = [x for x in flat if x['phase'] == phase]
        errors = np.array([x['pitch_error_deg'] for x in group])
        event = max(group, key=lambda x: abs(x['pitch_error_deg']))
        x = next(a for a in actions if a['frame'] == event['frame'])
        actual = np.array(x['before_state']['mechanism_qpos'])
        desired = np.array(x['desired'])
        # Exact 4-axis Shapley partition; order-independent kinematic
        # attribution, not a dynamical intervention or force causal proof.
        combinations = {}
        for mask in range(16):
            q = desired.copy()
            for i in range(4):
                if mask & (1 << i):
                    q[AXES[i]] = actual[AXES[i]]
            combinations[mask] = pitch(project_axes(desired, q[AXES]))
        contributions = []
        for i in range(4):
            value = 0.
            for mask in range(16):
                if mask & (1 << i):
                    continue
                k = mask.bit_count()
                weight = math.factorial(k) * math.factorial(3 - k) / math.factorial(4)
                value += weight * angle_delta(combinations[mask | (1 << i)], combinations[mask])
            contributions.append(float(np.rad2deg(value)))
        passive = float(np.rad2deg(angle_delta(pitch(actual), combinations[15])))
        reference_projection = float(np.rad2deg(angle_delta(combinations[0], pitch(desired))))
        assert abs(sum(contributions) + passive + reference_projection - event['pitch_error_deg']) < 1e-8
        gradients = []
        for address in AXES:
            a = desired.copy(); b = desired.copy()
            a[address] += 1e-6; b[address] -= 1e-6
            gradients.append(float(np.rad2deg(angle_delta(pitch(project(a)), pitch(project(b)))) / 2e-6))
        slopes = np.diff(errors)
        changes = int(np.count_nonzero(slopes[1:] * slopes[:-1] < 0))
        spectrum = np.fft.rfft(errors - np.polyval(np.polyfit(np.arange(len(errors)), errors, 1), np.arange(len(errors))))
        frequencies = np.fft.rfftfreq(len(errors), 1 / 60)
        high = frequencies >= 2.
        power = abs(spectrum) ** 2
        metrics[phase] = dict(frames=len(group), pitch_error_deg=summarize(errors),
            peak_error_frame=event['frame'], pitch_error_turning_points=changes,
            detrended_power_at_or_above_2Hz_fraction=float(power[high].sum() / max(power.sum(), 1e-30)),
            passive_projection_pitch_delta_deg=summarize([r['passive_projection_pitch_delta_deg'] for r in group]),
            axis_reference_error_si={name:summarize([r[name + '_reference_error_si'] for r in group]) for name in ('swing','boom','arm','bucket')},
            bucket_integral_N=summarize([r['bucket_integral_N'] for r in group]))
        peaks.append(dict(phase=phase, frame=event['frame'], pitch_error_deg=event['pitch_error_deg'],
            exact_kinematic_axis_contribution_deg=dict(zip(('swing','boom','arm','bucket'), contributions)),
            passive_closure_contribution_deg=passive, reference_projection_contribution_deg=reference_projection,
            local_pitch_gradient_degrees_per_SI=dict(zip(('swing','boom','arm','bucket'), gradients))))
    baseline = {}
    for phase in ('CLEAR_PILE','ALIGN'):
        old = [x for x in rows(OLD / 'geometry-actions.ndjson') if x['phase'] == phase]
        if phase == 'ALIGN':
            old = old[:120]
        e = [float(np.rad2deg(angle_delta(x['detail']['pitch_diagnostic_rad'], pitch(x['desired'])))) for x in old]
        baseline[phase] = dict(frames=len(old), pitch_error_deg=summarize(e),
            scope='Saved original trajectory; ALIGN first120 only; changed durations/profile prevent same-time causal comparison.')
    boundary = []
    limits = np.array([.12,.0225,.0225,.0225])
    for i, x in enumerate(actions):
        if i == 0 or x['phase'] != actions[i-1]['phase']:
            previous = np.array(dispatch[x['frame'] - 2]['command'])
            current = np.array(x['command'])
            boundary.append(dict(frame=x['frame'], phase=x['phase'],
                normalized_command_change=float(np.max(abs(current-previous)/limits)),
                old_slew_limit=2/60, previous_command=previous.tolist(), command=current.tolist(),
                load_bias=x['detail']['load_bias']))
    checks = dict(all_input_hashes_unchanged=all(sha(p)==h for p,h in before_hashes.items()),
        aligned_before_after_dispatch_and_reference=True, exact_pitch_recomputed=True,
        shapley_sum_matches_peak=True,
        all_bucket_active_commands_zero=all(x['command'][3]==0 for x in actions),
        all_guard_records_passed=all(x['hard_guard'] and x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in physics),
        no_safety_projection_changed_command=not any(x['detail']['projected'] for x in actions),
        phase_boundary_commands_within_original_slew=all(x['normalized_command_change']<=2/60+1e-9 for x in boundary),
        no_native_or_warp_runtime_imported=all(x not in sys.modules for x in ('warp','newton','mujoco','layered_47_cycle_runtime')))
    result = dict(status='diagnosis_passed' if all(checks.values()) else 'diagnosis_failed',
        checks=checks, source_hashes=before_hashes, metrics=metrics, peak_decomposition=peaks,
        baseline=baseline, phase_boundaries=boundary, physics_steps=0, native_initializations=0,
        inference_calls=0, training_updates=0, wall_seconds=time.monotonic()-started,
        limitations=['Saved 60Hz snapshots do not contain per-substep effort/contact/wrench history.',
                     'Kinematic partition quantifies angle sensitivity; it does not identify initiating physical forces.',
                     'Candidate control evaluated on recorded states would not predict a new closed-loop response.'])
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','metrics','peak_decomposition','phase_boundaries','wall_seconds')},indent=2))
    assert all(checks.values()), checks


if __name__ == '__main__':
    main()
