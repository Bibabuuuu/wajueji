"""Read-only CPU audit of existing control logs; no runtime/physics construction.

Only the sibling review.json is written. Source files and frozen results are
hashed before and after. Cached poses establish geometry, not load causality.
"""
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP, tool_pose

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
THIRD = ROOT / 'runs/contact-full-response-20260924-v1/candidate60'
FIRST = ROOT / 'runs/contact-multisource-lift-20260924-v1/first-candidate'
MANUAL = ROOT / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-115812-1458627'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def closure(state):
    return (state['mechanism_qpos'][15] - .370) / (.852 - .370)


def windows(frames):
    result = []
    for frame in frames:
        if result and frame == result[-1][-1] + 1:
            result[-1].append(frame)
        else:
            result.append([frame])
    return [dict(first_frame=w[0], last_frame=w[-1], count=len(w),
                 command_duration_s=len(w)/60) for w in result]


def autonomous(rows):
    f80 = next(r['frame'] for r in rows if r['feedback']['closure_fraction'] >= .8)
    f95 = next(r['frame'] for r in rows if r['feedback']['closure_fraction'] >= .95)
    between = [r for r in rows if f80 <= r['frame'] < f95]
    curl = next(r for r in rows if r['phase'] == 'curl_load')
    f = curl['feedback']
    return dict(
        frames=len(rows), state_convention='before_state: state at frame-1 end',
        first_command_with_80pct_state=f80, first_command_with_95pct_state=f95,
        intervals_80_to_95=f95-f80, simulation_seconds_80_to_95=(f95-f80)/60,
        preview_scales_80_to_95=dict(Counter(str(r['feedback']['preview_scale']) for r in between)),
        preview_scales_all=dict(Counter(str(r['feedback']['preview_scale']) for r in rows)),
        zero_command_windows=windows([r['frame'] for r in rows if all(x == 0 for x in r['applied_command_si'])]),
        first_bc_curl_frame=curl['frame'],
        planned_cut_retract_m=f['contact'][0]-f['cut_target'][0],
        actual_contact_to_first_bc_retract_m=f['contact'][0]-f['coordinates'][0],
        first_bc_coordinates=f['coordinates'], soil_plan=f['soil_plan'])


def main():
    reports = {k: read(p/'report.json') for k, p in [('third', THIRD), ('first', FIRST), ('manual', MANUAL)]}
    assert reports['third']['source']['sha256'] == reports['manual']['source_checkpoint']['sha256']
    assembly, chain = _load_kinematics(KIN, KIN/'scene/input/model/A.xml', KIN/'scene/input/mesh')
    source = Path(reports['third']['source']['path'])
    paths = [Path(__file__), source, KIN/'scene/input/model/A.xml',
             Path(inspect.getfile(type(assembly))), Path(inspect.getfile(type(chain))),
             Path(inspect.getfile(tool_pose)), ROOT/'src/v4/layered_excavation_approach_planner.py',
             ROOT/'src/v4/contact_geometric_feedback.py', ROOT/'src/v4/contact_cut_reference.py',
             ROOT/'src/v4/contact_phase_candidate.py',
             ROOT/'work/contact-preview-lift-20260924-v1/lift_feedback.py',
             ROOT/'work/contact-preview-lift-20260924-v1/fill-review.json',
             ROOT/'work/contact-derived-demonstration-20260924-v1/fill-comparison-v2.json']
    for p in (THIRD, FIRST, MANUAL):
        paths.extend([p/'report.json', p/('physics.ndjson' if p == MANUAL else 'actions.ndjson')])
    paths = sorted(set(paths))
    before = {str(p): sha(p) for p in paths}
    assert before[str(source)] == reports['third']['source']['sha256']
    third, first, manual = lines(THIRD/'actions.ndjson'), lines(FIRST/'actions.ndjson'), lines(MANUAL/'physics.ndjson')
    radial = np.asarray(reports['third']['selected_goal']['entry_xy_m'], float)
    radial /= np.linalg.norm(radial)
    lateral = np.array([-radial[1], radial[0]])
    geometry = {}
    for name, rows, field in [('manual_legacy_physics', manual, 'after_state'), ('autonomous_full_response', third, 'before_state')]:
        samples = []
        for fraction in (.56, .64, .8, .9, .95, .99):
            # Source closure is < .56; first crossing is unambiguous here.
            row = next(r for r in rows if closure(r[field]) >= fraction)
            state = row[field]
            tip, pitch = tool_pose(assembly, np.asarray(state['mechanism_qpos']), TIP)
            samples.append(dict(threshold=fraction, record_frame=row['frame'],
                state_frame_end=row['frame'] if field == 'after_state' else row['frame']-1,
                actual_closure_fraction=closure(state), tip_world_m=tip.tolist(),
                tip_radial_m=float(tip[:2]@radial), tip_lateral_m=float(tip[:2]@lateral),
                pitch_rad=float(pitch), bucket_origin_world_m=state['bucket_origin_world_m']))
        geometry[name] = samples
    after = {str(p): sha(p) for p in paths}
    assert before == after
    report = dict(schema='contact_intake_control_readonly_review_v1',
        status='cpu_cached_log_and_pose_audit_passed', physics_steps_added=0, training_updates=0,
        inputs_sha256=before, inputs_unchanged=True,
        same_initial_checkpoint=True, initial_checkpoint_sha256=before[str(source)],
        comparison_is_causal_ab=False,
        limitations=[
            'Manual and autonomous runs share an initial checkpoint but use different physics/contact interfaces and actions.',
            'Pose/command differences do not establish missing-load causality or quantify early spills.',
            'D152 includes the historical preview stall; its 48.81% result does not evaluate the corrected full-entry D154 controller.',
            '80% no-down is a confirmed user constraint; this review does not authorize its removal.',
            'Manual endpoints use after_state; autonomous samples use before_state, with explicit state_frame_end.',
            'The .575m target was reached before BC handoff in both audited autonomous runs.'],
        third=autonomous(third), first=autonomous(first),
        manual_80_to_95_intervals=geometry['manual_legacy_physics'][4]['state_frame_end']-geometry['manual_legacy_physics'][2]['state_frame_end'],
        selected_radial_unit_xy=radial.tolist(), same_source_geometry=geometry)
    (HERE/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'inputs_unchanged', 'physics_steps_added', 'same_initial_checkpoint']}))


if __name__ == '__main__':
    main()
