"""Audit one manual episode as a potential supervised residual source, read-only."""
from pathlib import Path
from collections import Counter
import hashlib
import json

ROOT = Path('/home/qunwang/wajueji')
SOURCE = ROOT / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20261012-121413-64218'
OUT = ROOT / 'runs/roadshow-measurements-20261011/manual-training-audit-20261012-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(name):
    path = SOURCE / (name + '.ndjson')
    return [json.loads(line) for line in path.open()], dict(path=str(path), sha256=sha(path))


def main():
    if OUT.exists():
        raise ValueError('immutable_output_exists')
    input_rows, input_source = rows('input')
    physics_rows, physics_source = rows('physics')
    observation_rows, observation_source = rows('observation')
    annotations, annotation_source = rows('annotations')
    report_path = SOURCE / 'report.json'
    episode = json.loads(report_path.read_text())
    count = len(input_rows)
    assert count == len(physics_rows) == 1169
    assert [x['frame'] for x in input_rows] == list(range(1, count+1))
    assert [x['frame'] for x in physics_rows] == list(range(1, count+1))
    assert [x['frame'] for x in observation_rows] == list(range(3, count+1, 3))
    assert all(x['state']['mechanism_qpos'] == physics_rows[x['frame']-1]['after_state']['mechanism_qpos'] for x in observation_rows)
    assert all(x['applied_command_si'] == y['applied_command_si'] and
               x['requested_command_si'] == y['requested_command_si']
               for x, y in zip(input_rows, physics_rows, strict=True))
    assert all(not x['training_eligible'] for x in input_rows+observation_rows)
    projected = [x for x in input_rows if x['command_projection_reason'] is not None]
    axis_projection_counts = [sum(abs(x['requested_command_si'][a]-x['applied_command_si'][a])>1e-12
                                  for x in input_rows) for a in range(4)]
    assert axis_projection_counts == [0, 0, 0, 390]
    phase_counts = dict(Counter(x['phase'] for x in input_rows))
    marker_frames = {x['label']: x['next_frame'] for x in annotations if x['kind']=='phase'}
    assert marker_frames == {'penetrate_deepen':247, 'lift_clear':755, 'off_soil_confirm':1049}
    assert 'curl_load' not in marker_frames
    zones = [(1,246,'unmarked'),(247,754,'penetrate_deepen'),(755,1048,'lift_clear'),
             (1049,1139,'off_soil_confirm'),(1140,1169,'terminal_zero_hold')]
    segments = []
    for first,last,phase in zones:
        batch = input_rows[first-1:last]
        start = physics_rows[first-1]['after_state']['bucket_origin_world_m'][2]
        end = physics_rows[last-1]['after_state']['bucket_origin_world_m'][2]
        segments.append(dict(first_frame=first,last_frame=last,phase=phase,rows=len(batch),
                             zero_command_frames=sum(all(abs(v)<1e-12 for v in x['applied_command_si']) for x in batch),
                             bucket_origin_rise_m=end-start,
                             upper_target_projection_frames=sum(x['command_projection_reason']=='bucket_internal_upper_target' for x in batch)))
    result = dict(status='audited_training_not_granted', source_run=str(SOURCE),
                  source_sha256=[input_source, physics_source, observation_source, annotation_source,
                                 dict(path=str(report_path), sha256=sha(report_path))],
                  acquisition_domain=episode['acquisition_domain'],
                  same_current_full_response_domain_verified=False,
                  episode_training_eligible=episode['episode_summary']['training_eligible'],
                  frames=count, observation_frames=len(observation_rows), observation_stride=3,
                  observation_same_frame_is_post_action_state=True,
                  input_physics_commands_exact=True,
                  phase_counts=phase_counts, marker_frames=marker_frames,
                  missing_curl_marker=True,
                  projection=dict(frames=len(projected), first=projected[0]['frame'],
                                  last=projected[-1]['frame'], reason_counts=dict(Counter(x['command_projection_reason'] for x in projected)),
                                  affected_axis_counts=axis_projection_counts),
                  zero_all_axis_frames=sum(all(abs(v)<1e-12 for v in x['applied_command_si']) for x in input_rows),
                  segments=segments,
                  manually_marked_off_soil_not_independently_confirmed=True,
                  live_fill_valid=episode['live_bucket_fill']['last_display_result']['valid_estimate'],
                  live_fill_invalid_reasons=episode['live_bucket_fill']['last_display_result']['quality_reasons'],
                  conclusion='Retain as bounded manual diagnostic; do not add to current D152 residual-BC fit or validation without domain and causal-label qualification. Re-recording through the same old entry would retain the domain gap.')
    OUT.mkdir()
    (OUT / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], frames=count, observations=len(observation_rows),
                          projected=len(projected), marker_frames=marker_frames,
                          domain=episode['acquisition_domain'], live_fill_valid=result['live_fill_valid'])))


if __name__ == '__main__':
    main()
