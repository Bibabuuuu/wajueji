"""Record complete batch quality before the bounded offline pilot, without relabeling."""
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = W / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assignment = json.loads((W / 'assignment.json').read_text())
    assert assignment['source_manifest_sha256'] == sha(manifest_path)
    episodes = []
    for entry in manifest['episodes']:
        for key in ('report', 'independent_audit', 'fill_audit'):
            assert sha(entry[key]['path']) == entry[key]['sha256']
        report = json.loads(Path(entry['report']['path']).read_text())
        trace_path = Path(entry['report']['path']).parent / 'phase-trace.ndjson'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert len(trace) == report['frames_completed']
        unresolved = [r['frame'] for r in trace if r['phase']['curl_boundary_unresolved']]
        assert not unresolved, (entry['run'], 'unresolved_curl_boundary')
        assert all(r['ground']['safe'] for r in trace)
        counts = entry['phase_counts']
        assert all(counts[p] > 0 for p in ('approach_contact', 'penetrate', 'curl_load', 'lift'))
        episodes.append(dict(
            run=entry['run'], split=entry['split'], expert_rows=entry['rows'],
            phase_counts=counts, actual_fill_percent=entry['actual_fill_percent'],
            fit_weight=assignment['episode_weights'][entry['run']],
            fill_difference_percentage_points=entry['fill_difference_percentage_points'],
            off_soil_confirmation_frame_end=entry['off_soil_confirmation_frame_end'],
            subsequent_below_3mm_frames=len(entry['post_confirmation_below_gap_frames']),
            minimum_bucket_z_m=min(r['ground']['minimum_bucket_z_m'] for r in trace),
            unresolved_curl_frames=unresolved, trace_sha256=sha(trace_path),
            maximum_rigid_deviations=entry['maximum_rigid_deviations']))
    result = dict(
        schema='derived_batch_limited_pilot_review_v1', passed=True,
        manifest_sha256=sha(manifest_path), reviewer_sha256=sha(__file__), episodes=episodes,
        episode_count=len(episodes), expert_rows=sum(e['expert_rows'] for e in episodes),
        limited_offline_pilot_admitted=True, formal_training_admission=False,
        physics_rollout_qualified=False, phase_rules_changed=False,
        validation_domain_caveats=assignment['validation_domain_caveats'],
        qualifications=[
            'Original applied actions, lineage split and goal masks retained. Original episode weights retained except diagnostic-only 100857 is zero for fitting and normalization.',
            'Micro replay differences are recorded under the user decision, not abort thresholds.',
            'First causal clearance is latched; subsequent recontacts remain and do not certify stable transport.',
            'Shared original terrain and previously inspected validation trajectories are not a sealed generalization test.',
            'Only a fixed 1000-update / 300-second offline phase-conditioned pilot is admitted.'])
    (W / 'batch-quality-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
