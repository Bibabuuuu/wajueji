import json
import hashlib
from pathlib import Path
import sys

paths = [Path(name).resolve() for name in sys.argv[1:]]
if len(paths) != 4:
    raise SystemExit('Expected 120Hz candidate, 60Hz candidate, 120Hz audit, 60Hz audit')
reports = [json.loads(path.read_text()) for path in paths]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
seed = reports[1].get('seed', {})
passed = (all(r['status'] == 'candidate_short_hold_passed' for r in reports[:2])
          and all(r['status'] == 'sparse_resume_audit_passed' for r in reports[2:])
          and all(r.get('checks') and all(r['checks'].values()) for r in reports)
          and len({r['rocks'] for r in reports}) == 1
          and [r['policy']['hz'] for r in reports[:2]] == [120, 60]
          and [r['hz'] for r in reports[2:]] == [120, 60]
          and [r.get('transition') for r in reports[:2]] == ['grid', 'frequency']
          and reports[0]['source_checkpoint'] == reports[1]['source_checkpoint']
          and seed.get('candidate_report_sha256') == digest(paths[0])
          and seed.get('resume_report_sha256') == digest(paths[2])
          and all(r['candidate_report_sha256'] == digest(paths[i]) for i, r in enumerate(reports[2:])))
rates = [r['timing']['simulation_seconds_per_wall_second'] for r in reports[:2]]
print('COPY_RESULT_BEGIN')
print(json.dumps(dict(status='sparse_pair_short_hold_passed' if passed else 'sparse_pair_rejected',
    report_paths=[str(p) for p in paths], rocks=reports[0]['rocks'],
    speedup=rates[1]/rates[0] if passed and all(rates) else None,
    comparison='sequential_qualified_static_windows_not_identical_initial_state_actions',
    timing=[r['timing'] for r in reports], checks=[r.get('checks') for r in reports],
    all_transition_frames_strict=[r['all_transition_frames_strict'] for r in reports[:2]],
    next_gate='review_and_noncontact_motion_before_contact_no_auto_promotion',
    formal_configuration_changed=False, training_eligible=False), indent=2))
print('COPY_RESULT_END')
raise SystemExit(0 if passed else 1)
