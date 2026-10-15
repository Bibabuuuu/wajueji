"""Versioned CPU-only rescoring; original physics and labels remain immutable."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest
from manual_residual_sac_env import retained_score, TERMINAL_SCORE_VERSION, INPUT, SOURCE, INPUT_SHA, SOURCE_SHA

root=Path('/home/qunwang/wajueji')
prior=root/'runs/manual-wait-strength-validation-20261013-v2'
out=root/'runs/manual-hold-rescore-20261013-v1'
out.mkdir(exist_ok=False);started=time.monotonic()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((prior/'review.json').read_text())
audit=json.loads((prior/'read-only-audit.json').read_text())
paths=[prior/'review.json',prior/'read-only-audit.json',prior/'final-review.json']
for entry in original['episodes']:
    d=prior/entry['label']/'episode-0001'
    paths += [d/n for n in ['report.json','actions.ndjson','alignment.ndjson','policy.ndjson','physics/frames.ndjson','endpoint.npz']]
before={str(p):digest(p) for p in paths}
checks={'original_physics_audit_passed':audit['status']=='passed' and all(audit['global_checks'].values()),
        'source_snapshot_matches':all(digest(prior/'source-snapshot'/n)==h for n,h in original['code'].items()),
        'original_sources_unchanged':digest(INPUT)==INPUT_SHA and digest(SOURCE)==SOURCE_SHA,
        'cuda_disabled':os.environ.get('CUDA_VISIBLE_DEVICES')==''}
code={}
for name in ['src/v4/manual_residual_sac_env.py','tests/v4/test_manual_hold_score.py',
             'work/manual-hold-rescore-20261013/rescore_manual_hold.py']:
    p=root/name;code[name]=digest(p);target=out/'source-snapshot'/name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
sys.path.insert(0,str(root/'tests/v4'))
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
    unittest.defaultTestLoader.discover(str(root/'tests/v4'),pattern='test_manual*.py'))
(out/'tests.txt').write_text(stream.getvalue());checks['cpu_regression_passed']=result.wasSuccessful()
episodes=[]
for entry in original['episodes']:
    d=prior/entry['label']/'episode-0001';r=json.loads((d/'report.json').read_text())
    actions=[json.loads(s) for s in (d/'actions.ndjson').read_text().splitlines()]
    physics=[json.loads(s) for s in (d/'physics/frames.ndjson').read_text().splitlines()]
    alignments={v['frame']:v for v in (json.loads(s) for s in (d/'alignment.ndjson').read_text().splitlines()) if v['stage']=='after'}
    c={'executed_frame_count':len(actions)==len(physics)==r['physical_frames'],
       'original_source_unchanged':r['source_unchanged'],
       'all_executed_protections':all(p['hard_guard'] and all(p['hard_guard_checks'].values()) and
                                    p['hencky_material_guard']['safe'] and p['moving_bucket_sweep']['safe'] for p in physics)}
    eligible=r['status'] in ['completed','invalid_endpoint'] and len(r['scores'])==3
    score=retained_score(r['scores']) if eligible else None
    if eligible:
        first=r['scores'][0]['frame'];last=r['scores'][-1]['frame']
        c['completed_reference_and_pose_at_hold_entry']=actions[first-1]['cursor_before']+actions[first-1]['pace']>=1170 and alignments[first]['endpoint_ready']
        c['three_sample_times']=[s['frame'] for s in r['scores']]==[first,first+15,first+30]
        c['full_half_second_hold']=last==r['physical_frames'] and last-first==30
        c['actual_hold_frames_and_zero_commands']=all(a['hold_frame']==i and all(v==0 for v in a['request_si']) for i,a in enumerate(actions[first:last]))
        c['three_valid_measurements']=score is not None
    else:
        c['timeout_stays_failure']=r['status']=='timeout' and score is None
    success=eligible and score is not None and all(c.values())
    checks[entry['label']+'_evidence_complete']=all(c.values())
    episodes.append(dict(label=entry['label'],strength=entry['strength'],original_status=r['status'],
                         original_success=r['success'],reevaluated_status='completed' if success else r['status'],
                         reevaluated_success=success,physical_frames=r['physical_frames'],
                         qualified_fill_percent=100*score if success else None,
                         new_raw_terminal_score=score if success else -2.,checks=c,
                         gap_checks=r['gap_checks'],gap_role='diagnostic_only'))
baseline=episodes[0]['qualified_fill_percent']
for e in episodes:
    e['difference_from_manual_percentage_points']=e['qualified_fill_percent']-baseline if e['qualified_fill_percent'] is not None else None
checks['original_files_unchanged']=all(digest(p)==h for p,h in before.items())
review=dict(status='passed' if all(checks.values()) else 'failed',terminal_score_version=TERMINAL_SCORE_VERSION,
            checks=checks,check_count=len(checks)+sum(len(e['checks']) for e in episodes),
            cpu_tests=result.testsRun,test_errors=len(result.errors),test_failures=len(result.failures),
            code=code,original_evidence_sha256=before,episodes=episodes,
            elapsed_seconds=time.monotonic()-started,new_physics_frames=0,neural_updates=0,
            training_admission=False,old_replay_relabelled=False,
            score_scope='Minimum of three valid held bucket display estimates at 0/.25/.5s, including rocks/heap; not truck delivery.',
            conclusion='Two fixed recovery controls complete the corrected hold criterion; both remain below the manual baseline. No learned improvement is demonstrated.')
(out/'review.json').write_text(json.dumps(review,indent=2))
print(json.dumps({k:review[k] for k in ['status','cpu_tests','test_errors','test_failures','checks','check_count','elapsed_seconds','episodes']}))
sys.exit(0 if all(checks.values()) else 1)
