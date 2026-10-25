from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-dig-20261014-v1';D=R/'runs/second-scoop-qerror-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (W/'closure.json').exists()
b=read(W/'budget.json');p=read(D/'report.json');a=read(D/'read-only-audit.json');t=time.time()
assert p['status'] in ('failed','full_second_dig_completed') and p['physical_run_finished_epoch']<=t
assert a['status'] in ('passed','diagnostic_audit_passed') and all(a['checks'].values())
c=dict(status='closed_after_actual_joint_wait_validation',closed_at=datetime.fromtimestamp(t,timezone(timedelta(hours=8))).isoformat(),epoch_closed=t,
 budget_sha256=sha(W/'budget.json'),epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],elapsed_seconds=t-b['epoch_start'],sealed_remaining_seconds=max(0,b['epoch_deadline']-t),
 physical_frames=p['frames'],physics_seconds=p['frames']/60,native_initializations=p['native_initializations'],policy_predictions=a['policy_decisions'],reference_cursor=p['reference_cursor'],prefix3_exact=a['prefix3_exact'],training_updates=p['neural_updates'],
 report_sha256=sha(D/'report.json'),audit_sha256=sha(D/'read-only-audit.json'),reason='Authorized wait-recovery/second-dig attempt ended. Save/stop; remaining time sealed; no transport/training/retry or third scoop.')
(W/'closure.json').write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))
