from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-20261014-v1';D=R/'runs/second-scoop-short-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (W/'closure.json').exists()
b=read(W/'budget.json');p=read(D/'report.json');a=read(D/'read-only-audit.json')
assert p['status']=='failed' and p['frames']==371 and a['status']=='diagnostic_audit_passed' and all(a['checks'].values())
t=time.time();assert t<b['epoch_deadline']
c=dict(status='closed_after_guard_failure',closed_at=datetime.fromtimestamp(t,timezone(timedelta(hours=8))).isoformat(),epoch_closed=t,
 budget_sha256=sha(W/'budget.json'),epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],elapsed_seconds=t-b['epoch_start'],sealed_remaining_seconds=b['epoch_deadline']-t,
 physical_frames=p['frames'],physics_seconds=p['frames']/60,native_initializations=1,policy_predictions=0,residual_frames=0,training_updates=0,
 report_sha256=sha(D/'report.json'),audit_sha256=sha(D/'read-only-audit.json'),stop_diagnosis_sha256=sha(D/'stop-diagnosis.json'),
 reason='User scope required failure=>save and stop. No retry; subsequent minimum-height CPU candidate is a separately confirmed CPU-only step, not physical reuse of this budget.')
(W/'closure.json').write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))
