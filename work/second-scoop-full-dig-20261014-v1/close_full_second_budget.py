from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';D=R/'runs/second-scoop-full-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (W/'closure.json').exists()
b=read(W/'budget.json');p=read(D/'report.json');a=read(D/'read-only-audit.json');t=time.time()
assert t<b['epoch_deadline'] and p['status']=='failed' and p['frames']==1560 and a['status']=='diagnostic_audit_passed' and all(a['checks'].values())
c=dict(status='closed_after_frozen_dig_timeout',closed_at=datetime.fromtimestamp(t,timezone(timedelta(hours=8))).isoformat(),epoch_closed=t,
 budget_sha256=sha(W/'budget.json'),epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],elapsed_seconds=t-b['epoch_start'],sealed_remaining_seconds=b['epoch_deadline']-t,
 physical_frames=1560,physics_seconds=26.,native_initializations=1,policy_predictions=521,reference_cursor=76.,prefix3_exact=True,training_updates=0,
 report_sha256=sha(D/'report.json'),audit_sha256=sha(D/'read-only-audit.json'),diagnosis_sha256=sha(D/'stall-diagnosis.json'),
 reason='Failed at1560-frame cap. Save/stop; no transport/training/retry. Any separately chosen CPU correction candidate adds no physical authorization.')
(W/'closure.json').write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))
