from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-phased-20261014-v1';D=R/'runs/second-scoop-short-phased-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (W/'closure.json').exists()
b=read(W/'budget.json');p=read(D/'report.json');a=read(D/'read-only-audit.json');t=time.time()
assert t<b['epoch_deadline'] and p['status']=='second_scoop_short_completed' and p['frames']==4882 and a['status']=='passed' and all(a['checks'].values())
c=dict(status='closed_after_completed_short_and_audit',closed_at=datetime.fromtimestamp(t,timezone(timedelta(hours=8))).isoformat(),epoch_closed=t,
 budget_sha256=sha(W/'budget.json'),epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],elapsed_seconds=t-b['epoch_start'],sealed_remaining_seconds=b['epoch_deadline']-t,
 physical_frames=4882,physics_seconds=4882/60,native_initializations=1,policy_predictions=40,residual_frames=120,training_updates=0,
 report_sha256=sha(D/'report.json'),audit_sha256=sha(D/'read-only-audit.json'),aligned_source_candidate_sha256=sha(W/'aligned-source-candidate.json'),
 reason='Authorized short scope completed and independently audited; leftover physical budget sealed. Full second excavation and new native prepoint restore require separately authorized scope/budget.')
(W/'closure.json').write_text(json.dumps(c,indent=2));print(json.dumps(c,indent=2))
