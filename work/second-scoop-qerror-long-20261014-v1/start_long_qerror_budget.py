from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';W.mkdir(exist_ok=False)
t=time.time();tz=timezone(timedelta(hours=8));source=R/'work/second-scoop-short-phased-20261014-v1/aligned-source-candidate.json';a=json.loads(source.read_text())
d=dict(purpose='second_scoop_actual_joint_wait_long_validation',seconds=1800,epoch_start=t,epoch_deadline=t+1800,
 started_at=datetime.fromtimestamp(t,tz).isoformat(),deadline_at=datetime.fromtimestamp(t+1800,tz).isoformat(),reserve_seconds=300,
 max_frames=7200,training_updates_allowed=0,prefix_frames=3,
 source=a['source'],source_sha256=a['source_sha256'],source_metadata_sha256=a['metadata_sha256'],source_control_sha256=a['control_sha256'],
 authorization='User chose NEW30 minutes actual-joint wait mapping: restore aligned pre-dig checkpoint, zero24/native gate and first3 saved SAC-prefix gate, then same instance full1170 reference plus30-frame retention; max7200 residual frames, prep/save/audit included; no training/transport; failure or deadline=>save and stop',old_budgets='all sealed, no reuse',observation_time_denominator=1560,accepted_risk='time observation may exceed training range; cached20-case sensitivity is not rollout qualification')
(W/'budget.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
