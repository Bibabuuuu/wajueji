from pathlib import Path
from datetime import datetime,timezone,timedelta
import time,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-minimum-20261014-v1'
W.mkdir(exist_ok=False);t=time.time();tz=timezone(timedelta(hours=8))
d=dict(purpose='second_scoop_minimum_restore_approach_short',seconds=2700,epoch_start=t,epoch_deadline=t+2700,
 started_at=datetime.fromtimestamp(t,tz).isoformat(),deadline_at=datetime.fromtimestamp(t+2700,tz).isoformat(),reserve_seconds=420,
 max_frames=5000,max_residual_frames=120,training_updates_allowed=0,
 authorization='User explicitly approved a NEW45-minute minimum-height version: restore qualified original return, rise/approach and <=120 frozen residual frames; prep/save/audit included; failure=>save and stop, no training/full scoop',
 old_budgets='all closed, no reuse',source='runs/continuous-single-scoop-20261014-v1/final.npz',source_sha256='6fe4091069bdbedb2ea99e5a44475047f8f4367957a1fb0c5745f0045137ef8e')
(W/'budget.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
