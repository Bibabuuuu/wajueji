from pathlib import Path
import datetime, json, time
R=Path('/home/qunwang/wajueji')
W=R/'work/edge-adjacent-59um-20261015-v1'
W.mkdir(exist_ok=False)
start=datetime.datetime.fromisoformat('2026-10-15T09:09:46+08:00').timestamp()
now=time.time()
b=dict(purpose='edge_adjacent_second_59um',seconds=3600,epoch_start=start,epoch_deadline=start+3600,
       monotonic_deadline=time.monotonic()+start+3600-now,reserve_seconds=600,
       max_frames=13000,max_dig_frames=7200,training_updates_allowed=0,
       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
       source=str(R/'runs/edge-first-two-scoop-20261015-continuation/first-returned.npz'),
       tolerance_um=59,authorization='User explicitly selected new60min, restore and complete second dig; no second transport or training',
       old_budgets_reopened=False)
(W/'budget.json').write_text(json.dumps(b,indent=2))
print(json.dumps(dict(start_local='2026-10-15T09:09:46+08:00',deadline_local='2026-10-15T10:09:46+08:00',remaining_seconds=start+3600-now)))
