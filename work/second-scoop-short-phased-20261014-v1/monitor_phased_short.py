from pathlib import Path
import json,time,sys
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-phased-20261014-v1';D=R/'runs/second-scoop-short-phased-20261014-v1'
b=json.loads((W/'budget.json').read_text());p=json.loads((D/'report.json').read_text())
print(json.dumps({k:p.get(k) for k in ['status','frames','phase','failure','residual_frames','reference_cursor']},ensure_ascii=False))
print('budget_remaining_seconds',round(b['epoch_deadline']-time.time()))
with (D/'physics/frames.ndjson').open() as f:
 first=json.loads(next(f));last=first
 for line in f:last=json.loads(line)
print('last_physics_frame',last['frame'],'hard_guard',last['hard_guard'])
g=D/'geometry-clearance.ndjson'
if g.exists():
 with g.open() as stream:
  for line in stream:latest=json.loads(line)
 print('last_geometry',json.dumps({k:latest.get(k) for k in ['global_frame','phase','axis_error','bucket_observed_terrain_m','minimum_height_step_m','closure_m']}))
if '--schema' in sys.argv:
 print('frame_keys',list(first));print('state_keys',list(first['after_state']))
 print('wrench_fields',{k:v for k,v in first.items() if 'wrench' in k})
print('\n'.join((W/'console.log').read_text().splitlines()[-4:]))
