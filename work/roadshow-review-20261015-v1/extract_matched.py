from pathlib import Path
import json,csv
R=Path('/home/qunwang/wajueji');O=R/'work/roadshow-review-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
manual=R/'runs/roadshow-three-way-20261014-v1/manual'
res=read(R/'runs/roadshow-three-way-20261014-v1/residual-v3/report.json')
policy=R/'runs/roadshow-residual-media-20261014-v1'
assert read(policy/'review.json')['status']=='captured_frozen_verified_repeat'
assert read(policy/'review.json')['result']['source_sha256']==res['source']['sha256']
for case,folder in [('manual',manual),('residual',policy)]:
 states=read(folder/'display/states.json')
 with (O/f'{case}-trajectory.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['frame','simulation_time_s','bucket_origin_x_m','bucket_origin_y_m','bucket_origin_z_m'])
  for i,s in enumerate(states):w.writerow([i,i/60,*s['bucket_origin_world_m']])
 print(case,'STATES',len(states),'FIRST',states[0]['bucket_origin_world_m'],'LAST',states[-1]['bucket_origin_world_m'])
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
write(O/'matched-sources.json',dict(manual=str(manual),residual=str(policy),source_sha256=res['source']['sha256']))
