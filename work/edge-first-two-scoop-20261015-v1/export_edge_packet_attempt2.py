"""Export small source-data packet after audit; no new physics or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,time
import numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_edge_pair_attempt2 as m
D=m.OUT;O=D/'summary';O.mkdir(exist_ok=False);p=m.read(D/'report.json');audit=m.read(D/'read-only-audit.json');assert audit['record_integrity_passed']
old1=R/'runs/continuous-single-scoop-20261014-v1';old2=R/'runs/second-scoop-qerror-long-20261014-v1'
historical=[dict(scoop=1,retained_fill_fraction=m.read(old1/'report.json')['residual_result']['retained_fill_fraction'],mapping='native_target_wait'),dict(scoop=2,retained_fill_fraction=m.read(old2/'report.json')['residual_result']['retained_fill_fraction'],mapping='actual_joint_wait')]
sources={str(x):m.sha(x) for x in [D/'report.json',D/'read-only-audit.json',W/'budget.json',W/'live-protocol-attempt2.json',old1/'report.json',old2/'report.json']}
summary=dict(status=p['status'],failure=p.get('failure'),two_scoop_complete=audit['two_scoop_goal_complete'],new_scoops=audit['scoops'],historical=historical,
 historical_sum_display_percentage_points=sum(x['retained_fill_fraction']*100 for x in historical),
 new_sum_display_percentage_points=(sum(x['retained_fill_fraction']*100 for x in audit['scoops']) if len(audit['scoops'])==2 and all(x['complete'] for x in audit['scoops']) else None),
 sources_sha256=sources,physical_frames=p['frames'],physical_seconds=p['frames']/60,native_instances=p['native_initializations'],training_updates=p['neural_updates'],
 caveats=['One exploratory pair, no statistical success-rate claim','Historical central pair has different first-scoop wait mapping and a second-scoop restore; not a matched causal placement comparison','Display fill has not been calibrated as bulk payload or stable received volume','A sum of display percentages is a descriptive loading index, not measured total volume'])
with (O/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=['layout','scoop','completed','retained_display_percent','dig_frames','dig_seconds','wait_frames','wait_fraction','mapping']);writer.writeheader()
 for x in historical:writer.writerow(dict(layout='historical_central',scoop=x['scoop'],completed=True,retained_display_percent=x['retained_fill_fraction']*100,mapping=x['mapping']))
 for i in (1,2):
  item=next((x for x in audit['scoops'] if x['scoop']==i),None)
  writer.writerow(dict(layout='edge_adjacent',scoop=i,completed=bool(item and item['complete']),retained_display_percent=(item['retained_fill_fraction']*100 if item and item['complete'] else ''),dig_frames=(item['frames'] if item else ''),dig_seconds=(item['frames']/60 if item else ''),wait_frames=(item['wait_frames'] if item else ''),wait_fraction=(item['wait_fraction'] if item else ''),mapping='actual_joint_wait'))
m.write(O/'results.json',summary)
r=m.CpuRuntime(D/'zero.npz',W/('export-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
paths={}
for i in (1,2):
 file=D/('scoop-%d-plan'%i)/'actual-cut-trace.json'
 if not file.exists():continue
 trace=m.read(file)['trace'];path=[]
 for item in trace:
  pose=r.assembly.assemble(np.asarray(item['q']),include_meshes=False)[1]['bucket'];center=pose[:3,:3]@m.MOUTH+pose[:3,3];side=pose[:3,1]*.3
  path.append(dict(frame=item['frame'],cursor=item['cursor'],center=center,left=center+side,right=center-side))
 paths[str(i)]=path
m.write(O/'actual-mouth-paths.json',dict(paths=paths,width_m=.6,reference_window=[120,840],sampling='At frozen-policy decisions; actual FK from recorded q, not excavated volume'))
terrain=D/('scoop-2-plan' if (D/'scoop-2-plan/terrain.npz').exists() else 'scoop-1-plan')/'terrain.npz'
with np.load(terrain) as z:np.savez(O/'observed-terrain.npz',**{k:z[k] for k in z.files})
summary['terrain_source_sha256']=m.sha(terrain);m.write(O/'results.json',summary)
print(m.encode({k:summary[k] for k in ['status','two_scoop_complete','new_scoops','historical_sum_display_percentage_points','new_sum_display_percentage_points']}),flush=True)

