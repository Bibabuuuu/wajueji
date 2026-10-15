"""CPU-only stable first-episode evidence while later native phases continue."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
from live_bucket_fill import MixedFillEstimator,load_config
D=m.OUT;folder=D/'policy-1/episode-0001';p=m.read(folder/'report.json');assert p['success'] and p['status']=='completed';offset=p['actual_entry_global_frame'];wanted={offset+x['frame'] for x in p['scores']};states={}
with (D/'physics/frames.ndjson').open() as f:
 for line in f:
  row=json.loads(line)
  if row['frame'] in wanted:states[row['frame']]=row['after_state']
  if row['frame']>=max(wanted):break
r=m.CpuRuntime(p['checkpoint']['path'],W/'first-dig-analysis-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
vertices=m.manual._cache_rock_vertices(r.root,r.records);estimator=MixedFillEstimator(load_config(R/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in r.records],vertices);cloud=np.load(D/'display/particles.npy',mmap_mode='r');samples=[]
for x in p['scores']:
 g=offset+x['frame'];state=states[g];pose=r.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket'];new=estimator.measure_snapshot(dict(frame=x['frame'],soil=cloud[g],pose=pose,rocks=np.asarray([v['qpos'] for v in state['rocks']])))
 assert new['valid_estimate'] and abs(new['fill_percent']-x['fill_percent'])<=1e-9;samples.append(new)
actions=[json.loads(s) for s in (folder/'actions.ndjson').read_text().splitlines()];waits=[x for x in actions if x['execution_mode']=='wait'];runs=[]
for x in waits:
 if runs and x['frame']==runs[-1]['last_frame']+1 and x['cursor_before']==runs[-1]['cursor']:
  runs[-1]['last_frame']=x['frame'];runs[-1]['frames']+=1
 else:runs.append(dict(cursor=x['cursor_before'],first_frame=x['frame'],last_frame=x['frame'],frames=1))
result=dict(status='first_hold_values_independently_verified',retained_display_percent=min(x['fill_percent'] for x in samples),recomputed_samples=samples,physical_frames=len(actions),simulation_seconds=len(actions)/60,waiting_frames=len(waits),waiting_fraction=len(waits)/len(actions),longest_waits=sorted(runs,key=lambda x:x['frames'],reverse=True)[:8],mean_policy_wait_strength=((np.array([x['effective_action'][:4] for x in waits])+1)*.5).mean(0),source_episode_report_sha256=m.sha(folder/'report.json'),audit_new_physical_steps=0,training_updates=0,scope='First scoop only; no second-scoop or combined loading result. Historical central first used a different waiting map.')
m.write(D/'first-dig-analysis.json',result);print(m.encode({k:v for k,v in result.items() if k!='recomputed_samples'}),flush=True)
