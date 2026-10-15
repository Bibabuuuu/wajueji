"""Offline fill/gap samples from already audited cache; no physics or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';sys.path.insert(0,str(W))
import run_long_qerror as m
import numpy as np
from live_bucket_fill import MixedFillEstimator,load_config
start=time.monotonic();D=m.OUT
p=m.read(D/'policy/episode-0001/report.json')
actions=[json.loads(s) for s in (D/'policy/episode-0001/actions.ndjson').read_text().splitlines()]
states=m.read(D/'display/states.json');particles=np.load(D/'display/particles.npy',mmap_mode='r')
r=m.CpuRuntime(D/'zero.npz',W/('loading-diagnosis-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
vertices=m.manual._cache_rock_vertices(r.root,r.records)
estimator=MixedFillEstimator(load_config(R/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in r.records],vertices)
geometry=m.SolidBucketGap(r.proxy)
samples=[]
for phase in [0,120,240,360,480,600,720,840,960,1080,1140,1170]:
 row=next(x for x in actions if x['cursor_before']>=phase);f=row['frame']-1;state=states[f]
 pose=r.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
 score=estimator.measure_snapshot(dict(frame=f,soil=particles[f],pose=pose,rocks=np.asarray([x['qpos'] for x in state['rocks']])))
 gap=geometry.observe(particles[f],pose)
 samples.append(dict(frame=f,simulation_seconds=f/60,reference_cursor=row['cursor_before'],bucket_origin=state['bucket_origin_world_m'],fill_percent=score['fill_percent'],valid_estimate=score['valid_estimate'],soil_gap_m=gap['remaining_soil_gap_m']))
 print(json.dumps(samples[-1]),flush=True)
follow=[x for x in actions if x['execution_mode']=='follow'];pa=np.asarray([x['policy_action'][:4] for x in follow])
result=dict(status='offline_loading_samples_measured',source_run=str(D),source_report_sha256=m.sha(D/'report.json'),samples=samples,hold_scores=p['scores'],hold_drop_percentage_points=p['scores'][0]['fill_percent']-p['scores'][-1]['fill_percent'],normal_follow_max_abs_policy_axes=np.max(abs(pa),axis=0),normal_follow_max_residual_as_fraction_of_full_speed=np.max(abs(np.asarray([x['residual_si'] for x in follow]))/np.array([.12,.0225,.0225,.0675]),axis=0),training_updates_this_second_scoop=p['training_updates'],physical_steps_by_diagnosis=0,native_initializations_by_diagnosis=0,wall_seconds=time.monotonic()-start,limitation='Contact-time bucket occupancy can include ground-connected soil and is not retained loading. Samples cannot causally separate terrain mismatch, trajectory/retiming, digging and lift spillage without controlled evaluations. No policy or input changed.')
m.write(W/'loading-diagnosis-20261014-v1.json',result)
print('CPU_DIAGNOSIS_FINISHED',round(result['wall_seconds'],2),flush=True)
