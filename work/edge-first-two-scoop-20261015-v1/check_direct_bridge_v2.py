from pathlib import Path
import sys,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_edge_pair as m
from direct_reference_bridge import direct_bridge
from inplace_cache import self_check
out=W/'cpu-direct-bridge-v2';out.mkdir(exist_ok=False);start=time.monotonic();b=m.Budget();b.check()
prior=m.read(m.OUT/'report.json');prior_sha=m.sha(m.OUT/'report.json');assert prior['status']=='failed' and prior['frames']==4654 and 'original_approach32cm' in prior['failure']
source=Path(b.data['source']);source_sha=m.sha(source);r=m.CpuRuntime(source,out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
folder=m.OUT/'scoop-1-plan';p=m.read(folder/'report.json')
with np.load(folder/'approach.npz') as z:oldpath=dict(qpos=z['qpos'].copy(),velocity=z['velocity'].copy(),segments=p['segments'],total_frames=p['total_frames'])
with np.load(folder/'terrain.npz') as z:terrain={k:z[k] for k in z.files}
new=direct_bridge(r,terrain,oldpath,out/'reference-read');offset=len(oldpath['velocity'])-oldpath['segments'][-1]['frames']
cache_checks=self_check(out/'cache-format-tests')
checks=dict(old_failed_attempt_preserved=m.sha(m.OUT/'report.json')==prior_sha,
 original_source_unchanged=m.sha(source)==source_sha,earlier_approach_q_unchanged=np.array_equal(new['qpos'][:offset+1],oldpath['qpos'][:offset+1]),earlier_approach_commands_unchanged=np.array_equal(new['velocity'][:offset],oldpath['velocity'][:offset]),
 original_reference_corridor_passes=new['direct_reference_bridge']['alignment']['trusted'],
 minimum_clearance_has_positive_margin=new['direct_reference_bridge']['minimum_clearance']['bucket_observed_terrain_m']>=.325,
 original32cm_mechanism_static_joint_gates_preserved=all(new['direct_reference_bridge']['minimum_clearance'][k]>=v for k,v in [('bucket_truck_m',.04),('mechanism_truck_m',.02),('mechanism_ground_m',.02),('bucket_observed_terrain_m',.32),('mechanism_terrain_clearance_m',.02)]),
 cache_inplace_bytes_and_shapes_exact=all(cache_checks),no_new_physics=r.frames==0 and r.bridge is None)
assert all(checks.values()),checks
np.savez(out/'approach.npz',qpos=new['qpos'],velocity=new['velocity'])
m.write(out/'report.json',dict(status='cpu_direct_bridge_passed',checks=checks,cache_format_cases_passed=len(cache_checks),direct_reference_bridge=new['direct_reference_bridge'],total_frames=new['total_frames'],old_total_frames=oldpath['total_frames'],physical_steps=0,training_updates=0,wall_seconds=time.monotonic()-start,remaining_seconds=b.remaining()))
print(m.encode(m.read(out/'report.json')),flush=True)
