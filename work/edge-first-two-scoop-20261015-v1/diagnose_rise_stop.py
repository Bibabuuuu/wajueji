import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
D=m.OUT;p=m.read(D/'report.json');states=m.read(D/'display/states.json');plan=m.read(D/'scoop-2-plan/report.json');source=D/'first-returned.npz';r=m.CpuRuntime(source,W/'rise-stop-diagnostic-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);static=m.FixedRouteClearance(r)
spec=m.importlib.util.spec_from_file_location('rise_failure_stock',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');obs=m.importlib.util.module_from_spec(spec);spec.loader.exec_module(obs);terrain,observation=obs.stationary_terrain(r,r.arrays)
from planning.rear_lower_prepoint import ground_completed_terrain,sampled_surface_clearance
terrain=ground_completed_terrain(terrain)
def measure(q):
 pose=r.assembly.assemble(np.asarray(q),include_meshes=False)[1]['bucket'];points=static.samples@pose[:3,:3].T+pose[:3,3]
 return dict(minimum_bucket_z_m=float(points[:,2].min()),stock_gap_m=sampled_surface_clearance(points,terrain)['clearance_m'])
before=measure(states[-2]['mechanism_qpos']);after=measure(states[-1]['mechanism_qpos']);sourceframe=m.read(D/'first-returned.control.json')['frame'];n=p['frames'];nominal=plan['segments'][0]['frames'];delta=after['minimum_bucket_z_m']-before['minimum_bucket_z_m']
report=dict(status='failure_rule_independently_reproduced',global_frame=n,stage=p['phase'],stage_executed_frames=n-sourceframe,stage_nominal_frames=nominal,nominal_path_completed=n-sourceframe>=nominal,
 before=before,after=after,minimum_height_change_m=delta,downward_step_mm=-delta*1000,original_tolerance_mm=.001,original_rule_failed=delta< -1e-6,
 phase_switches=p.get('phase_switches',[]),selected_second=plan['selected'],source_sha256=m.sha(source),last_state_source_sha256=m.sha(D/'display/states.json'),
 new_physics_steps=0,training_updates=0,conclusion='Second dig never began. The inherited minimum-height rise guard stopped actual tracking during return rise. No fill or layout benefit can be inferred for the second scoop.')
assert report['original_rule_failed'];m.write(D/'rise-stop-diagnosis.json',report);print(m.encode(report),flush=True)
