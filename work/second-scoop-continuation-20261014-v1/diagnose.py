"""Inspect inherited return pose and candidate goals; CPU/no commands executed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sys.path.insert(0,str(ROOT/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import CpuRuntime,project,FixedRouteClearance,read,write,sha
from planning.cycle_geometry import solve_pose,tool_pose,TIP
from planning.rear_lower_prepoint import ground_completed_terrain,sampled_surface_clearance
r=CpuRuntime(ROOT/'runs/continuous-single-scoop-20261014-v1/final.npz',W/'diagnostic-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
with np.load(W/'cpu-v2/terrain.npz',allow_pickle=False) as f:t={k:f[k] for k in f.files}
t=ground_completed_terrain(t);checker=FixedRouteClearance(r);rows=[]
q=project(r,r.arrays['mechanism_qpos'])
def inspect(name,q):
 pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket'];points=checker.samples@pose[:3,:3].T+pose[:3,3]
 terrain=sampled_surface_clearance(points,t)
 row=dict(name=name,tip=tool_pose(r.assembly,q,TIP)[0],terrain=terrain,static=checker.check(q),qpos=q)
 rows.append(row);print(name,row['tip'],'bucket terrain',terrain['clearance_m'],flush=True)
inspect('SOURCE_RETURNED',q)
candidates=read(W/'cpu-v2/candidates.json')['candidates']
for c in candidates[:2]:
 xy=np.array(c['entry_xy_m']);radial=xy/np.linalg.norm(xy);swing=q.copy();swing[7]=np.arctan2(xy[1],xy[0]);inspect('YAW_'+str(xy),swing)
 seed=swing
 for name,radius,height,pitch in [('EXTEND',5.2,2.7,.2),('OPEN',5.4,2.7,-.32)]:
  try:seed=solve_pose(r.assembly,r.chain,seed,np.r_[radius*radial,height],pitch,TIP);inspect(name+'_'+str(xy),seed)
  except ValueError as e:rows.append(dict(name=name,failure=str(e)))
write(W/'return-clearance-diagnosis.json',dict(status='diagnostic_only',rows=rows,physics_steps_added=0))
