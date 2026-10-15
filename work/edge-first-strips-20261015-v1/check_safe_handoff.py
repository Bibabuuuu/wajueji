import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-strips-20261015-v1'
sys.path[:0]=[str(R/'work/continuous-single-scoop-20261014-v1'),str(R/'work/second-scoop-continuation-20261014-v1')]
from continuous_scoop import CpuRuntime,read,write,FixedRouteClearance,ProgressAlignment,project
from manual_residual_sac_env import ManualResidualEnv
from planning.rear_lower_prepoint import sampled_surface_clearance,ground_completed_terrain
r=CpuRuntime(R/'runs/continuous-single-scoop-20261014-v1/safe_start.npz',W/'handoff-read-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
env=ManualResidualEnv(W/'handoff-reference',None,collect=False,waiting_action_mode='error_strength_v1')
with np.load(W/'cpu-edge-v2/edge-first-approach.npz') as f:q=f['qpos'][-1]
angle=float(q[7]-env.reference_q[0,7]);reference=env.reference_q.copy();reference[:,7]+=angle
poses=np.array([r.assembly.assemble(x,include_meshes=False)[1]['bucket'] for x in reference]);alignment=ProgressAlignment(reference,poses).observe(q,r.assembly.assemble(q,include_meshes=False)[1]['bucket'],0.)
with np.load(W/'cpu-edge-v2/terrain.npz') as f:t=ground_completed_terrain({k:f[k] for k in f.files})
c=FixedRouteClearance(r)
def gap(q):
 pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket'];return sampled_surface_clearance(c.samples@pose[:3,:3].T+pose[:3,3],t)['clearance_m']
data=dict(status='measured',selected=read(W/'cpu-edge-v2/report.json')['selected'],yaw_offset_rad=angle,alignment_at_safe_prepoint=alignment,safe_prepoint_gap_m=gap(q),nominal_reference_start_gap_m=gap(project(r,reference[0])),safe_prepoint_to_reference_axis_difference=(q-reference[0])[[7,9,12,15]],physical_steps=0)
write(W/'safe-handoff-check.json',data);print(data)
