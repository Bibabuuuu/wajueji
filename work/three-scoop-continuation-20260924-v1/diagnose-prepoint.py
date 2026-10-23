import json,numpy as np
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime,REPO
from layered_47_cycle_geometry import stationary_terrain
from layered_47_cycle_selection import rank_workspace_candidates
from planning.rear_lower_prepoint import plan_rear_lower_prepoint
from planning.cycle_geometry import solve_pose,TIP
r=CycleRuntime('runs/layered-47-cycle-v1/transport-return-v1/return_high.npz','runs/layered-47-cycle-v1/prepoint-diagnosis-v1')
from layered_excavation_approach_planner import build_fused_terrain
terrain,obs=build_fused_terrain(r.root,r.records,r.arrays['particle_q'],r.arrays['rigid_qpos']);candidates=rank_workspace_candidates(terrain,2);print('CANDIDATES',len(candidates),flush=True)
goals=json.loads(Path('runs/layered-47-cycle-v1/return-transfer-goals-v1/goals.json').read_text());q=np.array(goals[2]['qpos'])
with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as d:truck=d['world_vertices'].copy()
for c in [min(candidates,key=lambda c:np.linalg.norm(np.array(c['entry_xy_m'])-[5.65,-.05]))]:
 print('TRY',c['entry_xy_m'],c['contact'],flush=True)
 for h,p in [( .65,-.45),(.75,-.50),(.85,-.55)]:
  try:
   path=plan_rear_lower_prepoint(terrain=terrain,contact_selection=c['contact'],operator_qpos=q,open_bucket_proxy=r.proxy,truck_world_vertices=truck,assembly=r.assembly,chain=r.chain,transfer_profiles=((h,p),));print('PASSED',h,p,path['total_frames'],flush=True)
  except ValueError as e:print('FAIL',h,p,str(e),flush=True)
