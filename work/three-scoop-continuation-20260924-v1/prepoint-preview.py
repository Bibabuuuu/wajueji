"""Prospective geometry only; never saves a physical checkpoint."""
import json,numpy as np
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime,REPO
from layered_47_cycle_geometry import stationary_terrain
from layered_47_cycle_selection import rank_workspace_candidates
from planning.rear_lower_prepoint import plan_rear_lower_prepoint
from run_layered_47_sparse_manual_contact_episode import _clean
r=CycleRuntime('runs/layered-47-cycle-v1/truck-hold-v1/final.npz','runs/layered-47-cycle-v1/prospective-prepoint-v1')
terrain,observation=stationary_terrain(r)
goals=json.loads(Path('runs/layered-47-cycle-v1/transport-geometry-v7/plan.json').read_text())['segments']
q=np.array(goals[-1]['goal'])
candidates=rank_workspace_candidates(terrain,2)
print('CANDIDATES',len(candidates),[(c['entry_xy_m'],c['future_entry_xy_m']) for c in candidates[:6]],flush=True)
with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:truck=data['world_vertices'].copy()
tried=[];rejections=[]
for c in candidates:
 xy=np.array(c['entry_xy_m'])
 if any(np.linalg.norm(xy-prev)<.14 for prev in tried):continue
 tried.append(xy)
 try:
  path=plan_rear_lower_prepoint(terrain=terrain,contact_selection=c['contact'],operator_qpos=q,open_bucket_proxy=r.proxy,truck_world_vertices=truck,assembly=r.assembly,chain=r.chain)
  print('CANDIDATE_PASSED',c['entry_xy_m'],path['total_frames'],flush=True)
  (r.output/'prospective-plan.json').write_text(json.dumps(_clean(dict(physics_stepped=False,future_actual_observation_required=True,selected=c,path=path,rejections=rejections)),indent=2));break
 except ValueError as error:
  rejections.append(dict(entry=c['entry_xy_m'],failure=str(error)));print('REJECT',c['entry_xy_m'],str(error),flush=True)
 if len(tried)>=16:break
