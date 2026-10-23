import json,numpy as np
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime,REPO
from layered_47_cycle_geometry import TransportClearance
from planning.approach import sampled_approach
from planning.rear_lower_prepoint import plan_rear_lower_prepoint
from layered_47_cycle_selection import rank_workspace_candidates
from run_layered_47_sparse_manual_contact_episode import _clean
r=CycleRuntime('runs/layered-47-cycle-v1/truck-hold-v1/final.npz','runs/layered-47-cycle-v1/transfer-route-preview-v2');q=np.array(json.loads(Path('runs/layered-47-cycle-v1/transport-geometry-v7/plan.json').read_text())['segments'][-1]['goal']);goals=json.loads(Path('runs/layered-47-cycle-v1/return-transfer-goals-v1/goals.json').read_text());checker=TransportClearance(r);paths=[]
def advance(q,v,dt):r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
def check(q,prev):
 d=checker.check(q,prev)
 if d['bucket_terrain_m']<.32:raise ValueError('transfer_bucket_margin:'+str(d))
 return d
for i in [0,2]:
 goal=np.array(goals[i]['qpos']);print('SEGMENT',goals[i]['parameters'],flush=True)
 path=sampled_approach(q,goal,axes=r.chain.axes,speed_limits=np.array([.12,.0225,.0225,.0225]),advance=advance,check_pose=check,dt=1/60,validation_substeps=2,max_steps=2400)
 print('PASS',path['steps'],min(d['bucket_terrain_m'] for d in path['diagnostics']),flush=True);path.pop('diagnostics');paths.append(path);q=goal
with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:truck=data['world_vertices'].copy()
for c in rank_workspace_candidates(checker.terrain,2)[:4]:
 try:
  path=plan_rear_lower_prepoint(terrain=checker.terrain,contact_selection=c['contact'],operator_qpos=q,open_bucket_proxy=r.proxy,truck_world_vertices=truck,assembly=r.assembly,chain=r.chain,transfer_profiles=((.65,-.45),(.60,-.40),(.575,-.375)))
  print('PREPOINT PASS',c['entry_xy_m'],path['total_frames'],flush=True)
  (r.output/'plan.json').write_text(json.dumps(_clean(dict(prospective_only=True,transfer_paths=paths,selected=c,prepoint=path)),indent=2));break
 except ValueError as e:print('PREPOINT REJECT',c['entry_xy_m'],str(e),flush=True)
