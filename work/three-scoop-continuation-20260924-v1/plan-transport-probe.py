import json,numpy as np
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime
from planning.cycle_geometry import MOUTH,solve_pose,tool_pose
r=CycleRuntime('runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-010900-1088617/final-001311.diagnostic.npz','runs/layered-47-cycle-v1/geometry-probe-v2')
q=r.arrays['mechanism_qpos'];point,pitch=tool_pose(r.assembly,q,MOUTH)
print('START',point,pitch,'axes',q[r.chain.axes])
contract=json.loads(Path('evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text());contract['y_limits_m']=np.array(contract['triangles']['floor']).reshape(-1,3)[:,1][[0,-1]];tf=np.array(contract['local_to_world']);center=tf@np.array([0,np.mean(contract['y_limits_m']),0,1]);dump=tf@np.array([.6,np.mean(contract['y_limits_m']),0,1]);radial=point[:2]/np.linalg.norm(point[:2]);goals={}
for name,target,angle in [('LIFT',np.r_[4.3*radial,3.3],1),('ALIGN',np.r_[center[:2],3.5],1),('DUMP',np.r_[dump[:2],3.6],-.45),('CLEAR_TRUCK',np.r_[center[:2],3.5],1),('RETURN_HIGH',np.r_[4.3*radial,3.3],1)]:
 try:
  q=solve_pose(r.assembly,r.chain,q,target,angle,MOUTH)
  goals[name]=q.tolist();print(name,tool_pose(r.assembly,q,MOUTH),'axes',q[r.chain.axes])
 except ValueError as e: print(name,str(e));break
(r.output/'goals.json').write_text(json.dumps(goals,indent=2))
