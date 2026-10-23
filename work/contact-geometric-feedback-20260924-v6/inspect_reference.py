import json,numpy as np
from pathlib import Path
from contact_phase_geometry import ground_connected_mask
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP,tool_pose
root=Path('/home/qunwang/wajueji');source=root/'runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz';archive=np.load(source)
p=archive['particle_q'];p=p[ground_connected_mask(p)]
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
e=json.loads((root/'runs/contact-geometric-probe-20260924-v5/contact900/actions.ndjson').read_text().splitlines()[422]);q=np.array(e['before_state']['mechanism_qpos']);tip,pitch=tool_pose(a,q,TIP);pose=a.assemble(q,include_meshes=False)[1]['bucket'];rad=tip[:2]/np.linalg.norm(tip[:2]);lat=np.array([-rad[1],rad[0]])
angle=.322-pitch;co,si=np.cos(angle),np.sin(angle);rotation=pose[:3,:3]@np.array([[co,0,si],[0,1,0],[-si,0,co]])
offset=rotation@(np.array([.021459,0,-.157745])-TIP);xy=tip[:2]-.575*rad+offset[:2]
heights=[]
for y in np.linspace(-.27,.27,7):
 center=xy+y*lat;d=p[:,:2]-center;v=p[(abs(d@rad)<=.075)&(abs(d@lat)<=.045),2];heights.append(float(np.quantile(v,.95)) if len(v)>=8 else None)
print(dict(root_xy=xy.tolist(),upright_root_above_tip=offset[2],heights=heights,lower_quartile=float(np.quantile(heights,.25)),tip_z=float(np.quantile(heights,.25)-offset[2]-.03)))
