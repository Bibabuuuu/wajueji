import json
from pathlib import Path
import numpy as np
from layered_excavation_approach_planner import _load_kinematics
from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import KINEMATICS
p=Path(__file__).resolve().parent/'manual-prefix-continuation-1620'
states=json.loads((p/'display/states.json').read_text());q=np.load(p/'display/particles.npy',mmap_mode='r')
a,_=_load_kinematics(str(KINEMATICS),str(KINEMATICS/'scene/input/model/A.xml'),str(KINEMATICS/'scene/input/mesh'))
c=json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text());front=np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
polygon=np.asarray(c['bucket']['polygon_xz_m']);ylimits=np.asarray(c['bucket']['y_limits_m'])
result=[];initial=None
for f in (1080,1680,1800):
 i=f-180;pose=a.assemble(np.asarray(states[i]['mechanism_qpos']),include_meshes=False)[1]['bucket'];local=(q[i]-pose[:3,3])@pose[:3,:3];mask=cavity_mask(local,polygon,ylimits,front)
 if initial is None:initial=mask.copy()
 _,counts=np.unique(np.floor(local[initial]/.02).astype(int),axis=0,return_counts=True)
 result.append(dict(frame=f,initial_cavity_particles=int(initial.sum()),initial_retained=int((mask&initial).sum()),total_in_cavity=int(mask.sum()),occupied_2cm_cells=len(counts),maximum_particles_per_cell=int(counts.max())))
(p/'retention-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
