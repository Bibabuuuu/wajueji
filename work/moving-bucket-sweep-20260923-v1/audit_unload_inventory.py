"""Read-only geometric receiver occupancy; no stable cargo/volume qualification."""
from pathlib import Path
import json,sys
import numpy as np
root=Path('/home/qunwang/wajueji');w=Path(__file__).resolve().parent;run=w/'unload-180'
sys.path.insert(0,str(root/'evidence/autonomous-multiscoop-2026-09-14-v1'))
from truck_receiver import TruckReceiver
receiver=TruckReceiver(json.loads((root/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text()))
sel=np.load(run/'tracking-selection.npz');mask=sel['initial_cavity'];ids=sel['ids'][mask];q=np.load(run/'post-projection-world.npy',mmap_mode='r')[:,mask]
labels=np.stack([receiver.classify(x) for x in q]);names=['outside','cavity','boundary','above_rim','other_owner']
r=json.loads((run/'report.json').read_text());a=np.load(r['terminal']['path']);v=a['particle_qd'][ids];core=labels[-1]==1
result={'scope':'geometric receiver occupancy of original 6494 bucket soil particles; not stable received volume or rocks','source_count':len(ids),'frames':{str(f):{n:int(np.sum(labels[f]==i)) for i,n in enumerate(names)} for f in (0,60,120,150,180)},'last_30_frame_always_core':int(np.all(labels[-30:]==1,axis=0).sum()),'last_30_frame_always_core_or_boundary':int(np.all((labels[-30:]==1)|(labels[-30:]==2),axis=0).sum()),'final_core_speed_m_s_quantiles':np.quantile(np.linalg.norm(v[core],axis=1),[0,.5,.95,1]).tolist() if core.any() else None,'final_cohort_speed_m_s_quantiles':np.quantile(np.linalg.norm(v,axis=1),[0,.5,.95,1]).tolist(),'last_30_frame_cohort_max_displacement_m':float(np.max(np.linalg.norm(q[-1]-q[-30],axis=1))),'stable_cargo_qualified':False}
(w/'unload-inventory-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
