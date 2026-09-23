"""Conservative geometric and Newton-query cross-check of full outer-face ejection."""
import json
from pathlib import Path
import numpy as np
from layered_excavation_approach_planner import quaternion_matrix_wxyz
w=Path(__file__).resolve().parent;rows=json.loads((w/'event-collision-queries.json').read_text());proxy=json.loads(Path('/home/qunwang/wajueji/reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text());boxes={b['name']:b for b in proxy['shell_boxes']};strong=[]
for r in rows:
 if not(r['query_owner']==48 and r['query_material']==54+int(r['face'].split('_')[-1]) and r['query_sdf_m']<0 and 0<r['slab_outward_z_pre_m']<r['slab_half_thickness_m'] and r['slab_outward_z_post_m']>r['slab_half_thickness_m']+1e-4):continue
 b=boxes[r['face']];rot=quaternion_matrix_wxyz(b['quaternion_wxyz']);h=np.array(b['half_size_m']);a=(np.array(r['pre_local'])-b['position_bucket_m'])@rot;c=(np.array(r['post_local_same_collider_pose'])-b['position_bucket_m'])@rot
 if np.all(np.abs(a[:2])<h[:2]-1e-4) and np.all(np.abs(c[:2])<h[:2]-1e-4) and np.dot(r['bucket_gradient_local'],rot[:,2])>.999:
  strong.append(r)
d={'query_only_no_physics':True,'criterion':'initial cavity particle; first closed proxy boundary exit at shell; within 5 frames reaches actual outer face; same shell material owner; pre point strictly inside outer half slab; post point past outer face by >0.1mm; pre/post tangential coordinates within wall by >0.1mm; SDF negative and outward gradient dot >0.999','count':len(strong),'unique_particles':len(set(r['particle_id'] for r in strong)),'source_cavity_particles':6494,'examples':strong[:12]};d['share']=d['unique_particles']/6494;(w/'strict-proof.json').write_text(json.dumps(d,indent=2)+'\n');print('STRICT_OUTER_FACE_EJECTION',d['unique_particles'])
