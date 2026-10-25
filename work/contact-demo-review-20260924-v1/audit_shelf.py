"""Read-only source XML and convex SAT attribution of the final shelf contact."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from layered_excavation_approach_planner import quaternion_matrix_wxyz
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
root=Path('/home/qunwang/wajueji');w=Path(__file__).resolve().parent;repo=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo');run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-235749-1040071';r=json.loads((run/'report.json').read_text());source=repo/'reports/v4/mixed-material-r3-c-bridge-preflight-2026-09-20-v1/generated-model.xml';sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==r['restore']['runtime_build']['model']['base_capacity_model_manifest']['source_model_sha256']
fixtures=[dict(g.attrib) for g in ET.parse(source).iter('geom') if g.get('name','').startswith('r1_ground_')];assert len(fixtures)==1;f=fixtures[0];c=np.fromstring(f['pos'],sep=' ');h=np.fromstring(f['size'],sep=' ');assert f['quat']=='1 0 0 0'
records=json.loads((repo/'results/layered-mound-physical-preflight-v1/scene-manifest.json').read_text())['rocks'];records=[x for x in records if x['layer_id'] in ('lower','lower_middle','middle','upper')];vertices=_cache_rock_vertices(repo,records);a=np.load(run/'failure-001498.diagnostic.npz');overlaps=[]
for idx,(record,v,q) in enumerate(zip(records,vertices,a['rigid_qpos'])):
 v=v@quaternion_matrix_wxyz(q[3:]).T+q[:3]-c
 if np.any(v.max(0)<-h) or np.any(v.min(0)>h):continue
 hull=ConvexHull(v);axes=[*np.eye(3),*hull.equations[:,:3]]
 for tri in hull.simplices:
  for i,j in [(0,1),(1,2),(2,0)]:
   for axis in np.eye(3):
    normal=np.cross(v[tri[j]]-v[tri[i]],axis);norm=np.linalg.norm(normal)
    if norm>1e-12:axes.append(normal/norm)
 gaps=[]
 for axis in axes:
  proj=v@axis;radius=abs(axis)@h;gaps.append(max(proj.min()-radius,-radius-proj.max()))
 if max(gaps)<=1e-9:overlaps.append({'rock_index':idx,'rock_id':record['rock_id'],'sat_max_separation_m':float(max(gaps)),'center_world_m':q[:3].tolist()})
result={'source_model':str(source),'source_sha_matches_recorded_run':True,'ramp_class_fixture_count':len(fixtures),'fixture':f,'fixture_world_bounds':[list(c-h),list(c+h)],'final_convex_sat_overlaps':overlaps,'physics_steps_executed':0,'finding':'legacy push-test shelf retained in capacity model; actual rock_ramp flag maps to this sole r1_ground geometry, not truck','scope':'independent geometric contact attribution, no replay or model modification'}
(w/'shelf-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
