"""Read-only bucket CAD clearance audit of original demonstrations and BC."""
from pathlib import Path
import json,numpy as np
from layered_excavation_approach_planner import _load_kinematics
from contact_bc_ground_guard import BucketGroundGuard
ROOT=Path('/home/qunwang/wajueji');K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
manifest=json.loads((ROOT/'local-data/contact-bc-native-tensors-20260924-v1/manifest.json').read_text())
report=json.loads((ROOT/'runs/contact-bc-contact-window-20260924-v1/report.json').read_text())
with np.load(report['source']['path']) as f:q=f['mechanism_qpos']
geometries,_=assembly.assemble(q);local=np.concatenate([g['body_local_vertices'] for g in geometries if g['body']=='bucket'])
guard=BucketGroundGuard(assembly,q)
runs=[(e['run'],ROOT/'runs/layered-47-sparse-manual-contact-v1'/e['run']/'physics.ndjson') for e in manifest['episodes']]
runs.append(('bc600',ROOT/'runs/contact-bc-contact-window-20260924-v1/physics/frames.ndjson'))
results=[]
for name,path in runs:
 heights=[];frames=[];guard_stops=[];max_error=0.
 for line in path.open():
  row=json.loads(line);pose=assembly.assemble(np.array(row['after_state']['mechanism_qpos']),include_meshes=False)[1]['bucket']
  heights.append(float(np.min(local@pose[2,:3]+pose[2,3])));frames.append(row['frame'])
  check=guard.measure(row['after_state']['mechanism_qpos']);max_error=max(max_error,abs(check['minimum_bucket_z_m']-heights[-1]))
  if not check['safe']:guard_stops.append(row['frame'])
 z=np.array(heights);results.append(dict(run=name,frames=len(z),minimum_bucket_mesh_z_m=float(z.min()),minimum_frame=frames[int(z.argmin())],
  ground_guard_first_stop_frame=guard_stops[0] if guard_stops else None,guard_geometry_error_m=max_error,
  below_zero_frames=int(np.sum(z<0)),first_below_zero_frame=next((f for f,v in zip(frames,z) if v<0),None),
  below_2cm_frames=int(np.sum(z<=.02)),first_below_2cm_frame=next((f for f,v in zip(frames,z) if v<=.02),None)))
assert all(x['guard_geometry_error_m']<1e-12 for x in results)
assert results[-1]['ground_guard_first_stop_frame']<results[-1]['first_below_zero_frame']
result=dict(ground_reference_z_m=0.,guard_margin_m=.005,guard_validation='Read-only evaluation on recorded states; no guarded physical rerun.',bucket_geometry='actual display CAD vertices, existing assembly',results=results)
Path(__file__).with_name('demo-ground.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
