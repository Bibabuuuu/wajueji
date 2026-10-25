"""Renderer-only controlled translation, no physics step or live state mutation."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import warp as wp
import newton
from soil_surface import SoilSurface
root=Path('/home/qunwang/wajueji');out=root/'work/carry-sync-curl-audit-20260923-v1'
source=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-163839-751464/final-001370.diagnostic.npz'
original_sha=hashlib.sha256(source.read_bytes()).hexdigest();points=np.load(source)['particle_q']
wp.init();wp.set_device('cuda:0')
q=wp.array(points,dtype=wp.vec3)
viewer=newton.viewer.ViewerGL(width=800,height=600,headless=True)
results=[]
try:
 for interval in (6,1):
  surface=SoilSurface(q,particle_spacing_m=.025,voxel_size_m=.04);rows=[]
  for frame in range(13):
   shifted=points+np.array([0,0,.004*frame],dtype=np.float32);q.assign(shifted)
   before=q.numpy().copy();started=time.perf_counter();surface.update(q,frame//interval)
   viewer.begin_frame(frame/60);surface.draw(viewer,q);viewer.end_frame()
   centroid=surface.marching_cubes.verts.numpy().mean(axis=0)
   rows.append({'frame':frame,'token':surface.last_update_token,'mesh_centroid':centroid.tolist(),'wall_seconds':time.perf_counter()-started,'particle_state_unchanged':bool(np.array_equal(before,q.numpy()))})
   # Repainting the same physical frame must not rebuild the surface.
   updates=len(surface.update_seconds);surface.update(q,frame//interval);assert len(surface.update_seconds)==updates
  results.append({'interval':interval,'rows':rows,'update_count':len(surface.update_seconds),'warm_update_seconds_mean':float(np.mean(surface.update_seconds[1:])),'warm_draw_plus_update_seconds_mean':float(np.mean([x['wall_seconds'] for x in rows[1:]]))})
 assert all(r['particle_state_unchanged'] for result in results for r in result['rows'])
 assert results[0]['update_count']==3 and results[1]['update_count']==13
 assert all(results[0]['rows'][i]['mesh_centroid']==results[0]['rows'][0]['mesh_centroid'] for i in range(1,6))
 assert all(results[1]['rows'][i]['mesh_centroid']!=results[1]['rows'][i-1]['mesh_centroid'] for i in range(1,13))
 report={'physics_steps':0,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==original_sha,'source_sha256':original_sha,'synthetic_shift_m_per_frame':.004,'results':results,'pass':True}
 (out/'surface-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
finally:viewer.close()
