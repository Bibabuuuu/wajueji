import ctypes
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import warp as wp
import newton
from PIL import Image
import run_layered_47_interactive_proxy_practice as display_helpers
from soil_surface import SoilSurface
from newton._src.viewer.gl.opengl import RendererGL, MeshGL, MeshInstancerGL

root=Path('/home/qunwang/wajueji')
out=root/'work/bucket-side-winding-fix-20260923-v1'
display=json.loads((root/'configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
run=out
rows=[json.loads(x) for x in (run/'physics.ndjson').read_text().splitlines()]
source=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz'
if '--final' in sys.argv:
    source=run/'final-000065.diagnostic.npz'
data=np.load(source)
print('NPZ_KEYS',list(data),flush=True)
wp.init()
with wp.ScopedDevice('cuda:0'):
    assembly,model,state,names,to_pose=display_helpers._display_runtime(display)
    viewer=newton.viewer.ViewerGL(width=1000,height=700,headless=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(8,-9,1.2),pitch=0,yaw=128)
    gl=RendererGL.gl
    def read_buffer(handle,size):
        raw=(ctypes.c_ubyte*size)()
        gl.glBindBuffer(gl.GL_COPY_READ_BUFFER,handle)
        gl.glGetBufferSubData(gl.GL_COPY_READ_BUFFER,0,size,raw)
        return hashlib.sha256(bytes(raw)).hexdigest()
    def inventory():
        return {k:(read_buffer(v.vbo,v.vbo_size),read_buffer(v.ebo,v.ebo_size)) for k,v in viewer.objects.items() if isinstance(v,MeshGL) and k!='/soil_surface'}
    qkey=next(k for k in data.files if k in ('q','particle_q','mpm_particle_q'))
    q=wp.array(data[qkey],dtype=wp.vec3)
    surface=SoilSurface(q,particle_spacing_m=.025,voxel_size_m=.04)
    baseline=None
    for frame in range(4 if '--final' in sys.argv else 18):
        row=rows[-1] if '--final' in sys.argv else rows[min(frame*4,len(rows)-1)]
        poses=assembly.assemble(np.asarray(row['after_state']['mechanism_qpos']),include_meshes=False)[1]
        state.body_q.assign(wp.array([to_pose(poses[n]) for n in names],dtype=wp.transform))
        viewer.begin_frame(frame/60)
        viewer.log_state(state)
        if frame==0: baseline=inventory()
        if frame>0:
            surface.update(q,frame)
            surface.draw(viewer,q)
        if '--final' in sys.argv and frame==2:
            viewer.objects['/soil_surface'].hidden=True
            viewer.objects['/soil_loose'].hidden=True
            viewer.log_points('/actual_particles',q,.007,wp.full(len(q),wp.vec3(.5,.3,.15),dtype=wp.vec3))
        if '--final' in sys.argv and frame==3:
            viewer.objects['/soil_surface'].hidden=False
            viewer.objects['/actual_particles'].hidden=True
            patch=data[qkey][np.load(root/'work/surface-render-diagnosis-20260923/patch-particles.npz')['ids']]
            viewer.log_points('/displaced_patch',wp.array(patch,dtype=wp.vec3),.012,wp.full(len(patch),wp.vec3(1,0,0),dtype=wp.vec3))
        viewer.end_frame()
        diffs=[k for k,v in inventory().items() if k in baseline and v!=baseline[k]]
        print(json.dumps({'frame':frame,'changed_model_buffers':diffs,'surface_vertices':len(surface.marching_cubes.verts) if frame else 0}),flush=True)
        if '--final' in sys.argv or frame in (0,1,6,17):
            pixels=viewer.get_frame().numpy()
            prefix='final' if '--final' in sys.argv else 'scene-low'
            Image.fromarray(pixels).save(out/f'{prefix}-{frame:02}.png')
    viewer.close()
