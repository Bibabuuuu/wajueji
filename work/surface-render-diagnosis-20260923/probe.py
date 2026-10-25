import ctypes
import json
import numpy as np
import warp as wp
import newton
from soil_surface import SoilSurface
from newton._src.viewer.gl.opengl import RendererGL

wp.init()
with wp.ScopedDevice('cpu'):
    viewer = newton.viewer.ViewerGL(width=320, height=240, headless=True)
    pts = wp.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=wp.vec3)
    idx = wp.array([0,1,2,0,2,3,0,3,1,1,3,2],dtype=wp.int32)
    viewer.log_mesh('/sentinel',pts,idx)
    viewer.log_instances('/sentinel_instance','/sentinel',wp.array([wp.transform_identity()],dtype=wp.transform),None,None,None)
    gl = RendererGL.gl
    def read_buffer(handle, size):
        data = (ctypes.c_ubyte * size)()
        gl.glBindBuffer(gl.GL_COPY_READ_BUFFER, handle)
        gl.glGetBufferSubData(gl.GL_COPY_READ_BUFFER, 0, size, data)
        return bytes(data)
    mesh=viewer.objects['/sentinel']
    expected=read_buffer(mesh.vbo,mesh.vbo_size)
    expected_idx=read_buffer(mesh.ebo,mesh.ebo_size)
    coords=np.stack(np.meshgrid(np.arange(4.6,5.4,.05),np.arange(-.4,.4,.05),np.arange(.1,.7,.05)),axis=-1).reshape(-1,3).astype('f')
    q=wp.array(coords,dtype=wp.vec3)
    surface=SoilSurface(q,particle_spacing_m=.05,voxel_size_m=.1)
    for frame in range(14):
        altered=coords.copy();altered[:,2] += .01*np.sin(frame)
        q.assign(altered)
        viewer.begin_frame(frame/60)
        surface.update(q,frame)
        surface.draw(viewer,q)
        viewer.end_frame()
        print(json.dumps({'frame':frame,'verts':len(surface.marching_cubes.verts),'indices':len(surface.marching_cubes.indices),'sentinel_vertices_unchanged':read_buffer(mesh.vbo,mesh.vbo_size)==expected,'sentinel_indices_unchanged':read_buffer(mesh.ebo,mesh.ebo_size)==expected_idx}),flush=True)
    viewer.close()
