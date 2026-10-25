import numpy as np
import warp as wp
import warp.fem as fem
from observer import observe_update
wp.init();wp.set_device('cuda:0')
grid=fem.Grid3D(res=wp.vec3i(1,1,1))
domain=fem.Cells(grid)
points=wp.array([[.2,.2,.2],[.4,.4,.4],[.6,.6,.6],[.8,.8,.8]],dtype=wp.vec3)
pic=fem.PicQuadrature(domain,points)
mapping=wp.array([0,1,-1,2],dtype=int)
volume=wp.full(4,.25,dtype=float)
result=wp.zeros((3,20),dtype=float)
identity=wp.mat33(np.eye(3,dtype=np.float32)*.01)
fem.interpolate(observe_update,at=pic,values={'mapping':mapping,'volume':volume,'output':result},fields={'elastic':fem.UniformField(domain,identity),'plastic':fem.UniformField(domain,identity),'velocity':fem.UniformField(domain,wp.vec3(0.))})
a=result.numpy()
np.testing.assert_allclose(a[:,0],1.,rtol=0,atol=1e-6)
np.testing.assert_allclose(a[:,1:10],np.broadcast_to(np.eye(3).ravel()*.01,(3,9)),atol=1e-6)
np.testing.assert_allclose(a[:,10:19],0,atol=1e-6)
np.testing.assert_allclose(a[:,19],.03,atol=1e-6)
print('OBSERVER_KERNEL_PREFLIGHT_PASS',flush=True)
