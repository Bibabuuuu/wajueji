"""Read-only candidate increment/guard observation, before particle update."""
import newton
import warp as wp
import warp.fem as fem
from soil_hencky_update_candidate import hencky_proposal

@fem.integrand
def observe(s:fem.Sample,domain:fem.Domain,elastic:fem.Field,velocity:fem.Field,
            dt:float,flags:wp.array[int],density:wp.array[float],volume:wp.array[float],
            previous:wp.array[wp.mat33],data:wp.array(dtype=float,ndim=2),rejection:wp.array[int]):
    i=s.qp_index
    if ~flags[i] & newton.ParticleFlags.ACTIVE or density[i]==0.0:
        return
    E=elastic(s);G=fem.grad(velocity,s);W=.5*dt*(G-wp.transpose(G))
    _proposal,code=hencky_proposal(previous[i],E,W)
    wp.atomic_max(rejection,i,code)
    U,xi,_V=wp.svd3(previous[i])
    H=U@wp.diag(wp.vec3(wp.log(xi[0]),wp.log(xi[1]),wp.log(xi[2])))@wp.transpose(U)
    target=H+.5*(E+wp.transpose(E))
    spin=wp.vec3(W[2,1],W[0,2],W[1,0]);angle=wp.length(spin)
    Q=wp.identity(n=3,dtype=float)
    if angle>0.0:Q=wp.quat_to_matrix(wp.quat_from_axis_angle(spin/angle,angle))
    target=Q@target@wp.transpose(Q)
    w=s.qp_weight*fem.measure(domain,s)/volume[i]
    wp.atomic_add(data,i,0,w)
    for a in range(3):
        for b in range(3):
            wp.atomic_add(data,i,1+3*a+b,w*E[a,b])
            wp.atomic_add(data,i,10+3*a+b,w*G[a,b])
            wp.atomic_add(data,i,19+3*a+b,w*target[a,b])

if __name__=='__main__':
    import numpy as np
    wp.init();wp.set_device('cuda:0')
    domain=fem.Cells(fem.Grid3D(res=wp.vec3i(1,1,1)))
    pic=fem.PicQuadrature(domain,wp.array([[.5,.5,.5]],dtype=wp.vec3))
    data=wp.zeros((1,28),dtype=float);rejection=wp.zeros(1,dtype=int)
    fem.interpolate(observe,at=pic,fields={'elastic':fem.UniformField(domain,wp.mat33(np.eye(3,dtype=np.float32)*.01)),
        'velocity':fem.UniformField(domain,wp.vec3(0.))},values={'dt':1/60,'flags':wp.ones(1,dtype=int),
        'density':wp.ones(1),'volume':wp.ones(1),'previous':wp.array([np.eye(3)],dtype=wp.mat33),
        'data':data,'rejection':rejection})
    a=data.numpy();np.testing.assert_allclose(a[0,0],1.,atol=1e-6)
    np.testing.assert_allclose(a[0,19:28].reshape(3,3),.01*np.eye(3),atol=1e-6)
    assert rejection.numpy()[0]==0
    print('OBSERVER_PREFLIGHT_PASS')
