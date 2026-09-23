"""Observation kernels only: never write simulation state."""
import warp as wp
import warp.fem as fem
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, collision_sdf

@fem.integrand
def observe_update(s:fem.Sample, domain:fem.Domain, elastic:fem.Field,
                   plastic:fem.Field, velocity:fem.Field,
                   mapping:wp.array[int], volume:wp.array[float],
                   output:wp.array(dtype=float,ndim=2)):
    j=mapping[s.qp_index]
    if j>=0:
        w=s.qp_weight*fem.measure(domain,s)/volume[s.qp_index]
        e=elastic(s)
        g=fem.grad(velocity,s)
        wp.atomic_add(output,j,0,w)
        for a in range(3):
            for b in range(3):
                wp.atomic_add(output,j,1+3*a+b,w*e[a,b])
                wp.atomic_add(output,j,10+3*a+b,w*g[a,b])
        wp.atomic_add(output,j,19,w*wp.trace(plastic(s)))

@wp.kernel
def query(points:wp.array[wp.vec3], ids:wp.array[int], collider:Collider,
          body_q:wp.array[wp.transform], body_qd:wp.array[wp.spatial_vector],
          previous:wp.array[wp.transform], dt:float,
          result:wp.array(dtype=float,ndim=2), hit:wp.array[int]):
    i=wp.tid()
    sdf,n,v,cid,mid=collision_sdf(points[ids[i]],-2,collider,body_q,body_qd,previous,dt)
    result[i,0]=sdf
    result[i,1]=n[0];result[i,2]=n[1];result[i,3]=n[2]
    result[i,4]=v[0];result[i,5]=v[1];result[i,6]=v[2]
    hit[i]=cid
