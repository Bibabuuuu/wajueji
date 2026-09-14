"""Experimental moving mesh point sweep in the tool's local frame.

Same triangle surfaces and Coulomb friction; no new adhesion, thickness or speed
clamp. The relative chord approximates rotation within one small physics step.
This does not implement a constitutive stress-transfer or fracture model.
"""
import warp as wp


@wp.kernel
def sweep_tool(mesh:wp.uint64,old_pose:wp.transform,new_pose:wp.transform,
               linear:wp.vec3,angular:wp.vec3,mu:float,
               previous:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.vec3),
               v:wp.array(dtype=wp.vec3),grad:wp.array(dtype=wp.mat33),hits:wp.array(dtype=int)):
    i=wp.tid();inv=wp.transform_inverse(new_pose)
    start=wp.transform_point(wp.transform_inverse(old_pose),previous[i])
    target=wp.transform_point(inv,q[i]);count=int(0)
    # Velocity relative to the kinematic surface at the current particle position.
    surface=linear+wp.cross(angular,q[i]-wp.transform_get_translation(new_pose))
    relative=wp.transform_vector(inv,v[i]-surface)
    for iteration in range(4):
        displacement=target-start;length=wp.length(displacement)
        if length<1.e-7:break
        direction=displacement/length
        result=wp.mesh_query_ray(mesh,start,direction,length)
        if not result.result:break
        normal=result.normal
        if wp.dot(direction,normal)>=-1.e-6:
            start=start+direction*(result.t+1.e-5)
            continue
        contact=start+direction*result.t
        rest=target-contact;normal_rest=wp.min(wp.dot(rest,normal),0.)
        tangent_rest=rest-normal*normal_rest
        vn=wp.min(wp.dot(relative,normal),0.);vt=relative-normal*vn
        target_scale=wp.max(0.,1.+mu*normal_rest/wp.max(wp.length(tangent_rest),1.e-12))
        velocity_scale=wp.max(0.,1.+mu*vn/wp.max(wp.length(vt),1.e-12))
        relative=vt*velocity_scale
        start=contact+normal*1.e-5;target=start+tangent_rest*target_scale;count+=1
    hits[i]=count
    # Avoid float roundtrips or velocity changes on untouched particles.
    if count>0:
        corrected=wp.transform_point(new_pose,target)
        q[i]=corrected
        v[i]=wp.transform_vector(new_pose,relative)+linear+wp.cross(angular,corrected-wp.transform_get_translation(new_pose))
        gradient=grad[i];grad[i]=.5*(gradient-wp.transpose(gradient))


class MovingSweep:
    def __init__(self,vertices,faces,state):
        import numpy as np
        self.mesh=wp.Mesh(points=wp.array(np.asarray(vertices,np.float32),dtype=wp.vec3,device=state.particle_q.device),
                          indices=wp.array(np.asarray(faces,np.int32).ravel(),dtype=int,device=state.particle_q.device))
        self.previous=wp.empty_like(state.particle_q)
        self.hits=wp.zeros(state.particle_q.shape[0],dtype=int,device=state.particle_q.device)

    def begin(self,state):wp.copy(self.previous,state.particle_q)

    def finish(self,state,old_pose,new_pose,linear,angular,mu=.45):
        wp.launch(sweep_tool,dim=state.particle_q.shape[0],inputs=[self.mesh.id,old_pose,new_pose,wp.vec3(*linear),wp.vec3(*angular),mu,self.previous,state.particle_q,state.particle_qd,state.mpm.particle_qd_grad,self.hits],device=state.particle_q.device)
        return int(self.hits.numpy().sum())
