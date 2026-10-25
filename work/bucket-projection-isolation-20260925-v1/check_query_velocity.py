"""CPU rigid co-motion velocity-extension diagnostic; no scene physics."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, collision_sdf
from query_point_velocity import collision_sdf_query_velocity

@wp.kernel
def query(c:Collider,q:wp.array[wp.transform],v:wp.array[wp.spatial_vector],previous:wp.array[wp.transform],out:wp.array[wp.vec3]):
    x=wp.vec3(0.,0.,.06)
    d,n,velocity,id,mat=collision_sdf(x,-2,c,q,v,previous,1./60.)
    out[0]=velocity
    d2,n2,velocity2,id2,mat2=collision_sdf_query_velocity(x,-2,c,q,v,previous,1./60.)
    out[1]=velocity2

wp.init()
b=newton.ModelBuilder()
for i in range(48):b.add_body(xform=wp.transform_identity(),mass=1.,is_kinematic=True)
b.add_shape_box(47,hx=.5,hy=.5,hz=.018,cfg=newton.ModelBuilder.ShapeConfig(density=0.,mu=.45))
m=b.finalize(device='cpu');state=m.state()
s=newton.ModelBuilder();SolverImplicitMPM.register_custom_attributes(s)
s.add_particle(pos=(0,0,.06),vel=(0,0,0),mass=1.,radius=.01)
solver=SolverImplicitMPM(s.finalize(device='cpu'),config=SolverImplicitMPM.Config());solver.setup_collider(model=m)
v=np.zeros((48,6),np.float32);v[47,4]=.6;state.body_qd.assign(v)
out=wp.zeros(2,dtype=wp.vec3,device='cpu')
wp.launch(query,dim=1,inputs=[solver._mpm_model.collider,state.body_q,state.body_qd,wp.array(dtype=wp.transform),out],device='cpu')
values=out.numpy();angle=.6/60.;expected=np.array([.06*np.sin(angle)*60,0,.06*(np.cos(angle)-1)*60])
r=dict(native=values[0].tolist(),query_extension=values[1].tolist(),co_motion_expected=expected.tolist(),native_error=float(np.linalg.norm(values[0]-expected)),candidate_error=float(np.linalg.norm(values[1]-expected)),scope='bucket-only tangent velocity extension; not physical retention evidence')
print(r,flush=True)
assert r['candidate_error']<1e-6 and r['native_error']>.02
Path(__file__).with_name('query-velocity-check.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
