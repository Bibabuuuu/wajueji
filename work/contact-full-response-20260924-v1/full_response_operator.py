"""D152 instance-local MPM closed-linkage compliance. Original rocks unchanged."""
import types
import numpy as np
import warp as wp
import warp.sparse as wps
from newton._src.solvers.implicit_mpm import rasterized_collisions as rc

@wp.kernel
def patch_bucket(node_positions:wp.array[wp.vec3],body_q:wp.array[wp.transform],collider_ids:wp.array[int],
                 bucket_collider:int,bucket_body:int,volume:float,C:wp.array[wp.mat33],
                 rows:wp.array[int],cols:wp.array[int],J:wp.array[wp.mat33],IJ:wp.array[wp.mat33]):
    i=wp.tid()
    if collider_ids[i]==bucket_collider:
        s=3*i
        W=wp.skew(wp.transform_get_translation(body_q[bucket_body])-node_positions[i])
        I=wp.identity(n=3,dtype=float)
        rows[s]=i;rows[s+1]=i;cols[s]=2*bucket_body;cols[s+1]=2*bucket_body+1
        J[s]=W;J[s+1]=I
        # J=[W,I] in original angular/linear block order. C blocks are
        # [linear-linear, linear-angular, angular-linear, angular-angular].
        IJ[s]=volume*(C[2]-C[3]@W)
        IJ[s+1]=volume*(C[0]-C[1]@W)

class FullResponseOperator:
    def __init__(self,solver,bucket_body,bucket_collider):
        self.solver=solver;self.body=bucket_body;self.collider=bucket_collider
        self.C=wp.zeros(4,dtype=wp.mat33,device=solver.model.device)
        self.original=solver._build_collider_rigidity_operator
        self.calls=0;self.ready=False
        solver._build_collider_rigidity_operator=types.MethodType(lambda owner,state_in,scratch,cell_volume:self.build(state_in,scratch,cell_volume),solver)
    def set(self,C):
        C=np.asarray(C,dtype=float)
        if C.shape!=(6,6) or not np.isfinite(C).all() or np.max(np.abs(C-C.T))>1e-10 or np.linalg.eigvalsh(C).min()<-1e-12:raise ValueError('invalid_full_response')
        blocks=np.array([C[:3,:3],C[:3,3:],C[3:,:3],C[3:,3:]],dtype=np.float32)
        self.C.assign(wp.array(blocks,dtype=wp.mat33,device=self.C.device));self.ready=True
    def build(self,state,scratch,volume):
        if not self.ready:raise RuntimeError('response_missing_for_frame')
        model=self.solver._mpm_model;collider=model.collider;n=scratch.collider_node_volume.shape[0];b=state.body_q.shape[0];offset=2*b
        rows=wp.empty(n*3,dtype=int);cols=wp.empty(n*3,dtype=int);J=wp.empty(n*3,dtype=wp.mat33);IJ=wp.empty(n*3,dtype=wp.mat33)
        wp.launch(rc.fill_collider_coupling_matrices,dim=n,inputs=[scratch.collider_position_field.dof_values,collider,state.body_q,model.collider_body_mass,model.collider_body_inv_inertia,model.model.particle_mass,volume,offset,scratch.collider_ids,rows,cols,J,IJ])
        wp.launch(patch_bucket,dim=n,inputs=[scratch.collider_position_field.dof_values,state.body_q,scratch.collider_ids,self.collider,self.body,volume,self.C,rows,cols,J,IJ])
        dofs=offset+collider.collider_particle_ids.shape[0];self.calls+=1
        return (wps.bsr_from_triplets(rows_of_blocks=n,cols_of_blocks=dofs,rows=rows,columns=cols,values=J),wps.bsr_from_triplets(rows_of_blocks=dofs,cols_of_blocks=n,rows=cols,columns=rows,values=IJ))
    def close(self):self.solver._build_collider_rigidity_operator=self.original
