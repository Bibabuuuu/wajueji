"""Installed Newton postprocess kernel on prescribed homogeneous return states.

This deliberately isolates postprocessing from the global grid/velocity solve.
Three independent velocity basis blocks supply an exact prescribed total strain.
No solver monkeypatch, source modification, contact, gravity, or scene advance.
"""
import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.fem as fem
from newton._src.solvers.implicit_mpm import rheology_solver_kernels as rk
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import stress_strain_relationship

OUT=Path(__file__).resolve().parent

@wp.kernel
def prepare(H:wp.array[wp.mat33], L:wp.array[wp.mat33], sig:wp.array[wp.mat33],
            rhs:wp.array[rk.vec6],stress:wp.array[rk.vec6],comp:wp.array[rk.mat66],
            diag:wp.array[rk.vec6],rot:wp.array[rk.mat55],yp:wp.array[rk.YieldParamVec],
            bvec:wp.array[wp.vec3],vel:wp.array[wp.vec3]):
    i=wp.tid()
    rhs[i]=fem.SymmetricTensorMapper.value_to_dof_3d(H[i])
    stress[i]=fem.SymmetricTensorMapper.value_to_dof_3d(sig[i])
    C=rk.mat66(0.0)
    for j in range(6):
        e=rk.vec6(0.0);e[j]=1.0
        tensor=fem.SymmetricTensorMapper.dof_to_value_3d(e)
        col=fem.SymmetricTensorMapper.value_to_dof_3d(stress_strain_relationship(tensor,1.0/2000000.0,0.3))
        for k in range(6):C[k,j]=col[k]
    comp[i]=C
    diag[i]=wp.get_diag(C)
    rot[i]=wp.identity(n=5,dtype=float)
    yp[i]=rk.YieldParamVec.from_values(0.68,100000.0,0.0025,12000.0,0.0,0.0)
    for j in range(3):
        b=wp.vec3(0.0);b[j]=0.5
        bvec[3*i+j]=b
        vel[3*i+j]=wp.vec3(L[i][0,j],L[i][1,j],L[i][2,j])

@wp.kernel
def decode(E:wp.array[rk.vec6],P:wp.array[rk.vec6],outE:wp.array[wp.mat33],outP:wp.array[wp.mat33]):
    i=wp.tid()
    outE[i]=fem.SymmetricTensorMapper.dof_to_value_3d(E[i])
    outP[i]=fem.SymmetricTensorMapper.dof_to_value_3d(P[i])

def main():
    wp.init(); device='cuda:0'
    a=np.load(OUT/'homogeneous-kernel-input.npz');n=len(a['names'])
    def arr(x,dtype):return wp.array(x,dtype=dtype,device=device)
    def zeros(count,dtype):return wp.zeros(count,dtype=dtype,device=device)
    rhs=zeros(n,rk.vec6);stress=zeros(n,rk.vec6);comp=zeros(n,rk.mat66)
    diag=zeros(n,rk.vec6);rot=zeros(n,rk.mat55);yp=zeros(n,rk.YieldParamVec)
    bvec=zeros(n*3,wp.vec3);vel=zeros(n*3,wp.vec3)
    wp.launch(prepare,n,inputs=[arr(a['H'],wp.mat33),arr(a['L'],wp.mat33),arr(a['stress'],wp.mat33),rhs,stress,comp,diag,rot,yp,bvec,vel],device=device)
    E=zeros(n,rk.vec6);P=zeros(n,rk.vec6)
    wp.launch(rk.postprocess_stress_and_strain,n,inputs=[
        arr(np.arange(n+1),wp.int32),arr(np.arange(n),wp.int32),comp,
        arr(3*np.arange(n+1),wp.int32),arr(np.arange(3*n),wp.int32),bvec,
        diag,rot,zeros(n,wp.float32),yp,arr(np.ones(n),wp.float32),rhs,stress,vel,E,P],device=device)
    outE=zeros(n,wp.mat33);outP=zeros(n,wp.mat33)
    wp.launch(decode,n,inputs=[E,P,outE,outP],device=device)
    e=outE.numpy().astype(float);p=outP.numpy().astype(float)
    rows=[]
    for i,name in enumerate(a['names']):
        rows.append({'name':str(name),'elastic_max_abs_error':float(np.max(np.abs(e[i]-a['expected_E'][i]))),'plastic_max_abs_error':float(np.max(np.abs(p[i]-a['expected_P'][i]))),'strain_split_max_abs_error':float(np.max(np.abs(e[i]+p[i]-a['L'][i]))),'elastic_increment':e[i].tolist(),'plastic_increment':p[i].tolist()})
    checks={'elastic_kernel_matches_oracle':all(r['elastic_max_abs_error']<2e-5 for r in rows),'plastic_kernel_matches_oracle':all(r['plastic_max_abs_error']<2e-5 for r in rows),'strain_split_matches_prescribed_total':all(r['strain_split_max_abs_error']<2e-5 for r in rows)}
    result={'scope':'Actual installed float32 CUDA postprocess kernel with prescribed exact homogeneous hydrostatic return; not global stress/velocity solver convergence','rows':rows,'checks':checks}
    (OUT/'kernel-report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));assert all(checks.values())

if __name__=='__main__':main()
