"""Fit the existing free-body interface to measured finite-horizon response."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

W=Path(__file__).resolve().parent;ROOT=W.parents[1]

def skew(r):
    x,y,z=r
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def free_body(a,B,r):
    S=skew(r)
    return np.block([[a*np.eye(3)-S@B@S,S@B],[-B@S,B]])

def fit(C):
    scale=np.linalg.norm(C)
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid_response')
    target=C/scale
    basis=[]
    for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        B=np.zeros((3,3));B[i,j]=1;B[j,i]=1;basis.append(B)
    def solve_at(r):
        A=np.stack([free_body(1,np.zeros((3,3)),r).ravel()]+[free_body(0,B,r).ravel() for B in basis],axis=1)
        beta=np.linalg.lstsq(A,target.ravel(),rcond=None)[0]
        return A@beta-target.ravel(),beta
    approx=target[:3,3:]@np.linalg.pinv(target[3:,3:]);approx=(approx-approx.T)/2
    r0=np.array([approx[2,1],approx[0,2],approx[1,0]])
    seeds=[np.zeros(3),np.clip(r0,-19,19),np.array([1.,0,0]),np.array([-1.,0,0]),np.array([0.,1,0]),np.array([0.,-1,0]),np.array([0.,0,1]),np.array([0.,0,-1])]
    candidates=[]
    for seed in seeds:
        o=least_squares(lambda r:solve_at(r)[0],seed,bounds=(-20,20),max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        residual,beta=solve_at(o.x);candidates.append((float(np.linalg.norm(residual)),o,beta))
    error,opt,beta=min(candidates,key=lambda x:x[0]);beta=beta*scale
    B=sum(b*x for b,x in zip(beta[1:],basis));model=free_body(beta[0],B,opt.x)
    norms=np.linalg.norm(C,axis=0);floor=float(norms.max()*.01)
    percolumn=np.linalg.norm(model-C,axis=0)/np.maximum(norms,floor)
    return dict(relative_frobenius_error=error,relative_column_errors=percolumn.tolist(),
        fitted_inverse_mass=float(beta[0]),fitted_mass_kg=None if beta[0]<=0 else float(1/beta[0]),
        fitted_scaled_inverse_inertia=B.tolist(),inverse_inertia_eigenvalues=np.linalg.eigvalsh(B).tolist(),
        fitted_center_offset_over_characteristic_length=opt.x.tolist(),
        physical_parameters=bool(beta[0]>0 and np.linalg.eigvalsh(B).min()>=-1e-10*max(1.,np.linalg.norm(B))),
        center_bound_hit=bool(np.any(abs(opt.x)>19.99)),fitted_matrix=model.tolist(),
        all_start_errors=[x[0] for x in candidates],
        fit_scope='Relaxed symmetric inertia and inverse mass allowed negative during fit; final physicality separately required. 8 initial centers, bounds +/-12m (20*0.6m). Not a proof of global minimum.')

def main():
    source=ROOT/'runs/contact-effective-response-20260924-v1/exact-response.json'
    raw=json.loads(source.read_text());protocol=json.loads((W/'protocol.json').read_text());admission=protocol['admission'];result={}
    for context in raw['contexts']:
        C=np.array(context['matrices']['100.0']);D=np.array(context['matrices']['200.0'])
        response=fit(C);linearity=float(np.linalg.norm(C-D)/np.linalg.norm(C))
        checks=dict(native_restore_and_guards=all(all(t['checks'].values()) for t in context['trials']),
           baseline_repeat=context['trials'][0]['states']==context['trials'][1]['states'],
           linearity=linearity<=admission['response_linearity_relative_frobenius'],
           interface_overall_error=response['relative_frobenius_error']<=admission['free_body_fit_relative_frobenius'],
           interface_each_column=max(response['relative_column_errors'])<=admission['free_body_fit_each_excited_column_relative'],
           physical_parameters=response['physical_parameters'],center_not_on_bound=not response['center_bound_hit'])
        result[context['context']]=dict(passed=all(checks.values()),checks=checks,linearity_relative_error=linearity,
           measured_matrix=C.tolist(),singular_values=np.linalg.svd(C,compute_uv=False).tolist(),
           symmetric_model_residual_lower_bound=float(np.linalg.norm(C-C.T)/2/np.linalg.norm(C)),fit=response)
    report=dict(status='interface_admitted' if all(x['passed'] for x in result.values()) else 'interface_not_admitted',contexts=result,
        CPU_probe_wall_seconds=raw['wall_seconds'],probe_count=sum(c['probe_count'] for c in raw['contexts']),
        total_native_steps=raw['native_diagnostic_steps'],full_scene_MPM_steps=0,source_unchanged=raw['inputs_unchanged'],
        limits=raw['limitations'],qualification='Local interface capability audit only. No coupling stability, soil, excavation, fill or RL qualification.')
    (W/'response-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
