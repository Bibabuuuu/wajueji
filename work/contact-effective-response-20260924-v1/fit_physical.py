"""Search physically valid free-body fits with both overall and column weights."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from fit_response import W,free_body

def physical_fit(C):
    scale=np.linalg.norm(C);T=C/scale
    norms=np.linalg.norm(T,axis=0);column_floor=max(norms)*.01
    weight=1/np.maximum(norms,column_floor)
    ev,V=np.linalg.eigh((T[3:,3:]+T[3:,3:].T)/2)
    initial_B=(V*np.maximum(ev,1e-7))@V.T
    L=np.linalg.cholesky(initial_B)
    initial=np.r_[np.log(max(np.trace(T[:3,:3])/3,1e-6)),np.log(np.diag(L)),L[1,0],L[2,0],L[2,1],0.,0.,0.]
    lower=np.r_[[-30]*4,[-100]*3,[-20]*3];upper=np.r_[[10]*4,[100]*3,[20]*3]
    def decode(x):
        L=np.array([[np.exp(x[1]),0,0],[x[4],np.exp(x[2]),0],[x[5],x[6],np.exp(x[3])]])
        B=L@L.T
        return np.exp(x[0]),B,x[7:10]
    candidates=[]
    seeds=[np.zeros(3),np.array([1.,0,0]),np.array([-1.,0,0]),np.array([0.,1,0]),np.array([0.,-1,0]),np.array([0.,0,1]),np.array([0.,0,-1])]
    for weighted in (False,True):
        for r in seeds:
            x=initial.copy();x[7:]=r
            def residual(x):
                difference=free_body(*decode(x))-T
                return (difference*(weight if weighted else 1)).ravel()
            o=least_squares(residual,x,bounds=(lower,upper),max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            a,B,r=decode(o.x);model=free_body(a,B,r);diff=model-T
            total=float(np.linalg.norm(diff));cols=np.linalg.norm(diff,axis=0)/np.maximum(norms,column_floor)
            score=max(total/.2,float(cols.max())/.3)
            candidates.append(dict(weighted=weighted,relative_frobenius_error=total,relative_column_errors=cols.tolist(),
                admission_violation_score=score,passed=bool(score<=1 and np.max(abs(r))<19.99),
                inverse_mass=float(a*scale),mass_kg=float(1/(a*scale)),scaled_inverse_inertia=(B*scale).tolist(),
                inverse_inertia_eigenvalues=(np.linalg.eigvalsh(B)*scale).tolist(),center_offset_over_length=r.tolist(),
                center_bound_hit=bool(np.max(abs(r))>=19.99),nfev=o.nfev,status=int(o.status),fitted_matrix=(model*scale).tolist()))
    return dict(best=min(candidates,key=lambda c:c['admission_violation_score']),candidates=candidates,
                search_scope='14 positive-mass/positive-inertia fits, overall and per-column objectives, 7 initial centers each. No global-infeasibility proof.')

def main():
    started=time.perf_counter();raw=json.loads((W/'response-review.json').read_text());result={}
    for name,c in raw['contexts'].items():
        result[name]=physical_fit(np.array(c['measured_matrix']))
        print(name,result[name]['best']['passed'],result[name]['best']['relative_frobenius_error'],max(result[name]['best']['relative_column_errors']),flush=True)
    report=dict(status='physical_interface_admitted' if all(c['best']['passed'] for c in result.values()) else 'physical_interface_not_admitted',contexts=result,wall_seconds=time.perf_counter()-started,physical_steps_added=0)
    (W/'physical-fit-review.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
