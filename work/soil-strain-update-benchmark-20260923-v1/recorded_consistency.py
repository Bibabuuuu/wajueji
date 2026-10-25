"""Check local stress-vs-Hencky closure, not a grid solver residual."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
a=np.load(OUT.parent/'soil-local-onset-audit-20260923-v1/cohort-history.npz')
F=a['particle_elastic_strain'];u=a['material_update'];sig=a['particle_stress'];ids=a['ids'];frames=a['frames']
s=np.linalg.svd(F,compute_uv=False)[:,:,0];ts,js=np.nonzero((s[1:]>2)&(s[:-1]<=2));ts+=1

def measure(t,j):
    previous=F[t-1,j].astype(float);U,sv,V=np.linalg.svd(previous)
    H=(U*np.log(sv)[...,None,:])@np.swapaxes(U,-1,-2)
    E=u[t,j,1:10].reshape(H.shape).astype(float)
    S=sig[t,j].astype(float)
    strain_from_stress=((1+.3)*S-.3*np.trace(S,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))/2e6
    target=H+(E+np.swapaxes(E,-1,-2))/2
    return target,strain_from_stress,np.linalg.norm(target+strain_from_stress,axis=(-2,-1))

target,ss,res=measure(ts,js)
rows=[];j=int(np.flatnonzero(ids==53753)[0])
for frame in range(1039,1048):
    t=int(np.flatnonzero(frames==frame)[0]);h,c,r=measure(t,j)
    rows.append({'frame':frame,'H_target_eigenvalues':np.linalg.eigvalsh(h).tolist(),'minus_compliance_stress_eigenvalues':np.linalg.eigvalsh(-c).tolist(),'local_closure_residual_norm':float(r),'mean_normal_stress_Pa':float(np.trace(sig[t,j])/3)})
result={'scope':'Difference between updated local Hencky tensor and negative compliance times projected particle stress. Diagnostic only: not global stress-solver convergence residual; grid averaging, interpolation, stress projection and frame conventions may contribute.',
        'events':len(ts),'residual_norm_quantiles':np.quantile(res,[0,.01,.5,.99,1]).tolist(),'stress_compliance_norm_quantiles':np.quantile(np.linalg.norm(ss,axis=(1,2)),[0,.5,1]).tolist(),'sample':rows}
(OUT/'recorded-consistency.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
