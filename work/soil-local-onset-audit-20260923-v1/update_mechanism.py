"""Replay the recorded matrix update algebra, not a new physics model."""
import json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
a=np.load(out/'cohort-history.npz');frames=a['frames'];ids=a['ids']
F=a['particle_elastic_strain'].astype(float);u=a['material_update'].astype(float)
rows=[]
j=int(np.flatnonzero(ids==53753)[0])
for frame in range(1038,1048):
 k=int(np.flatnonzero(frames==frame)[0]);E=u[k,j,1:10].reshape(3,3);G=u[k,j,10:19].reshape(3,3);W=(G-G.T)/120
 prev=F[k-1,j];actual=F[k,j]
 pred=(u[k,j,0]*np.eye(3)+E+W)@prev
 eigen=np.linalg.eigvalsh((E+E.T)/2)
 row={'frame':frame,'particle_id':int(ids[j]),'previous_max_singular':float(np.linalg.svd(prev,compute_uv=False)[0]),'actual_max_singular':float(np.linalg.svd(actual,compute_uv=False)[0]),'elastic_increment_eigenvalues':eigen.tolist(),'linear_principal_factors_ignoring_rotation':(1+eigen).tolist(),'rotation_increment_norm':float(np.linalg.norm(W)),'relative_error_of_recorded_linear_update':float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),'position':a['particle_q'][k,j].tolist(),'speed':float(np.linalg.norm(a['particle_qd'][k,j])),'mean_normal_stress':float(np.trace(a['particle_stress'][k,j])/3),'weight':float(u[k,j,0])}
 rows.append(row)
# All upward crossings from <=2 to >2: isolate material and rotation terms.
sv=np.linalg.svd(F,compute_uv=False)[:,:,0]
mask=(sv[1:]>2)&(sv[:-1]<=2);ts,js=np.nonzero(mask);ts=ts+1
E=u[ts,js,1:10].reshape((-1,3,3));G=u[ts,js,10:19].reshape((-1,3,3));W=(G-np.swapaxes(G,1,2))/120
prev=F[ts-1,js];actual=F[ts,js]
pred=(u[ts,js,0,None,None]*np.eye(3)+E+W)@prev
err=np.linalg.norm(pred-actual,axis=(1,2))/np.linalg.norm(actual,axis=(1,2))
eig=np.linalg.eigvalsh((E+np.swapaxes(E,1,2))/2)
result={'sample_particle':53753,'sample_rows':rows,'upward_crossing_events':len(ts),'transition_elastic_min_eigen_below_minus2':int((eig[:,0]<-2).sum()),'transition_linear_reconstruction_error_max':float(err.max()),'transition_linear_reconstruction_error_p99':float(np.quantile(err,.99)),'transition_elastic_increment_norm_median':float(np.median(np.linalg.norm(E,axis=(1,2)))),'transition_rotation_increment_norm_median':float(np.median(np.linalg.norm(W,axis=(1,2)))),'interpretation':'In recorded events, a sufficiently negative compressive increment in Fnew=(I+E+W)Fprev yields factors below -1: sign reversal and magnitude growth. This is local update mechanism evidence, not a repaired full trajectory or proof of the upstream trigger of large grid strain increments.','physics_changed':False}
assert result['transition_linear_reconstruction_error_p99']<1e-4
(out/'update-mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
