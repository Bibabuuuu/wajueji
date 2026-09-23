"""Offline same-particle stress/deformation audit of common bottom deficits."""
import json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;root=out.parent.parent
base=root/'work/manual-soil-loss-replay-20260923-v1'
branch=root/'work/soil-projection-isolation-20260923-v2'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz'
q0=np.load(start)['particle_q'];cell0=np.floor(q0/.1).astype(int)
v0,n0=np.unique(cell0[q0[:,2]<.2],axis=0,return_counts=True)
def holes(q):
 v,n=np.unique(np.floor(q[q[:,2]<.2]/.1).astype(int),axis=0,return_counts=True);counts={tuple(x):y for x,y in zip(v,n)}
 return {tuple(x) for x,y in zip(v0,n0) if y>=32 and counts.get(tuple(x),0)<.25*y}
common=holes(np.load(base/'diagnostic-001329.npz')['particle_q']) & holes(np.load(branch/'candidate-001329.npz')['particle_q'])
ids=np.flatnonzero(np.array([tuple(x) in common for x in cell0]))
np.save(out/'common-hole-initial-particle-ids.npy',ids)
def quantiles(x):return np.quantile(x,[0,.5,.99,1]).tolist() if len(x) else []
def metrics(a,selected):
 q=a['particle_q'][selected];v=a['particle_qd'][selected];stress=a['particle_stress'][selected].astype(float);el=a['particle_elastic_strain'][selected].astype(float);g=a['particle_qd_grad'][selected].astype(float);jp=a['particle_Jp'][selected]
 sv=np.linalg.svd(el,compute_uv=False);det=np.linalg.det(el);pressure=np.trace(stress,axis1=1,axis2=2)/3
 return {'positions_median':np.median(q,axis=0).tolist(),'bounds':[q.min(0).tolist(),q.max(0).tolist()],'displacement':quantiles(np.linalg.norm(q-q0[selected],axis=1)),'speed':quantiles(np.linalg.norm(v,axis=1)),'stress_norm':quantiles(np.linalg.norm(stress,axis=(1,2))),'mean_normal_stress':quantiles(pressure),'at_tensile_limit':int((pressure < -249).sum()),'elastic_singular_max':quantiles(sv[:,0]),'elastic_singular_min':quantiles(sv[:,-1]),'elastic_determinant':quantiles(det),'elastic_negative_det_count':int((det<0).sum()),'elastic_singular_max_over10':int((sv[:,0]>10).sum()),'elastic_singular_max_over1000':int((sv[:,0]>1000).sum()),'gradient_norm':quantiles(np.linalg.norm(g,axis=(1,2))),'gradient_trace':quantiles(np.trace(g,axis1=1,axis2=2)),'Jp_inf':int(np.isinf(jp).sum()),'Jp_finite':quantiles(jp[np.isfinite(jp)])}
result={'cohort_definition':'initial particles in common depleted bottom10cm cells of baseline and no-late-projection endpoints','common_cells':[list(map(int,x)) for x in sorted(common)],'cohort_count':len(ids),'quantile_order':['min','median','p99','max'],'rows':[]}
paths=[('initial',0,start)]+[('baseline',f,base/f'diagnostic-{f:06}.npz') for f in [277,600,800,1000,1193,1329]]+[('no_late_projection',f,branch/f'candidate-{f:06}.npz') for f in [1193,1329]]
for tag,frame,path in paths:
 a=np.load(path);cohort=metrics(a,ids);q=a['particle_q'];el=a['particle_elastic_strain'].astype(float);norm=np.linalg.norm(el,axis=(1,2));worst=np.argsort(norm)[-5:][::-1]
 row={'tag':tag,'frame':frame,'cohort':cohort,'global_elastic_norm':quantiles(norm),'global_elastic_norm_over1000':int((norm>1000).sum()),'worst_elastic_particles':[{'id':int(i),'norm':float(norm[i]),'initial_xyz':q0[i].tolist(),'current_xyz':q[i].tolist()} for i in worst],'rock_positions':a['rigid_qpos'][:,:3].tolist()}
 result['rows'].append(row)
 print(tag,frame,{k:cohort[k] for k in ['speed','elastic_singular_max','elastic_singular_max_over1000','at_tensile_limit','Jp_inf']},flush=True)
(out/'snapshot-audit.json').write_text(json.dumps(result,indent=2)+'\n')
