"""Compare the preregistered late projection intervention with D-123."""
import json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent;root=out.parent.parent
base=root/'work/manual-soil-loss-replay-20260923-v1'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz'
q0=np.load(start)['particle_q'];bottom=q0[:,2]<.2
lo,hi=q0[:,:2].min(0),q0[:,:2].max(0)
v0,n0=np.unique(np.floor(q0[q0[:,2]<.2]/.1).astype(int),axis=0,return_counts=True)
result={'definitions':{'bottom_cohort':'same IDs initially z<0.2m','outside_footprint':'initial XY AABB expanded by0.1m','depleted_cells':'fixed0.1m cells belowz0.2,initial>=32particles,current<25%initial'},'count':len(q0),'bottom_count':int(bottom.sum()),'snapshots':{}}
for tag,directory,prefix in [('baseline',base,'diagnostic'),('no_late_projection',out,'candidate')]:
 result['snapshots'][tag]={}
 for frame in [1193,1329]:
  a=np.load(directory/f'{prefix}-{frame:06}.npz');q=a['particle_q'];v=a['particle_qd'];d=np.linalg.norm(q-q0,axis=1)
  outside=((q[:,:2]<lo-.1)|(q[:,:2]>hi+.1)).any(1)
  vf,nf=np.unique(np.floor(q[q[:,2]<.2]/.1).astype(int),axis=0,return_counts=True);counts={tuple(x):int(y) for x,y in zip(vf,nf)}
  depleted=sum(n>=32 and counts.get(tuple(cell),0)<.25*n for cell,n in zip(v0,n0))
  row={'particle_count':len(q),'bottom_moved_over20cm':int((bottom&(d>.2)).sum()),'bottom_outside_footprint':int((bottom&outside).sum()),'all_outside_footprint':int(outside.sum()),'bottom_displacement_median_p99_max_m':np.quantile(d[bottom],[.5,.99,1]).tolist(),'bottom_depleted_cells':int(depleted),'minimum_z_m':float(q[:,2].min()),'particles_below_minus5mm':int((q[:,2]<-.005).sum()),'particle_speed_p99_max_m_s':np.quantile(np.linalg.norm(v,axis=1),[.99,1]).tolist(),'nonfinite_fields':{k:int((~np.isfinite(a[k])).sum()) for k in a.files if a[k].dtype.kind in 'fc' and not np.isfinite(a[k]).all()},'flags_unchanged':bool(np.array_equal(a['particle_flags'],np.load(start)['particle_flags']))}
  if frame==1329:
   prev=np.load(directory/f'{prefix}-001193.npz')['particle_q'];delta=np.linalg.norm(q-prev,axis=1)
   row['last136_bottom_moved_over20cm']=int((bottom&(delta>.2)).sum())
  result['snapshots'][tag][str(frame)]=row
b=result['snapshots']['baseline']['1329'];c=result['snapshots']['no_late_projection']['1329']
result['final_ratio_candidate_over_baseline']={k:c[k]/b[k] if b[k] else None for k in ['bottom_moved_over20cm','bottom_outside_footprint','bottom_depleted_cells','last136_bottom_moved_over20cm']}
result['limits']=['No physical validity granted: existing invalid Jp retained in both branches.','Disabling final329frames does not isolate earlier projection effects.','New soft-limit fix not exercised; original applied commands used.','Counts are not calibrated volume or mass loss.','No screenshot ROI registration.','No deployment of disabled projection.']
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
