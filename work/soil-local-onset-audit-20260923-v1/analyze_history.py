import json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
a=np.load(out/'cohort-history.npz');frames=a['frames'];ids=a['ids']
q=a['particle_q'].astype(float);v=a['particle_qd'].astype(float)
F=a['particle_elastic_strain'].astype(float);stress=a['particle_stress'].astype(float)
update=a['material_update'].astype(float);sdf=a['sdf'];collider=a['collider']
mapping=json.loads((out/'collider-map.json').read_text());body_map=np.asarray(mapping['body_index'])
speed=np.linalg.norm(v,axis=2);disp=np.linalg.norm(q-q[0],axis=2)
sv=np.linalg.svd(F,compute_uv=False)[:,:,0]
mean_stress=np.trace(stress,axis1=2,axis2=3)/3
w=update[:,:,0];E=update[:,:,1:10].reshape((*w.shape,3,3));G=update[:,:,10:19].reshape((*w.shape,3,3))
W=.5/60*(G-np.swapaxes(G,-1,-2));en=np.linalg.norm(E,axis=(2,3));wn=np.linalg.norm(W,axis=(2,3))
def first(mask):
 return np.where(mask.any(axis=0),np.argmax(mask,axis=0),-1)
def qu(x):return np.quantile(x,[0,.5,.99,1]).tolist() if len(x) else []
f2=first(sv>2);f10=first(sv>10);f1000=first(sv>1000);fm=first(disp>.05);fs=first(speed>.1)
valid=(f10>=0)&(fm>=0)
report={'frames':[int(frames[0]),int(frames[-1])],'particle_count':len(ids),'quantile_order':['min','median','p99','max'],'weight_sum':qu(w.ravel()),'weight_max_abs_error':float(np.max(np.abs(w-1))), 'elastic10_vs_5cm_motion':{'both':int(valid.sum()),'elastic_before':int((valid&(f10<fm)).sum()),'same_frame':int((valid&(f10==fm)).sum()),'motion_before':int((valid&(f10>fm)).sum()),'elastic_only':int(((f10>=0)&(fm<0)).sum())},'first_events':{},'frame_summaries':[]}
for label,f in [('elastic2',f2),('elastic10',f10),('elastic1000',f1000),('motion5cm',fm),('speed10cm_s',fs)]:
 valid=f>=0
 report['first_events'][label]={'count':int(valid.sum()),'frame_quantiles':qu(frames[f[valid]])}
early=np.argsort(np.where(f10>=0,f10,len(frames)))[:12]
events=[]
for j in early:
 t=int(f10[j])
 if t<0:continue
 window=[]
 for k in range(max(1,t-4),min(len(frames),t+2)):
  predicted=(w[k,j]*np.eye(3)+E[k,j]+W[k,j])@F[k-1,j]
  cid=int(collider[k,j]);body=int(body_map[cid]) if cid>=0 else None
  window.append({'frame':int(frames[k]),'position':q[k,j].tolist(),'speed':float(speed[k,j]),'displacement_since900':float(disp[k,j]),'singular_max':float(sv[k,j]),'elastic_delta_norm':float(en[k,j]),'rotation_delta_norm':float(wn[k,j]),'plastic_trace_delta':float(update[k,j,19]),'weight':float(w[k,j]),'mean_stress':float(mean_stress[k,j]),'nearest_sdf_m':float(sdf[k,j,0]),'nearest_collider_id':cid,'nearest_body_id':body,'nearest_velocity':sdf[k,j,4:7].tolist(),'update_reconstruction_rel_error':float(np.linalg.norm(predicted-F[k,j])/max(1,np.linalg.norm(F[k,j])))})
 events.append({'particle_id':int(ids[j]),'elastic10_frame':int(frames[t]),'motion5cm_frame':int(frames[fm[j]]) if fm[j]>=0 else None,'window':window})
report['early_elastic10_examples']=events
for k,frame in enumerate(frames):
 if frame%10 or k==0:continue
 report['frame_summaries'].append({'frame':int(frame),'elastic_max_median_max':[float(np.median(sv[k])),float(sv[k].max())],'elastic_over2':int((sv[k]>2).sum()),'elastic_over10':int((sv[k]>10).sum()),'elastic_over1000':int((sv[k]>1000).sum()),'displacement_median_p99_max':qu(disp[k])[1:],'speed_median_p99_max':qu(speed[k])[1:],'at_tensile_limit':int((mean_stress[k]<-249).sum()),'elastic_delta_norm':qu(en[k]),'rotation_delta_norm':qu(wn[k]),'Jp_inf':int(np.isinf(a['particle_Jp'][k]).sum())})
valid=f2>=0;ix=np.flatnonzero(valid);t=f2[valid];ci=collider[t,ix];bi=np.array([body_map[c] if c>=0 else -999 for c in ci]);dist=sdf[t,ix,0]
report['at_first_elastic2']={'count':len(ix),'no_hit_within_query_gap':int((ci<0).sum()),'nearest_ground':int(((ci>=0)&(bi<0)).sum()),'nearest_bucket':int((bi==mapping['bucket_body']).sum()),'nearest_rock':int(((bi>=0)&(bi<mapping['bucket_body'])).sum()),'negative_nearest_sdf':int(((ci>=0)&(dist<0)).sum()),'nearest_sdf_valid_hits':qu(dist[ci>=0]),'elastic_delta_norm':qu(en[t,ix]),'rotation_delta_norm':qu(wn[t,ix]),'plastic_trace_delta':qu(update[t,ix,19]),'speed':qu(speed[t,ix]),'displacement':qu(disp[t,ix]),'tensile_count':int((mean_stress[t,ix]<-249).sum())}
report['limits']=['Posthoc cohort selection; diagnostic not qualification.','Nearest collider is not all grid-support contacts.','Weighted mean delta reconstruction ignores per-quadrature rejected updates; valid only where measured reconstruction error is small.','Temporal ordering does not establish the original trigger by itself.']
(out/'onset-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['early_elastic10_examples','frame_summaries']},indent=2))
print('FIRST_EXAMPLE',json.dumps(events[0],indent=2))
