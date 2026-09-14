"""Read-only geometric cohort diagnostics; not a ledger or calibrated fill."""
import json
from pathlib import Path
import numpy as np
from matplotlib_path_free import cavity_mask

class RetentionObserver:
 def __init__(self, run, masses, contract, front):
  self.run=Path(run);self.masses=np.array(masses,float,copy=True);self.poly=np.array(contract['bucket']['polygon_xz_m']);self.ylim=contract['bucket']['y_limits_m'];self.front=np.asarray(front)
  self.cohorts={};self.records=[]
 def derive_cohort(self,name,include_mark,exclude_mark):
  if not name or name in self.cohorts or include_mark not in self.cohorts or exclude_mark not in self.cohorts:raise ValueError('invalid_retention_cohort_derivation')
  self.cohorts[name]=self.cohorts[include_mark]&~self.cohorts[exclude_mark]
  return float(self.masses[self.cohorts[name]].sum())
 def sample(self, q, v, pose, *, time_s, stage, mark=None):
  q=np.asarray(q);v=np.asarray(v);pose=np.asarray(pose)
  if q.shape!=(len(self.masses),3) or v.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('retention_particle_state')
  local=(q-pose[:3,3])@pose[:3,:3];inside=cavity_mask(local,self.poly,self.ylim,self.front)
  lo=np.array([self.poly[:,0].min(),self.ylim[0],self.poly[:,1].min()])-.15
  hi=np.array([self.poly[:,0].max(),self.ylim[1],self.poly[:,1].max()])+.15
  near=((local>=lo)&(local<=hi)).all(1)
  if mark:self.cohorts[mark]=inside.copy()
  record={'time_s':time_s,'stage':stage,'mark':mark,'cavity_kg':float(self.masses[inside].sum()),'cohorts':{}}
  for name,mask in self.cohorts.items():
   total=float(self.masses[mask].sum());retained=float(self.masses[mask&inside].sum())
   record['cohorts'][name]={'initial_cavity_kg':total,'still_in_cavity_kg':retained,'cavity_retained_fraction':retained/total if total else None,'outside_cavity_kg':total-retained,'outside_expanded_bucket_box_kg':float(self.masses[mask&~near].sum()),'outside_box_falling_kg':float(self.masses[mask&~near&(v[:,2]<-.05)].sum())}
  self.records.append(record)
  with (self.run/'retention.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if mark:np.savez_compressed(self.run/('retention-'+mark+'.npz'),q=q,v=v,pose=pose,masses=self.masses,cavity_ids=np.flatnonzero(inside))
  return record
