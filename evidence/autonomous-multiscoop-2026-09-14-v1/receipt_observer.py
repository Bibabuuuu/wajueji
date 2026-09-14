"""Read-only persistent, low-speed CAD occupancy; no support or fill claim."""
import json
from pathlib import Path
import numpy as np
from truck_receiver import TruckReceiver, CAVITY, BOUNDARY, ABOVE_RIM
from matplotlib_path_free import cavity_mask

class StableWindow:
    def __init__(self,n):
        self.since=np.full(n,np.nan);self.last=None
    def update(self,now,eligible):
        if not np.isfinite(now) or (self.last is not None and now<self.last):raise ValueError('receipt_clock')
        if self.last is not None and now-self.last>.250001:self.since[:]=np.nan
        self.since[~eligible]=np.nan
        self.since[eligible & np.isnan(self.since)]=now
        self.last=now
        return eligible & (now-self.since>=5-1e-8)

class ReceiptObserver:
    def __init__(self,run,masses,contract,bucket):
        self.run=Path(run);self.mass=np.asarray(masses,float);self.receiver=TruckReceiver(contract);self.bucket=bucket
        self.window=StableWindow(len(masses));self.baseline=None;self.active_cohort=None
    def set_active_cohort(self,name):
        if name not in self.bucket.cohorts:raise ValueError('unknown_receipt_cohort')
        self.active_cohort=name
    def sample(self,q,v,pose,now,stage,mark=None):
        q=np.asarray(q);v=np.asarray(v)
        if q.shape!=(len(self.mass),3) or v.shape!=q.shape or not np.isfinite(v).all():raise ValueError('receipt_population')
        labels=self.receiver.classify(q)
        local=(q-pose[:3,3])@pose[:3,:3]
        inside=cavity_mask(local,self.bucket.poly,self.bucket.ylim,self.bucket.front)
        core=(labels==CAVITY)&~inside;speed=np.linalg.norm(v,axis=1)
        stable=self.window.update(now,core&(speed<=.05))
        if self.baseline is None:self.baseline=core.copy()
        stable_new=stable&~self.baseline;lost_initial=self.baseline&~stable
        ids=np.flatnonzero(core)
        cohort=self.bucket.cohorts.get(self.active_cohort,np.zeros(len(q),bool))
        record={'time_s':now,'stage':stage,'mark':mark,'core_kg':float(self.mass[core].sum()),
                'initial_core_kg':float(self.mass[self.baseline].sum()),'stable_kg':float(self.mass[stable].sum()),
                'stable_acquired_kg':float(self.mass[stable & cohort].sum()),'active_acquisition_cohort':self.active_cohort,'stable_new_kg':float(self.mass[stable_new].sum()),'initial_not_stable_kg':float(self.mass[lost_initial].sum()),
                'stable_net_kg':float(self.mass[stable_new].sum()-self.mass[lost_initial].sum()),
                'boundary_kg':float(self.mass[(labels==BOUNDARY)&~inside].sum()),
                'above_rim_unverified_kg':float(self.mass[(labels==ABOVE_RIM)&~inside].sum()),
                'bucket_kg':float(self.mass[inside].sum()),'core_ids':ids.tolist(),'core_speed_m_s':speed[ids].tolist(),
                'stable_ids':np.flatnonzero(stable).tolist(),'window_s':5.,'speed_limit_m_s':.05,
                'maximum_sample_gap_s':.250001,'scope':'5s persistent low-speed point-centre occupancy; support and fill uncalibrated'}
        with (self.run/'receipt.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return {k:v for k,v in record.items() if k not in ('core_ids','core_speed_m_s','stable_ids')}
