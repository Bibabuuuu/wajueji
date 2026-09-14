"""Finite diagnostic domain: original acquisition box plus actual truck footprint.

The old y=4 boundary cuts through the live truck and bucket unloading pose.
This changes only experiment supervision, never particles, contacts or ledger.
The original lower/upper height and original diagnostic remain available.
"""
import numpy as np

class WorkEnvelope:
    def __init__(self,truck_triangles):
        tri=np.asarray(truck_triangles,float)
        if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():raise ValueError('invalid_truck_domain')
        self.truck_low=tri.min((0,1));self.truck_high=tri.max((0,1))
    def outside(self,q):
        q=np.asarray(q)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('invalid_domain_positions')
        original=(q[:,0]>=1)&(q[:,0]<=9)&(abs(q[:,1])<=4)
        truck=np.all((q[:,:2]>=self.truck_low[:2])&(q[:,:2]<=self.truck_high[:2]),axis=1)
        return ~((original|truck)&(q[:,2]>=-.05)&(q[:,2]<=5))
