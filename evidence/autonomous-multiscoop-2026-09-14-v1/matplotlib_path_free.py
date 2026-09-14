"""Small vectorized geometry predicate; no GUI/plot dependency."""
import numpy as np
def cavity_mask(q,p,ylim,front):
 x=q[:,0];z=q[:,2];inside=np.zeros(len(q),bool)
 for a,b in zip(p,np.roll(p,-1,axis=0)):
  if abs(b[1]-a[1])>1e-14:
   inside^=((a[1]>z)!=(b[1]>z))&(x<(b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0])
 return inside&(q[:,1]>=ylim[0])&(q[:,1]<=ylim[1])&(x>=np.interp(z,front[:,1],front[:,0],left=-10,right=-10))
