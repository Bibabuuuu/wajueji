"""Small, explicitly bounded phase-BC coordination search; no learned weights."""
import numpy as np
BASELINE=dict(boom_scale=1.,arm_scale=1.,curl_scale=1.)
CANDIDATES=[
 dict(name='boom_arm_075',boom_scale=.75,arm_scale=.75,curl_scale=1.),
 dict(name='boom_arm_050',boom_scale=.5,arm_scale=.5,curl_scale=1.),
 dict(name='boom_050',boom_scale=.5,arm_scale=1.,curl_scale=1.),
 dict(name='arm_050',boom_scale=1.,arm_scale=.5,curl_scale=1.),
 dict(name='curl_115',boom_scale=1.,arm_scale=1.,curl_scale=1.15),
 dict(name='balanced_curl_115',boom_scale=.75,arm_scale=.75,curl_scale=1.15),
]
def validate(p):
 if set(p)!=set(BASELINE):raise ValueError('search_parameter_keys')
 if not all(np.isfinite(v) for v in p.values()) or not (.5<=p['boom_scale']<=1. and .5<=p['arm_scale']<=1. and 1.<=p['curl_scale']<=1.15):raise ValueError('search_parameter_bounds')
 return p

def modify(action,phase,closure,p):
 validate(p);a=np.asarray(action,float)
 if a.shape!=(4,) or not np.isfinite(a).all() or np.any(abs(a)>1+1e-6):raise ValueError('invalid_phase_bc_action')
 if phase!='curl_load' or closure>=.95:return a.copy()
 # Apply before the existing constrained controller and projector.
 return np.clip(a*np.array([1.,p['boom_scale'],p['arm_scale'],p['curl_scale']]),-1.,1.)
