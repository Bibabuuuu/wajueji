"""Conservative sampled tool corridor; geometric buffer, not a soil failure law."""
import numpy as np


def swept_bounds(before, after, padding_m=0.):
    a=np.asarray(before,float);b=np.asarray(after,float)
    if a.ndim!=2 or a.shape[1]!=3 or not len(a) or b.shape!=a.shape or not np.isfinite([a,b]).all():
        raise ValueError('invalid_tool_vertices')
    p=float(padding_m)
    if not np.isfinite(p) or p<0:raise ValueError('invalid_tool_padding')
    return np.minimum(a.min(0),b.min(0))-p,np.maximum(a.max(0),b.max(0))+p


def solid_overlap(bank, low, high):
    lo=np.asarray(low,float);hi=np.asarray(high,float)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>=hi):
        raise ValueError('invalid_tool_bounds')
    cells=np.argwhere(bank.solid)
    cell_low=bank.origin+cells*bank.cell
    return bool(np.any(np.all(cell_low+bank.cell>lo,axis=1)&np.all(cell_low<hi,axis=1)))


def require_active_corridor(bank,before,after,padding_m=.02):
    """Refuse before advancing; caller owns activation and the physics commit."""
    low,high=swept_bounds(before,after,padding_m)
    if solid_overlap(bank,low,high):raise ValueError('tool_enters_unconverted_solid')
    return low,high
