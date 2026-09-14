"""Observation-driven single-soil cycle coordinator; no simulator ownership."""
from dataclasses import dataclass
import numpy as np


def _cut_inputs(terrain, bucket_width, excluded_entry_xy,
                minimum_entry_separation_m, corridor_depth_m, corridor_margin_m):
    h=np.asarray(terrain['height_m'],float);n=np.asarray(terrain['point_count'],float)
    origin=np.asarray(terrain['origin_xy_m'],float);cell=float(terrain['resolution_m'])
    if not terrain.get('valid') or h.ndim!=2 or n.shape!=h.shape or origin.shape!=(2,) or cell<=0:
        raise ValueError('invalid_cut_map')
    excluded=np.asarray(excluded_entry_xy,float)
    if excluded.size==0:excluded=np.empty((0,2))
    if (excluded.ndim!=2 or excluded.shape[1]!=2 or not np.isfinite(excluded).all()
            or not np.isfinite([bucket_width,minimum_entry_separation_m,corridor_depth_m,corridor_margin_m]).all()
            or min(bucket_width,minimum_entry_separation_m,corridor_depth_m)<=0 or corridor_margin_m<0):
        raise ValueError('invalid_cut_constraints')
    yy,xx=np.indices(h.shape)
    centers=np.stack((origin[0]+cell*(xx+.5),origin[1]+cell*(yy+.5)),axis=-1)
    return h,n,origin,cell,excluded,centers


def _cut_candidate(h,n,origin,cell,excluded,centers,ix,iy,bucket_width,
                   minimum_entry_separation_m,corridor_depth_m,corridor_margin_m):
    width=max(1,int(np.ceil(bucket_width/cell)))
    if iy<width//2 or iy>=h.shape[0]-width//2 or ix<5 or ix>=h.shape[1]-1:
        return None,'outside_patch'
    x,y=origin+cell*np.array([ix+.5,iy+.5]);xy=np.array([x,y])
    if not 5.2<=np.hypot(x,y)<=5.8 or abs(y)>.8:return None,'outside_search'
    if len(excluded) and np.linalg.norm(excluded-xy,axis=1).min()<minimum_entry_separation_m:
        return None,'history'
    patch=h[iy-width//2:iy+(width+1)//2,ix-5:ix+1]
    count=n[iy-width//2:iy+(width+1)//2,ix-5:ix+1]
    if not np.isfinite(patch).all() or (count<1).any() or patch.min()<.5:
        return None,'cut_patch'
    radial=xy/np.linalg.norm(xy);delta=centers-xy
    inward=-(delta[...,0]*radial[0]+delta[...,1]*radial[1])
    lateral=np.abs(delta[...,0]*radial[1]-delta[...,1]*radial[0])
    pad=np.sqrt(2.)*cell/2
    corridor=((inward>=-cell-pad)&(inward<=corridor_depth_m+cell+pad)&
              (lateral<=bucket_width/2+corridor_margin_m+pad))
    if not np.isfinite(h[corridor]).all() or (n[corridor]<1).any():
        return None,'unknown_corridor'
    height=float(np.quantile(patch,.5))
    score=float(np.minimum(patch,.8).sum()*cell**2)+height*.05-abs(y)*.02-abs(x-5.65)*.005
    return {'entry_xy_m':[x,y],'surface_z_m':height,'score':score,
            'corridor_cell_count':int(corridor.sum())},None


def select_cut(terrain, bucket_width=.60, *, excluded_entry_xy=(),
               minimum_entry_separation_m=.58, corridor_depth_m=1.35,
               corridor_margin_m=.10):
    """Choose an observed strip using an unvalidated development heuristic.

    Score remaining raised surface across the cut footprint. Each call uses the
    new map; no cycle-index lookup, hidden particle state or fixed three points.
    """
    h,n,origin,cell,excluded,centers=_cut_inputs(
        terrain,bucket_width,excluded_entry_xy,minimum_entry_separation_m,corridor_depth_m,corridor_margin_m)
    width=max(1,int(np.ceil(bucket_width/cell)));candidates=[]
    rejected_unknown=0;rejected_history=0
    for iy in range(width//2,h.shape[0]-width//2):
        for ix in range(5,h.shape[1]-1):
            candidate,reason=_cut_candidate(h,n,origin,cell,excluded,centers,ix,iy,bucket_width,
                minimum_entry_separation_m,corridor_depth_m,corridor_margin_m)
            if reason=='unknown_corridor':rejected_unknown+=1
            if reason=='history':rejected_history+=1
            if candidate is not None:candidates.append(candidate)
    if not candidates:raise ValueError('no_observed_reachable_cut_strip')
    # Preserve the frozen baseline's deterministic tie order: score, then the
    # larger x/y cell centre. This keeps symmetric sensor noise from flipping
    # the first cut while still allowing terrain and history to change it.
    chosen=max(candidates,key=lambda candidate:(candidate['score'],
        candidate['entry_xy_m'][0],candidate['entry_xy_m'][1],candidate['surface_z_m']))
    return {**chosen,
            'source':'fresh_observed_strip','candidate_count':len(candidates),'width_m':bucket_width,
            'corridor_depth_m':corridor_depth_m,'corridor_margin_m':corridor_margin_m,
            'minimum_entry_separation_m':minimum_entry_separation_m,
            'excluded_entry_count':len(excluded),'rejected_unknown_corridor_count':rejected_unknown,
            'rejected_history_count':rejected_history}


def validate_cut(terrain, selection, bucket_width=.60, *, excluded_entry_xy=(),
                 minimum_entry_separation_m=.58, corridor_depth_m=1.35,
                 corridor_margin_m=.10, maximum_surface_change_m=.10):
    """Recheck a planned entry against a new observation before dispatch."""
    h,n,origin,cell,excluded,centers=_cut_inputs(
        terrain,bucket_width,excluded_entry_xy,minimum_entry_separation_m,corridor_depth_m,corridor_margin_m)
    xy=np.asarray(selection.get('entry_xy_m'),float)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(maximum_surface_change_m) or maximum_surface_change_m<0:
        raise ValueError('invalid_cut_selection')
    index=np.floor((xy-origin)/cell).astype(int);ix,iy=int(index[0]),int(index[1])
    expected=origin+cell*np.array([ix+.5,iy+.5])
    if not np.allclose(xy,expected,atol=1e-8,rtol=0):raise ValueError('cut_entry_not_cell_center')
    candidate,reason=_cut_candidate(h,n,origin,cell,excluded,centers,ix,iy,bucket_width,
        minimum_entry_separation_m,corridor_depth_m,corridor_margin_m)
    if candidate is None:raise ValueError('cut_selection_invalid:'+reason)
    change=abs(candidate['surface_z_m']-float(selection['surface_z_m']))
    if change>maximum_surface_change_m:raise ValueError('cut_surface_changed')
    return {'valid':True,'surface_change_m':change,'corridor_cell_count':candidate['corridor_cell_count']}


@dataclass
class CycleConfig:
    minimum_acquired_kg: float = 20.
    minimum_delivered_kg: float = 10.
    residual_bucket_kg: float = 8.
    confirmation_s: float = .25
    stage_timeout_s: float = 35.
    # These are prototype flow checks, not 95% task/fill qualification.


class CycleController:
    STAGES=('APPROACH','CUT','DEEPEN','CURL','LIFT','TURN','ALIGN','DUMP','EMPTY_CHECK','CLEAR_TRUCK','RETURN_SWING','RETURN','COMPLETE')

    def __init__(self, goals, speed_limits, initial_truck_kg, config=None):
        self.goals={k:np.array(v,float,copy=True) for k,v in goals.items()}
        if set(self.STAGES)-{'EMPTY_CHECK','COMPLETE'} != set(self.goals):raise ValueError('incomplete_cycle_goals')
        if any(q.shape!=(19,) or not np.isfinite(q).all() for q in self.goals.values()):raise ValueError('invalid_cycle_goal')
        self.limits=np.asarray(speed_limits,float);self.config=config or CycleConfig()
        if self.limits.shape!=(4,) or not np.isfinite(self.limits).all() or (self.limits<=0).any():raise ValueError('invalid_cycle_limits')
        self.initial_truck=float(initial_truck_kg);self.index=0;self.entered=None;self.confirmed_since=None
        self.last_time=None;self.failure='';self.events=[];self.peak_bucket=0.;self.last_inventory_time=-np.inf
        self.last_bucket=0.;self.last_truck=self.initial_truck

    @property
    def stage(self):return 'FAILED' if self.failure else self.STAGES[self.index]

    def fail(self,reason):self.failure=reason;return np.zeros(4)

    def command(self, *, sim_s, q, reached, bucket_kg, truck_kg, inventory_time_s,
                perception_fresh, collision=False, manual=False, paused=False):
        if self.failure or self.stage=='COMPLETE':return np.zeros(4)
        q=np.asarray(q,float)
        if q.shape!=(19,) or not np.isfinite(q).all() or not np.isfinite([sim_s,bucket_kg,truck_kg,inventory_time_s]).all():return self.fail('invalid_observation')
        if collision:return self.fail('collision')
        if manual or paused:return self.fail('manual_or_pause')
        if not perception_fresh:return self.fail('perception_expired')
        if self.last_time is not None and sim_s<self.last_time:return self.fail('clock_reversed')
        if inventory_time_s>sim_s+1e-8 or sim_s-inventory_time_s>.3:return self.fail('inventory_expired')
        distinct=self.last_time is None or sim_s>self.last_time
        self.last_time=sim_s
        if self.entered is None:self.entered=sim_s
        if sim_s-self.entered>self.config.stage_timeout_s:return self.fail('stage_timeout:'+self.stage)
        if inventory_time_s>self.last_inventory_time:
            self.last_inventory_time=inventory_time_s;self.last_bucket=bucket_kg;self.last_truck=truck_kg
            if self.stage in ('CUT','DEEPEN','CURL','LIFT'):self.peak_bucket=max(self.peak_bucket,bucket_kg)
        condition=bool(reached)
        if self.stage=='CURL':condition=condition and self.last_bucket>=self.config.minimum_acquired_kg
        if self.stage=='LIFT' and condition and self.last_bucket<self.config.minimum_acquired_kg:return self.fail('empty_lift')
        if self.stage=='EMPTY_CHECK':
            condition=self.last_bucket<=self.config.residual_bucket_kg and self.last_truck-self.initial_truck>=self.config.minimum_delivered_kg
        if distinct:
            if condition and self.confirmed_since is None:self.confirmed_since=sim_s
            if not condition:self.confirmed_since=None
            if self.confirmed_since is not None and sim_s-self.confirmed_since>=self.config.confirmation_s:
                old=self.stage;self.index+=1;self.entered=sim_s;self.confirmed_since=None
                self.events.append({'from':old,'to':self.stage,'sim_s':sim_s,'bucket_kg':self.last_bucket,
                                    'truck_net_kg':self.last_truck-self.initial_truck})
                return np.zeros(4)  # no same-tick cascade through several stages
        if self.stage=='EMPTY_CHECK':return np.zeros(4)
        error=self.goals[self.stage][[7,9,12,15]]-q[[7,9,12,15]]
        # Feedback acts on actual q, not a timed recording or forced q assignment.
        return np.clip(2.*error,-self.limits,self.limits)
