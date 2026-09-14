"""Observable approach/contact evidence, not a full excavation cycle controller."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from perception.core import observation_fresh, rigid_transform


@dataclass(frozen=True)
class PhaseConfig:
    ground_z_m: float = 0.
    raised_height_m: float = .08
    min_points: int = 3
    patch_relief_m: float = .20
    near_surface_m: float = .25
    bucket_exclusion_margin_m: float = .15
    contact_enter_n: float = 150.
    contact_exit_n: float = 75.
    confirmation_samples: int = 2
    max_gap_s: float = .30
    max_wrench_age_s: float = .30
    motion_deadband_m_s: float = .02

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):raise ValueError('nonfinite_configuration')
        if not 0 <= self.contact_exit_n < self.contact_enter_n:raise ValueError('invalid_contact_hysteresis')
        if any(getattr(self,k)<=0 for k in ('raised_height_m','min_points','patch_relief_m','near_surface_m','max_gap_s','max_wrench_age_s','confirmation_samples','motion_deadband_m_s')):raise ValueError('invalid_positive_configuration')
        if self.min_points!=int(self.min_points) or self.confirmation_samples!=int(self.confirmation_samples) or self.bucket_exclusion_margin_m<0:raise ValueError('invalid_configuration')


def bucket_shell_points(contract):
    """Known inner-shell section extruded to side planes; not calibrated teeth."""
    b=contract['bucket']
    if b['coordinate_frame']!='bucket_body':raise ValueError('bucket_frame')
    xz=np.asarray(b['polygon_xz_m'],float);ys=np.asarray(b['y_limits_m'],float)
    if xz.ndim!=2 or xz.shape[1]!=2 or len(xz)<3 or ys.shape!=(2,) or not np.isfinite(xz).all() or not np.isfinite(ys).all() or ys[0]>=ys[1]:raise ValueError('invalid_bucket_section')
    return np.array([[x,y,z] for x,z in xz for y in (ys[0],ys.mean(),ys[1])])


class PhaseObserver:
    def __init__(self, shell_local, config=None):
        self.config=config or PhaseConfig()
        self.shell=np.asarray(shell_local,float).copy()
        if self.shell.ndim!=2 or self.shell.shape[1]!=3 or len(self.shell)<3 or not np.isfinite(self.shell).all():raise ValueError('invalid_shell')
        self.episode=None;self.last=None;self.motion_previous=None;self.motion_latest=None;self._clear_contact()

    def _clear_contact(self):
        self.hits=0;self.active=False;self.contact_sample=None

    def _result(self, packet, stage, reason, **fields):
        return {'schema':'excavation_phase_observation_v1','episode':packet.get('episode'),
                'source_sequence':packet.get('sequence'),'stage':stage,'reason':reason,
                'valid':stage!='WAIT_OBSERVATION','candidate':None,'command':None,
                'autonomous_task_ready':False,'contact_confirmed':False,'thresholds_calibrated':False,**fields}

    def unavailable(self, reason='observation_unavailable'):
        self._clear_contact();self.motion_previous=None
        return self._result({},'WAIT_OBSERVATION',reason)

    def _motion(self, stamp, shell):
        previous=self.motion_previous
        if previous and stamp==previous[0]:return dict(self.motion_latest)
        point=shell.mean(axis=0)
        result={'vertical_motion':'UNKNOWN','reference_velocity_world_m_s':None,
                'motion_reference':'known_inner_shell_sample_centroid','motion_interval_sim_s':None,
                'contact_does_not_imply_excavation':True}
        if previous:
            dt=stamp[1]-previous[0][1];wall_dt=stamp[2]-previous[0][2]
            if 0<dt<=self.config.max_gap_s and 0<wall_dt<=self.config.max_gap_s:
                velocity=(point-previous[1])/dt;vz=velocity[2];threshold=self.config.motion_deadband_m_s
                result.update(vertical_motion='RISING' if vz>threshold else ('LOWERING' if vz<-threshold else 'VERTICALLY_STATIONARY'),
                              reference_velocity_world_m_s=velocity.tolist(),motion_interval_sim_s=dt)
        self.motion_previous=(stamp,point);self.motion_latest=result
        return dict(result)

    def update(self, packet, now_sim_s, now_wall_s):
        try:
            if not isinstance(packet,dict):raise ValueError('packet_not_object')
            result=self._update(packet,now_sim_s,now_wall_s)
            if result['valid']:
                result['expires_monotonic_s']=packet['time']['expires_monotonic_s']
                result['expires_sim_s']=packet['time']['expires_sim_s']
                if result.get('force_n') is not None:
                    result['expires_monotonic_s']=min(result['expires_monotonic_s'],packet['load']['sample_monotonic_s']+self.config.max_wrench_age_s)
                    result['expires_sim_s']=min(result['expires_sim_s'],packet['load']['sample_sim_s']+self.config.max_wrench_age_s)
            return result
        except (KeyError,TypeError,ValueError,IndexError,OverflowError,AttributeError) as error:
            self._clear_contact();self.motion_previous=None
            return self._result(packet if isinstance(packet,dict) else {},'WAIT_OBSERVATION','invalid_observation',error=str(error))

    def _update(self,p,sim,wall):
        c=self.config
        if p.get('schema')!='excavation_observation_v1' or p.get('frame')!='world' or any(p['units'].get(k)!=v for k,v in [('length','m'),('force','N'),('time','s')]):raise ValueError('schema_frame_or_units')
        if not observation_fresh(p,sim,wall):
            self._clear_contact();self.motion_previous=None
            return self._result(p,'WAIT_OBSERVATION','observation_expired')
        episode=p['episode'];seq=p['sequence'];t=p['time'];stamp=(seq,t['capture_sim_s'],t['capture_monotonic_s'])
        if not episode or not isinstance(seq,int) or isinstance(seq,bool) or seq<0:raise ValueError('episode_or_sequence')
        if episode!=self.episode:
            self.episode=episode;self.last=None;self.motion_previous=None;self._clear_contact()
        if self.last and (seq<self.last[0] or stamp[1]<self.last[1] or stamp[2]<self.last[2] or (seq==self.last[0] and stamp!=self.last)):raise ValueError('out_of_order')
        new=self.last is None or seq>self.last[0]
        if self.last and new and (stamp[1]-self.last[1]>c.max_gap_s or stamp[2]-self.last[2]>c.max_gap_s):self._clear_contact()
        pose=rigid_transform(p['proprioception']['world_from_bucket']);shell=self.shell@pose[:3,:3].T+pose[:3,3]
        terrain=p['terrain'];height=np.asarray(terrain['height_m'],float);count=np.asarray(terrain['point_count'],float)
        shape=tuple(terrain['shape_yx']);origin=np.asarray(terrain['origin_xy_m'],float);res=float(terrain['resolution_m'])
        if height.ndim!=2 or height.shape!=shape or count.shape!=shape or origin.shape!=(2,) or not np.isfinite(origin).all() or not math.isfinite(res) or res<=0 or not np.isfinite(count).all() or (count<0).any() or (count!=np.floor(count)).any() or np.isinf(height).any():raise ValueError('invalid_terrain_grid')
        self.last=stamp
        if not terrain['valid']:
            self._clear_contact();self.motion_previous=None
            return self._result(p,'WAIT_OBSERVATION','terrain_invalid')
        motion=self._motion(stamp,shell)
        observed=np.isfinite(height)&(count>=c.min_points);raised=observed&(height>=c.ground_z_m+c.raised_height_m)
        yy,xx=np.indices(shape);xyz=np.stack((origin[0]+(xx+.5)*res,origin[1]+(yy+.5)*res,height),axis=-1)
        visible=xyz[raised]
        distance=float(np.sqrt(((visible[:,None,:]-shell[None,:,:])**2).sum(axis=2)).min()) if len(visible) else None
        patch=np.zeros(shape,bool)
        if min(shape)>=3:
            windows=np.lib.stride_tricks.sliding_window_view(raised,(3,3));hwin=np.lib.stride_tricks.sliding_window_view(height,(3,3))
            patch[1:-1,1:-1]=windows.all(axis=(-1,-2))&((hwin.max(axis=(-1,-2))-hwin.min(axis=(-1,-2)))<=c.patch_relief_m)
        low=shell.min(axis=0)-c.bucket_exclusion_margin_m;high=shell.max(axis=0)+c.bucket_exclusion_margin_m
        under_bucket=(xyz[:,:,0]>=low[0])&(xyz[:,:,0]<=high[0])&(xyz[:,:,1]>=low[1])&(xyz[:,:,1]<=high[1])
        if min(shape)>=3:patch[1:-1,1:-1]&=~np.lib.stride_tricks.sliding_window_view(under_bucket,(3,3)).any(axis=(-1,-2))
        candidates=xyz[patch];candidate=None
        if len(candidates):
            target=candidates[np.argmin(np.linalg.norm(candidates-pose[:3,3],axis=1))]
            candidate={'surface_world_m':target.tolist(),'source':'observed_raised_3x3_patch','patch_width_m':3*res,
                       'reachability_checked':False,'collision_checked':False,'material_verified':False,'excavation_target_approved':False}
        load=p['load'];force=None;sample=None
        if load.get('valid') and load.get('wrench_world') is not None:
            wrench=np.asarray(load['wrench_world'],float)
            if wrench.shape!=(6,) or not np.isfinite(wrench).all():raise ValueError('invalid_wrench')
            ss=load.get('sample_sim_s');sw=load.get('sample_monotonic_s')
            if ss is not None and sw is not None and all(math.isfinite(v) for v in (ss,sw)) and ss<=stamp[1] and sw<=stamp[2] and 0<=sim-ss<=c.max_wrench_age_s and 0<=wall-sw<=c.max_wrench_age_s:
                force=float(np.linalg.norm(wrench[:3]));sample=(ss,sw)
        near=distance is not None and distance<=c.near_surface_m
        common={'candidate':candidate,'candidate_count':len(candidates),'nearest_shell_surface_m':distance,
                'motion':motion,
                'bucket_reference':'known_inner_shell_section_proxy_not_teeth','force_n':force,'source_time':dict(t),'contact_samples':0,
                'candidate_reason':'observed_patch_found' if candidate else 'no_complete_raised_patch_outside_bucket'}
        if force is None:
            self._clear_contact();return self._result(p,'CONTACT_UNKNOWN','contact_sample_missing_or_expired',**common)
        if self.contact_sample and (sample[0]<self.contact_sample[0] or sample[1]<self.contact_sample[1]):
            self._clear_contact();return self._result(p,'CONTACT_UNKNOWN','contact_sample_out_of_order',**common)
        threshold=c.contact_exit_n if self.active else c.contact_enter_n
        if force>=threshold and not near:
            self._clear_contact();return self._result(p,'CONTACT_UNLOCALIZED','force_without_near_visible_surface',**common)
        if force>=threshold and near:
            if new and (self.contact_sample is None or (sample[0]>self.contact_sample[0] and sample[1]>self.contact_sample[1])):
                self.hits+=1;self.contact_sample=sample
            self.active=self.hits>=c.confirmation_samples
            common.update(contact_samples=self.hits,contact_confirmed=self.active)
            return self._result(p,'CONTACT_CONFIRMED' if self.active else 'CONTACT_CANDIDATE',
                                'consecutive_distinct_physics_samples' if self.active else 'awaiting_next_contact_sample',**common)
        self._clear_contact()
        return self._result(p,'NEAR_SURFACE' if near else 'ASSESS_TERRAIN',
                            'near_surface_without_confirmed_force' if near else 'survey_visible_terrain',**common)
