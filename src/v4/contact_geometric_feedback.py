"""Opt-in geometry feedback before curl, and geometric correction of BC after it.

All queries are kinematic. No physics state, material, or phase labels are edited.
The original projector and measured CAD ground stop remain authoritative.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
from planning.cycle_geometry import TIP, tool_pose
from contact_bc_inference import LIMITS


@dataclass(frozen=True)
class FeedbackConfig:
    cut_drop_m: float = .30
    cut_retract_m: float = .575
    cut_pitch_rad: float = .38
    translation_speed_m_s: float = .22
    pitch_speed_rad_s: float = .30
    position_gain_s: float = 1.5
    preview_seconds: float = .15
    ground_clearance_m: float = .03
    maximum_lateral_error_m: float = .10

    def __post_init__(self):
        values=np.asarray(list(asdict(self).values()))
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('invalid_feedback_configuration')
        if self.ground_clearance_m<=.005 or self.preview_seconds>.5:
            raise ValueError('invalid_feedback_ground_preview')


def bounded_velocity(jac, target, ground_gradient, ground_lower, *, weights=(1.,4.,3.,.6), exact_pitch=False):
    """Four normalized axes: weighted velocity fit with box and ground constraints."""
    j=np.asarray(jac,float);v=np.asarray(target,float);g=np.asarray(ground_gradient,float)
    if j.shape!=(4,4) or v.shape!=(4,) or g.shape!=(4,) or not np.isfinite(np.r_[j.ravel(),v,g,ground_lower]).all():
        raise ValueError('invalid_feedback_velocity_problem')
    w=np.asarray(weights);a=w[:,None]*j;b=w*v
    h=a.T@a+1e-6*np.eye(4);f=a.T@b
    seed=np.clip(np.linalg.solve(h,f),-1,1)
    constraints=[{'type':'ineq','fun':lambda u:g@u-ground_lower,'jac':lambda u:g}]
    if exact_pitch:
        constraints.append({'type':'eq','fun':lambda u:j[3]@u-v[3],'jac':lambda u:j[3]})
    result=minimize(lambda u:.5*u@h@u-f@u,seed,jac=lambda u:h@u-f,
                    bounds=[(-1.,1.)]*4,method='SLSQP',
                    constraints=constraints,
                    options={'ftol':1e-10,'maxiter':60})
    if (not result.success or g@result.x<ground_lower-1e-7
            or (exact_pitch and abs(j[3]@result.x-v[3])>1e-7)):
        raise ValueError('feedback_velocity_infeasible:'+result.message)
    return np.clip(result.x,-1,1)


class ContactGeometryFeedback:
    def __init__(self,assembly,chain,ground_guard,goal,initial_qpos,config=FeedbackConfig()):
        self.assembly=assembly;self.chain=chain;self.ground=ground_guard;self.config=config
        self.entry=np.r_[goal['entry_xy_m'],goal['surface_z_m']].astype(float)
        if self.entry.shape!=(3,) or not np.isfinite(self.entry).all() or np.linalg.norm(self.entry[:2])<1e-6:
            raise ValueError('invalid_feedback_goal')
        self.radial=self.entry[:2]/np.linalg.norm(self.entry[:2]);self.lateral=np.array([-self.radial[1],self.radial[0]])
        self.rotation=np.array([[*self.radial,0.],[*self.lateral,0.],[0.,0.,1.]])
        self.initial=self.coordinates(initial_qpos);self.contact=None;self.cut_target=None
        self.last_phase='approach_contact';self.last_frame=0

    def coordinates(self,q):
        tip,pitch=tool_pose(self.assembly,np.asarray(q),TIP)
        return np.r_[self.rotation@tip,pitch]

    def linearize(self,q):
        """Differentiate on the closed-chain tangent, retaining the actual pose."""
        q=np.asarray(q,float)
        if q.shape!=(19,) or not np.isfinite(q).all():raise ValueError('invalid_feedback_qpos')
        addresses=list(self.chain.axes)+list(self.chain.passive);columns=[];h=1e-6
        for address in addresses:
            plus=q.copy();minus=q.copy();plus[address]+=h;minus[address]-=h
            columns.append((self.chain.residual(plus)-self.chain.residual(minus))/(2*h))
        closure=np.asarray(columns).T
        tangent=np.zeros((19,4));tangent[self.chain.axes]=np.diag(LIMITS)
        tangent[self.chain.passive]=np.linalg.lstsq(closure[:,4:],-closure[:,:4]*LIMITS,rcond=1e-10)[0]
        # The measured mechanism has nonzero (sub-mm) closure residual.
        # Permit at most 15 micrometres of tangent residual over the preview.
        if np.max(np.abs(closure@tangent[addresses]))>1e-4:raise ValueError('feedback_singular_closure')
        jac=[];ground=[];dt=1e-4
        for axis in range(4):
            plus=q+dt*tangent[:,axis];minus=q-dt*tangent[:,axis]
            difference=self.coordinates(plus)-self.coordinates(minus)
            difference[3]=np.arctan2(np.sin(difference[3]),np.cos(difference[3]))
            jac.append(difference/(2*dt))
            ground.append((self.ground.measure(plus)['clearance_m']-self.ground.measure(minus)['clearance_m'])/(2*dt))
        return np.asarray(jac).T,np.asarray(ground)

    def command(self,*,frame,qpos,phase,bc_action,projector):
        if frame!=self.last_frame+1:raise ValueError('feedback_noncontiguous_frame')
        allowed=('approach_contact','penetrate','curl_load','lift','off_soil')
        if phase not in allowed or allowed.index(phase)-allowed.index(self.last_phase) not in (0,1):
            raise ValueError('feedback_phase_order')
        q=np.asarray(qpos,float);state=self.coordinates(q);cfg=self.config
        bc=np.asarray(bc_action,float)
        if bc.shape!=(4,) or not np.isfinite(bc).all() or np.any(abs(bc)>1+1e-6):raise ValueError('invalid_feedback_bc_action')
        jac,ground_gradient=self.linearize(q)
        if phase=='penetrate' and self.contact is None:
            # Freeze the cut against the first causal confirmed contact pose.
            # Terrain erosion or later particle motion cannot deepen this target.
            self.contact=state.copy();self.cut_target=state.copy()
            self.cut_target[:3]+=np.array([-cfg.cut_retract_m,-state[1],-cfg.cut_drop_m])
            self.cut_target[3]=cfg.cut_pitch_rad
        target=None;progress=None
        if phase=='approach_contact':
            target=np.r_[self.rotation@self.entry,self.initial[3]]
            velocity=cfg.position_gain_s*(target-state)
            mode='geometric_approach'
        elif phase=='penetrate':
            target=self.cut_target.copy()
            progress=float(np.clip((self.contact[0]-state[0])/cfg.cut_retract_m,0.,1.))
            # Curl overlaps cutting, with pitch progressing by actual inward
            # motion rather than by a timer or a forced semantic phase change.
            target[3]=self.contact[3]+min(1.,progress+.15)*(cfg.cut_pitch_rad-self.contact[3])
            velocity=cfg.position_gain_s*(target-state)
            mode='geometric_penetrate'
        else:
            if self.cut_target is None:raise ValueError('feedback_missing_contact_target')
            velocity=jac@bc
            velocity[1]=-cfg.position_gain_s*state[1]
            velocity[2]=max(velocity[2],cfg.position_gain_s*(self.cut_target[2]-state[2]))
            mode='bc_with_lateral_depth_feedback'
        velocity[:3]=np.clip(velocity[:3],-cfg.translation_speed_m_s,cfg.translation_speed_m_s)
        velocity[3]=np.clip(velocity[3],-cfg.pitch_speed_rad_s,cfg.pitch_speed_rad_s)
        clearance=self.ground.measure(q)['clearance_m']
        lower=(cfg.ground_clearance_m-clearance)/cfg.preview_seconds
        normalized=bounded_velocity(jac,velocity,ground_gradient,lower,
                                    exact_pitch=phase in ('approach_contact','penetrate'))
        requested=normalized*LIMITS
        # Preserve the existing 150ms joint/closure projector, then inspect its
        # actual result; never check a different vector from the one dispatched.
        projected=projector.project(q,requested);reason=projector.reason
        accepted=None;minimum=None;preview_error=None
        for factor in (1.,.5,.25,.125,0.):
            candidate=projected*factor;chain_q=self.chain.q.copy()
            try:
                self.chain.q=q.copy();remaining=cfg.preview_seconds
                scale=max(1.,float(np.max(np.abs(candidate)/np.array([.12,.0225,.0225,.0225]))))
                minimum=clearance
                while remaining>1e-12:
                    dt=min(1/60,remaining)
                    predicted,_=projector._advance_preview(candidate/scale,dt*scale)
                    minimum=min(minimum,self.ground.measure(predicted)['clearance_m'])
                    remaining-=dt
                if minimum>=cfg.ground_clearance_m-1e-6:
                    accepted=candidate;break
            except ValueError as error:preview_error=str(error)
            finally:self.chain.q=chain_q
        if accepted is None:raise ValueError('feedback_ground_preview_infeasible:'+str(preview_error))
        if abs(state[1])>cfg.maximum_lateral_error_m:raise ValueError('feedback_lateral_corridor_exceeded')
        self.last_frame=frame;self.last_phase=phase
        return requested,accepted,dict(mode=mode,coordinates=state.tolist(),target=None if target is None else target.tolist(),
            cut_target=None if self.cut_target is None else self.cut_target.tolist(),contact=None if self.contact is None else self.contact.tolist(),
            progress=progress,desired_task_velocity=velocity.tolist(),predicted_task_velocity=(jac@(accepted/LIMITS)).tolist(),
            normalized_requested=normalized.tolist(),preview_minimum_clearance_m=minimum,preview_scale=factor,
            original_projection_reason=reason,ground_linear_lower=lower)
