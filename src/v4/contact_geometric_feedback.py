"""Opt-in geometry feedback before curl, and geometric correction of BC after it.

All queries are kinematic. No physics state, material, or phase labels are edited.
The original projector and measured CAD ground stop remain authoritative.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize
from planning.cycle_geometry import TIP, tool_pose
from contact_bc_inference import LIMITS
from itertools import product


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
    terrain_relative_cut: bool = False
    root_embed_m: float = .03
    late_curl_no_down: bool = False
    late_curl_fraction: float = .8

    def __post_init__(self):
        values=np.asarray([v for k,v in asdict(self).items() if k not in ("terrain_relative_cut","late_curl_no_down")])
        if any(type(v) is not bool for v in (self.terrain_relative_cut,self.late_curl_no_down)):raise ValueError("invalid_feedback_configuration")
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('invalid_feedback_configuration')
        if self.ground_clearance_m<=.005 or self.preview_seconds>.5:
            raise ValueError('invalid_feedback_ground_preview')
        if self.late_curl_fraction>1:raise ValueError('invalid_late_curl_fraction')


def bounded_velocity(jac, target, ground_gradient, ground_lower, *, weights=(1.,4.,3.,.6), exact_pitch=False, depth_lower=None, pitch_upper=None, no_down_gradient=None, axis_bounds=None):
    """Four normalized axes: weighted velocity fit with box and ground constraints."""
    j=np.asarray(jac,float);v=np.asarray(target,float);g=np.asarray(ground_gradient,float)
    if j.shape!=(4,4) or v.shape!=(4,) or g.shape!=(4,) or not np.isfinite(np.r_[j.ravel(),v,g,ground_lower]).all():
        raise ValueError('invalid_feedback_velocity_problem')
    w=np.asarray(weights);a=w[:,None]*j;b=w*v
    h=a.T@a+1e-6*np.eye(4);f=a.T@b
    bounds=np.asarray([(-1.,1.)]*4 if axis_bounds is None else axis_bounds,float)
    if (bounds.shape!=(4,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>bounds[:,1])
            or np.any(abs(bounds)>1)):raise ValueError('invalid_feedback_axis_bounds')
    seed=np.clip(np.linalg.solve(h,f),bounds[:,0],bounds[:,1])
    constraints=[{'type':'ineq','fun':lambda u:g@u-ground_lower,'jac':lambda u:g}]
    if exact_pitch:
        constraints.append({'type':'eq','fun':lambda u:j[3]@u-v[3],'jac':lambda u:j[3]})
    if depth_lower is not None:
        if not np.isfinite(depth_lower):raise ValueError('invalid_feedback_depth_velocity')
        constraints.append({'type':'ineq','fun':lambda u:j[2]@u-depth_lower,'jac':lambda u:j[2]})
    if pitch_upper is not None:
        if not np.isfinite(pitch_upper):raise ValueError('invalid_feedback_pitch_upper')
        constraints.append({'type':'ineq','fun':lambda u:pitch_upper-j[3]@u,'jac':lambda u:-j[3]})
    if no_down_gradient is not None:
        nd=np.asarray(no_down_gradient,float)
        if nd.ndim!=2 or nd.shape[1]!=4 or not len(nd) or not np.isfinite(nd).all():raise ValueError('invalid_no_down_gradient')
        constraints.append({'type':'ineq','fun':lambda u:nd@u,'jac':lambda u:nd})
    result=minimize(lambda u:.5*u@h@u-f@u,seed,jac=lambda u:h@u-f,
                    bounds=bounds,method='SLSQP',
                    constraints=constraints,
                    options={'ftol':1e-10,'maxiter':60})
    if (not result.success or g@result.x<ground_lower-1e-7
            or (exact_pitch and abs(j[3]@result.x-v[3])>1e-7)
            or (depth_lower is not None and j[2]@result.x<depth_lower-1e-7)
            or (pitch_upper is not None and j[3]@result.x>pitch_upper+1e-7)
            or (no_down_gradient is not None and np.min(nd@result.x)<-1e-7)):
        raise ValueError('feedback_velocity_infeasible:'+result.message)
    return np.clip(result.x,-1,1)


class ContactGeometryFeedback:
    def __init__(self,assembly,chain,ground_guard,goal,initial_qpos,config=FeedbackConfig(),cut_reference=None):
        self.assembly=assembly;self.chain=chain;self.ground=ground_guard;self.config=config
        self.entry=np.r_[goal['entry_xy_m'],goal['surface_z_m']].astype(float)
        if self.entry.shape!=(3,) or not np.isfinite(self.entry).all() or np.linalg.norm(self.entry[:2])<1e-6:
            raise ValueError('invalid_feedback_goal')
        self.radial=self.entry[:2]/np.linalg.norm(self.entry[:2]);self.lateral=np.array([-self.radial[1],self.radial[0]])
        self.rotation=np.array([[*self.radial,0.],[*self.lateral,0.],[0.,0.,1.]])
        self.initial=self.coordinates(initial_qpos);self.contact=None;self.cut_target=None
        self.last_phase='approach_contact';self.last_frame=0
        self.cut_reference=cut_reference;self.soil_plan=None;self.root_ready=False;self.root_ready_count=0
        self.no_down_latched=False;self.no_down_frame=None
        # Every CAD vertex lies in this local box. Rigid-point vertical
        # velocity is affine in local position: eight corners bound all of it.
        self.body_corners=np.asarray(list(product(*zip(ground_guard.vertices.min(axis=0),ground_guard.vertices.max(axis=0))))) if config.late_curl_no_down else None
        if config.terrain_relative_cut and cut_reference is None:raise ValueError('missing_cut_reference')

    def coordinates(self,q):
        tip,pitch=tool_pose(self.assembly,np.asarray(q),TIP)
        return np.r_[self.rotation@tip,pitch]

    def body_heights(self,q):
        pose=self.assembly.assemble(q,include_meshes=False)[1]['bucket']
        return self.body_corners@pose[2,:3]+pose[2,3]

    def linearize(self,q,*,include_body=False):
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
        jac=[];ground=[];body=[];dt=1e-4
        for axis in range(4):
            plus=q+dt*tangent[:,axis];minus=q-dt*tangent[:,axis]
            difference=self.coordinates(plus)-self.coordinates(minus)
            difference[3]=np.arctan2(np.sin(difference[3]),np.cos(difference[3]))
            jac.append(difference/(2*dt))
            ground.append((self.ground.measure(plus)['clearance_m']-self.ground.measure(minus)['clearance_m'])/(2*dt))
            if include_body:body.append((self.body_heights(plus)-self.body_heights(minus))/(2*dt))
        if include_body:return np.asarray(jac).T,np.asarray(ground),np.asarray(body).T
        return np.asarray(jac).T,np.asarray(ground)

    def command(self,*,frame,qpos,phase,bc_action,projector):
        if frame!=self.last_frame+1:raise ValueError('feedback_noncontiguous_frame')
        allowed=('approach_contact','penetrate','curl_load','lift','off_soil')
        if phase not in allowed or allowed.index(phase)-allowed.index(self.last_phase) not in (0,1):
            raise ValueError('feedback_phase_order')
        q=np.asarray(qpos,float);state=self.coordinates(q);cfg=self.config
        bc=np.asarray(bc_action,float)
        if bc.shape!=(4,) or not np.isfinite(bc).all() or np.any(abs(bc)>1+1e-6):raise ValueError('invalid_feedback_bc_action')
        closure_fraction=None;no_down_gradient=None;axis_bounds=None
        if cfg.late_curl_no_down:
            from run_layered_47_sparse_manual_contact_episode import BUCKET_INTERNAL_UPPER_TARGET_M, KINEMATIC_JOINT_MARGIN_M, COMMAND_PREVIEW_SECONDS
            low,high=self.chain.limits[self.chain.axes[3]]
            upper=min(BUCKET_INTERNAL_UPPER_TARGET_M,high-KINEMATIC_JOINT_MARGIN_M)
            closure_fraction=float((q[self.chain.axes[3]]-low)/(upper-low))
            if phase in ('curl_load','lift','off_soil') and closure_fraction>=cfg.late_curl_fraction and not self.no_down_latched:
                self.no_down_latched=True;self.no_down_frame=frame
        if self.no_down_latched:
            jac,ground_gradient,no_down_gradient=self.linearize(q,include_body=True)
            # Include the original bucket endpoint cap in the coupled solve.
            # Clipping curl afterwards alone can break vertical cancellation.
            bucket_max=float(np.clip((upper-q[self.chain.axes[3]])/COMMAND_PREVIEW_SECONDS/LIMITS[3],0.,1.))
            axis_bounds=[(-1.,1.),(0.,1.),(-1.,1.),(-1.,bucket_max)]
        else:jac,ground_gradient=self.linearize(q)
        if phase=='penetrate' and self.contact is None:
            # Freeze the cut against the first causal confirmed contact pose.
            # Terrain erosion or later particle motion cannot deepen this target.
            self.contact=state.copy();self.cut_target=state.copy()
            self.cut_target[:3]+=np.array([-cfg.cut_retract_m,-state[1],-cfg.cut_drop_m])
            self.cut_target[3]=cfg.cut_pitch_rad
            if cfg.terrain_relative_cut:
                pose=self.assembly.assemble(q,include_meshes=False)[1]['bucket']
                tip,_=tool_pose(self.assembly,q,TIP)
                self.soil_plan=self.cut_reference.plan(pose,tip,state[3],self.radial,cfg.cut_retract_m,cfg.root_embed_m)
                self.cut_target[2]=self.soil_plan['tip_target_z_m']
                if self.cut_target[2]>=state[2]-.02:raise ValueError('cut_reference_no_downward_cut')
        target=None;progress=None;root_excess=None;pitch_upper=None
        if self.soil_plan is not None:
            pose=self.assembly.assemble(q,include_meshes=False)[1]['bucket']
            root=pose[:3,:3]@np.asarray(self.soil_plan['root_local_m'])+pose[:3,3]
            root_excess=float(root[2]-self.soil_plan['surface_z_m'])
            self.root_ready_count=self.root_ready_count+1 if root_excess<=0 else 0
            self.root_ready=self.root_ready or self.root_ready_count>=3
        if phase=='approach_contact':
            target=np.r_[self.rotation@self.entry,self.initial[3]]
            velocity=cfg.position_gain_s*(target-state)
            mode='geometric_approach'
        elif phase=='penetrate':
            target=self.cut_target.copy()
            progress=float(np.clip((self.contact[0]-state[0])/cfg.cut_retract_m,0.,1.))
            # Curl overlaps cutting, with pitch progressing by actual inward
            # motion rather than by a timer or a forced semantic phase change.
            if self.soil_plan is None:
                target[3]=self.contact[3]+min(1.,progress+.15)*(cfg.cut_pitch_rad-self.contact[3])
            else:
                depth_progress=float(np.clip((self.contact[2]-state[2])/(self.contact[2]-self.cut_target[2]),0.,1.))
                target[3]=self.contact[3]+min(progress,depth_progress)*(cfg.cut_pitch_rad-self.contact[3])
                if not self.root_ready:
                    # Permit overlapping curl, but approach vertical only as
                    # the root reaches the pre-cut local soil surface.
                    pitch_ceiling=max(self.contact[3],self.soil_plan['vertical_pitch_rad']-max(0.,root_excess))
                    pitch_upper=cfg.position_gain_s*(pitch_ceiling-state[3])
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
        depth_lower=None if self.cut_target is None else cfg.position_gain_s*(self.cut_target[2]-state[2])
        normalized=bounded_velocity(jac,velocity,ground_gradient,lower,
                                    exact_pitch=phase=='approach_contact',depth_lower=depth_lower,pitch_upper=pitch_upper,
                                    no_down_gradient=no_down_gradient,axis_bounds=axis_bounds)
        requested=normalized*LIMITS
        # Preserve the existing 150ms joint/closure projector, then inspect its
        # actual result; never check a different vector from the one dispatched.
        projected=projector.project(q,requested);reason=projector.reason
        if self.no_down_latched and (projected[1]<-1e-10 or np.min(no_down_gradient@(projected/LIMITS))<-1e-7):
            raise ValueError('feedback_projected_command_moves_bucket_down')
        accepted=None;minimum=None;preview_error=None
        for factor in (1.,.5,.25,.125,0.):
            candidate=projected*factor;chain_q=self.chain.q.copy()
            try:
                self.chain.q=q.copy();remaining=cfg.preview_seconds
                scale=max(1.,float(np.max(np.abs(candidate)/np.array([.12,.0225,.0225,.0225]))))
                minimum=clearance
                body_start=self.body_heights(q) if self.no_down_latched else None
                minimum_body_delta=0.
                while remaining>1e-12:
                    dt=min(1/60,remaining,.05/scale)
                    predicted,_=projector._advance_preview(candidate/scale,min(.05,dt*scale))
                    minimum=min(minimum,self.ground.measure(predicted)['clearance_m'])
                    if body_start is not None:minimum_body_delta=min(minimum_body_delta,float(np.min(self.body_heights(predicted)-body_start)))
                    remaining-=dt
                if minimum>=cfg.ground_clearance_m-1e-6 and minimum_body_delta>=-1e-6:
                    accepted=candidate;break
            except ValueError as error:preview_error=str(error)
            finally:self.chain.q=chain_q
        if accepted is None:raise ValueError('feedback_ground_preview_infeasible:'+str(preview_error))
        if abs(state[1])>cfg.maximum_lateral_error_m:raise ValueError('feedback_lateral_corridor_exceeded')
        self.last_frame=frame;self.last_phase=phase
        return requested,accepted,dict(mode=mode,coordinates=state.tolist(),target=None if target is None else target.tolist(),
            cut_target=None if self.cut_target is None else self.cut_target.tolist(),contact=None if self.contact is None else self.contact.tolist(),
            soil_plan=self.soil_plan,root_above_surface_m=root_excess,root_ready=self.root_ready,pitch_velocity_upper=pitch_upper,
            progress=progress,desired_task_velocity=velocity.tolist(),predicted_task_velocity=(jac@(accepted/LIMITS)).tolist(),
            normalized_requested=normalized.tolist(),preview_minimum_clearance_m=minimum,preview_scale=factor,
            closure_fraction=closure_fraction,no_down_latched=self.no_down_latched,no_down_trigger_frame=self.no_down_frame,
            no_down_minimum_velocity_m_s=None if no_down_gradient is None else float(np.min(no_down_gradient@(accepted/LIMITS))),
            no_down_preview_minimum_delta_m=None if not self.no_down_latched else minimum_body_delta,
            original_projection_reason=reason,ground_linear_lower=lower,depth_velocity_lower=depth_lower)
