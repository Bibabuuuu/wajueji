"""D153 isolated controller; original production command function unchanged.

The command method is a source-bound copy with two explicit changes: continuous
soft-closure preview, and frozen bounded lift intent after 95% closure.
"""
import numpy as np
from contact_geometric_feedback import ContactGeometryFeedback as Base, bounded_velocity, rate_bounded_velocity, LIMITS, tool_pose, TIP
from continuous_preview import preview as continuous_preview
class LiftFeedback(Base):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.lift_intent=False
        self.lift_start_origin_z=None

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
            if phase in ('curl_load','lift') and closure_fraction is not None and closure_fraction>=.95 and not self.lift_intent:
                self.lift_intent=True
                self.lift_start_origin_z=float(self.assembly.assemble(q,include_meshes=False)[1]['bucket'][2,3])
            if self.lift_intent:
                origin_z=float(self.assembly.assemble(q,include_meshes=False)[1]['bucket'][2,3])
                remaining=self.lift_start_origin_z+.35-origin_z
                if remaining<=0:raise ValueError('D153_lift_height_budget_reached')
                velocity=np.array([-.03,-cfg.position_gain_s*state[1],min(.08,1.5*remaining),0.])
                mode='bounded_feedback_lift'
                if axis_bounds is None:raise ValueError('D153_lift_requires_no_down_latched')
                axis_bounds[3]=(0.,axis_bounds[3][1])

        velocity[:3]=np.clip(velocity[:3],-cfg.translation_speed_m_s,cfg.translation_speed_m_s)
        velocity[3]=np.clip(velocity[3],-cfg.pitch_speed_rad_s,cfg.pitch_speed_rad_s)
        clearance=self.ground.measure(q)['clearance_m']
        lower=(cfg.ground_clearance_m-clearance)/cfg.preview_seconds
        depth_lower=None if self.cut_target is None else cfg.position_gain_s*(self.cut_target[2]-state[2])
        constraints=dict(exact_pitch=phase=='approach_contact',depth_lower=depth_lower,
                         pitch_upper=pitch_upper,no_down_gradient=no_down_gradient)
        slew_override=None
        if cfg.command_rate_per_s is None:
            normalized=bounded_velocity(jac,velocity,ground_gradient,lower,axis_bounds=axis_bounds,**constraints)
        else:
            normalized,slew_override=rate_bounded_velocity(jac,velocity,ground_gradient,lower,
                previous=self.previous_command/LIMITS,rate=cfg.command_rate_per_s,
                axis_bounds=axis_bounds,**constraints)
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
                    predicted,_=continuous_preview(projector,candidate/scale,min(.05,dt*scale),self.chain.residual(q))
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
        self.previous_command=accepted.copy()
        return requested,accepted,dict(mode=mode,coordinates=state.tolist(),target=None if target is None else target.tolist(),
            cut_target=None if self.cut_target is None else self.cut_target.tolist(),contact=None if self.contact is None else self.contact.tolist(),
            soil_plan=self.soil_plan,root_above_surface_m=root_excess,root_ready=self.root_ready,pitch_velocity_upper=pitch_upper,
            progress=progress,desired_task_velocity=velocity.tolist(),predicted_task_velocity=(jac@(accepted/LIMITS)).tolist(),
            normalized_requested=normalized.tolist(),preview_minimum_clearance_m=minimum,preview_scale=factor,
            closure_fraction=closure_fraction,no_down_latched=self.no_down_latched,no_down_trigger_frame=self.no_down_frame,
            no_down_minimum_velocity_m_s=None if no_down_gradient is None else float(np.min(no_down_gradient@(accepted/LIMITS))),
            no_down_preview_minimum_delta_m=None if not self.no_down_latched else minimum_body_delta,
            lift_intent=self.lift_intent,lift_start_origin_z_m=self.lift_start_origin_z,
            original_projection_reason=reason,ground_linear_lower=lower,depth_velocity_lower=depth_lower,
            **({} if cfg.command_rate_per_s is None else dict(command_rate_per_s=cfg.command_rate_per_s,
                slew_safety_override=slew_override)))

