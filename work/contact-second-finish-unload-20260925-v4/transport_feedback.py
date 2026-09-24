"""Rate-limited reference tracking with loaded retention and existing clearance."""
import sys
import numpy as np
from runtime import ROOT
from load_tracking import LoadAwareReference
sys.path.insert(0,str(ROOT/'work/contact-preview-lift-20260924-v1'))
from continuous_preview import preview
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig,rate_bounded_velocity,LIMITS
from contact_bc_ground_guard import BucketGroundGuard
from planning.cycle_geometry import tool_pose,MOUTH

class TransportController:
    def __init__(self,r,checker,previous):
        self.r=r;self.checker=checker;self.previous=np.asarray(previous,float).copy();self.first=False
        q=r.arrays['mechanism_qpos'];mouth,self.pitch=tool_pose(r.assembly,q,MOUTH);self.cylinder=float(q[15])
        self.minimum_cylinder=.37+.99*(.852-.37)
        self.tracking=LoadAwareReference(q[r.chain.axes],r.metadata['controller_state']['controller_target'])
        self.ground=BucketGroundGuard(r.assembly,q)
        self.feedback=ContactGeometryFeedback(r.assembly,r.chain,self.ground,
            dict(entry_xy_m=mouth[:2],surface_z_m=mouth[2]),q,
            FeedbackConfig(late_curl_no_down=True,command_rate_per_s=2.))

    def check(self,q,stage,*,physical=False):
        if not self.ground.measure(q)['safe']:raise ValueError('transport_ground_guard')
        if stage=='CLEAR_PILE':
            measured,_=self.checker.measure_clearance(q)
            if (measured['bucket_truck_m']<.04 or measured['mechanism_truck_m']<.02
                or measured['mechanism_ground_m']<.02 or measured['closure_m']>(.001 if physical else .0001)):
                raise ValueError('transport_retreat_clearance:'+str(measured))
        else:measured=self.checker.check(q,physical=physical)
        if stage in ('CLEAR_PILE','ALIGN'):
            if q[15]<self.minimum_cylinder:
                raise ValueError('transport_actual_lost_retention')
        return measured

    def command(self,stage,q,target,desired,velocity):
        r=self.r;loaded=stage in ('CLEAR_PILE','ALIGN')
        requested=self.tracking.command(q[r.chain.axes],target,np.asarray(desired)[r.chain.axes],velocity,loaded=loaded)
        if loaded:requested[3]=0.
        override=None
        if self.first:command=self.previous.copy()
        elif loaded:
            _,gradient,body=self.feedback.linearize(q,include_body=True)
            lower=(.03-self.ground.measure(q)['minimum_bucket_z_m'])/.15
            normalized,override=rate_bounded_velocity(np.eye(4),requested/LIMITS,gradient,lower,
                previous=self.previous/LIMITS,rate=2.,axis_bounds=[(-1,1),(0,1),(-1,1),(0,0)],
                no_down_gradient=body.mean(axis=0,keepdims=True))
            if override is not None:raise ValueError('transport_no_continuous_loaded_command:'+override)
            command=normalized*LIMITS
        else:command=np.clip(requested,self.previous-2/60*LIMITS,self.previous+2/60*LIMITS)
        applied=r.projector.project(q,command)
        if np.max(abs((applied-self.previous)/LIMITS))>2/60+1e-9:raise ValueError('transport_projection_slew')
        if loaded and applied[3]<-1e-10:raise ValueError('transport_open_before_dump')
        if self.first and not np.array_equal(applied,self.previous):raise ValueError('transport_first_command_discontinuous')
        original=r.chain.q.copy();reference=r.chain.residual(q);start_heights=self.feedback.body_heights(q)
        try:
            r.chain.q=np.asarray(q).copy()
            scale=max(1.,float(np.max(abs(applied)/np.array([.12,.0225,.0225,.0225]))))
            for _ in range(9):
                predicted,_=preview(r.projector,applied/scale,scale/60,reference)
                if not self.ground.measure(predicted)['safe']:raise ValueError('transport_preview_ground')
                if loaded and (predicted[15]<self.minimum_cylinder):
                    raise ValueError('transport_preview_loses_retention')
                if loaded and np.mean(self.feedback.body_heights(predicted)-start_heights)<-.003:
                    raise ValueError('transport_preview_downward')
        finally:r.chain.q=original
        detail=dict(stage=stage,requested=requested.tolist(),projected=not np.array_equal(command,applied),
            load_bias=self.tracking.bias.tolist(),pitch_diagnostic_rad=tool_pose(r.assembly,q,MOUTH)[1],projection_reason=r.projector.reason,slew_override=override,first_command=self.first)
        self.previous=applied.copy();self.first=False
        return applied,detail
