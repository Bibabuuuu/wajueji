"""Geometric joint-target curl intent through the existing constrained feedback.

No learned weights or timer-based phase switch. Physics and guards stay in the
owning runtime; this candidate grants no task/volume qualification.
"""
import numpy as np

def joint_target_intent(actual, reference, lower, upper, speed_limits, gain):
    q=np.asarray(actual,float);ref=np.asarray(reference,float);limits=np.asarray(speed_limits,float)
    if q.shape!=(4,) or ref.shape!=(4,) or limits.shape!=(4,):raise ValueError('geometry_intent_shape')
    if not np.isfinite(np.r_[q,ref,limits,lower,upper,gain]).all():raise ValueError('geometry_intent_nonfinite')
    if upper<=lower or gain<=0 or np.any(limits<=0):raise ValueError('geometry_intent_bounds')
    target=ref.copy()
    # Explicit baseline design: hold the other axes at causal curl entry,
    # and close to 99.5% of the protected cylinder range by position feedback.
    target[3]=lower+.995*(upper-lower)
    return np.clip(gain*(target-q)/limits,-1,1),target

def geometric_feedback_class(base):
    class GeometricFeedback(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.geometry_curl_reference=None
            self.geometry_last_target=None

        def command(self,*,frame,qpos,phase,bc_action,projector):
            from contact_bc_inference import LIMITS
            from run_layered_47_sparse_manual_contact_episode import BUCKET_INTERNAL_UPPER_TARGET_M,KINEMATIC_JOINT_MARGIN_M
            q=np.asarray(qpos,float)
            if q.shape!=(19,) or not np.isfinite(q).all():raise ValueError('geometry_qpos')
            # Baseline ignores the caller's policy placeholder. Approach/cut
            # already use geometric feedback; the existing lift stays intact.
            intent=np.zeros(4)
            if phase in ('curl_load','lift','off_soil'):
                current=q[self.chain.axes]
                if self.geometry_curl_reference is None:self.geometry_curl_reference=current.copy()
                low,high=self.chain.limits[self.chain.axes[3]]
                upper=min(BUCKET_INTERNAL_UPPER_TARGET_M,high-KINEMATIC_JOINT_MARGIN_M)
                intent,self.geometry_last_target=joint_target_intent(current,self.geometry_curl_reference,low,upper,LIMITS,self.config.position_gain_s)
            request,applied,detail=super().command(frame=frame,qpos=q,phase=phase,bc_action=intent,projector=projector)
            if detail['mode']=='bc_with_lateral_depth_feedback':
                detail['mode']='geometric_curl_with_lateral_depth_feedback'
                detail['geometry_curl_reference']=self.geometry_curl_reference.tolist()
                detail['geometry_joint_target']=self.geometry_last_target.tolist()
            return request,applied,detail
    return GeometricFeedback

class NoPolicyWorker:
    """API adapter only; zeros are ignored by the geometric intent generator."""
    def __init__(self,*args,**kwargs):
        self.ready={'status':'ready','controller':'geometry_feedback_baseline','neural_inference':False}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def predict(self,*args):
        import time
        start=time.perf_counter()
        placeholder=np.zeros(4)
        return placeholder,{'roundtrip_seconds':time.perf_counter()-start,'worker_seconds':None,
                            'source':'local_placeholder_not_neural_inference'}

def self_check():
    limits=np.array([.12,.0225,.0225,.0675]);ref=np.array([.02,.28,.35,.51])
    action,target=joint_target_intent(ref,ref,.3,.847,limits,1.5)
    assert np.array_equal(action[:3],np.zeros(3)) and action[3]>0
    assert .3<target[3]<.847 and abs(target[3]-(.3+.995*(.847-.3)))<1e-12
    at_target,target2=joint_target_intent(target,ref,.3,.847,limits,1.5)
    assert np.array_equal(at_target,np.zeros(4)) and np.array_equal(target,target2)
    moved=ref.copy();moved[:3]+=[.005,-.001,.002]
    correction,_=joint_target_intent(moved,ref,.3,.847,limits,1.5)
    assert correction[0]<0 and correction[1]>0 and correction[2]<0
    for case in [([1,2],ref,.3,.847,limits,1.5),(ref,ref,.9,.8,limits,1.5),(ref,ref,.3,.8,limits,0)]:
        try:joint_target_intent(*case)
        except ValueError:pass
        else:raise AssertionError('invalid_input_accepted')
    print('geometry intent checks passed; physics_steps=0')

if __name__=='__main__':self_check()
