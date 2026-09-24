"""Preserve the pre-decision history if no physical frame is issued."""
import copy
from contact_phase_bc_inference import OnlinePhaseFeatures

class CausalFeatures(OnlinePhaseFeatures):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.before_update=None
    def update(self,row):
        self.before_update=copy.deepcopy((self.rows,self.last_frame,self.last_sequence))
        return super().update(row)
    def retain_executed(self,frame):
        if self.last_frame is None or self.last_frame<=frame:return
        expected=None if frame==0 else frame
        if self.before_update is None or self.before_update[1]!=expected:
            raise ValueError('causal_feature_snapshot_unavailable')
        self.rows,self.last_frame,self.last_sequence=copy.deepcopy(self.before_update)
