"""Unchanged acquisition targets; carry command history and fix preview frame."""
import copy,importlib.util,sys
import numpy as np
from runtime import ROOT,W,SOURCE,ContactRuntime,read
sys.path.insert(0,str(ROOT/'work/contact-preview-lift-20260924-v1'))
import lift_feedback as frozen
from control_history import DYNAMIC
sys.path.append(str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2'))
import yaw_preview
spec=importlib.util.spec_from_file_location('_frozen_d154_state',ROOT/'work/contact-multisource-lift-20260924-v1/candidate_multisource.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
MultisourceRuntime=ContactRuntime

class LiftFeedback(frozen.LiftFeedback):
    def __init__(self,*args,initial_previous=None,**kw):
        super().__init__(*args,**kw)
        if initial_previous is None:initial_previous=read(str(SOURCE)+'.json')['transport_controller_state']['previous_command']
        self.previous_command=np.asarray(initial_previous,float).copy()
        if self.previous_command.shape!=(4,) or not np.isfinite(self.previous_command).all():raise ValueError('invalid_previous_command')
    def command(self,**kw):
        original=frozen.continuous_preview;rotation=yaw_preview.rotation(self.chain,kw['qpos'])
        saved={k:copy.deepcopy(getattr(self,k)) for k in (*DYNAMIC,'lift_intent','lift_start_origin_z')}
        self.before_command=saved
        try:
            frozen.continuous_preview=lambda p,c,dt,r:yaw_preview.preview(p,c,dt,r,rotation)
            return super().command(**kw)
        except BaseException:
            for k,v in saved.items():setattr(self,k,v)
            raise
        finally:frozen.continuous_preview=original

def save_control(out,r,feedback,phases,features,previous_action,previous_load):
    # The frozen D154 loop can compute the next feature row before a bounded
    # lift stop. Retain exactly the history of issued/executed frames instead.
    features.retain_executed(r.frames)
    if r.frames==0 and feedback.last_frame>0:
        for k,v in feedback.before_command.items():setattr(feedback,k,copy.deepcopy(v))
    old.save_control(out,r,feedback,phases,features,previous_action,previous_load)
