"""Reuse qualified carry/tracking, with the qualified rotating preview."""
import copy,importlib.util,sys
import numpy as np
from runtime import ROOT,W
sys.path.append(str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2'))
import yaw_preview
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
carry=module('_second_carry_base',W/'carry_feedback.py')
sys.path.append(str(ROOT/'work/contact-carry-transport-20260924-v2'))
transport=module('_second_transport_base',W/'transport_feedback.py')
LIMITS=transport.LIMITS
def bounded_preview(owner,feedback,p,c,dt,r,rotation):
    predicted,error=yaw_preview.preview(p,c,dt,r,rotation)
    if getattr(owner,'stage',None)=='FINISH_CURL' and hasattr(owner,'loaded_center_peak') and np.mean(feedback.body_heights(predicted)) < owner.loaded_center_peak-.010-1e-9:
        raise ValueError('loaded_preview_cumulative_drawdown_exceeds_10mm')
    return predicted,error
class CarryFeedback(carry.CarryFeedback):
    def command(self,**kwargs):
        original=carry.preview;rotation=yaw_preview.rotation(self.chain,kwargs['qpos'])
        before={k:copy.deepcopy(getattr(self,k)) for k in ('previous_command','last_frame','first_command','stage','handoff_frame')}
        try:
            carry.preview=lambda p,c,dt,r:bounded_preview(self,self,p,c,dt,r,rotation)
            return super().command(**kwargs)
        except BaseException:
            for k,v in before.items():setattr(self,k,v)
            raise
        finally:carry.preview=original
class TransportController(transport.TransportController):
    def command(self,stage,q,target,desired,velocity):
        original=transport.preview;rotation=yaw_preview.rotation(self.r.chain,q)
        bias=self.tracking.bias.copy();previous=self.previous.copy();first=self.first
        try:
            transport.preview=lambda p,c,dt,r:yaw_preview.preview(p,c,dt,r,rotation)
            return super().command(stage,q,target,desired,velocity)
        except BaseException:
            self.tracking.bias=bias;self.previous=previous;self.first=first;raise
        finally:transport.preview=original
