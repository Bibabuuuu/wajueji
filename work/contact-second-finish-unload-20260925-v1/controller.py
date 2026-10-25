"""Reuse qualified carry/tracking, with the qualified rotating preview."""
import copy,importlib.util,sys
from runtime import ROOT
sys.path.append(str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2'))
import yaw_preview
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
carry=module('_second_carry_base',ROOT/'work/contact-carry-handoff-20260924-v1/carry_feedback.py')
sys.path.append(str(ROOT/'work/contact-carry-transport-20260924-v2'))
transport=module('_second_transport_base',ROOT/'work/contact-carry-transport-20260924-v2/controller.py')
LIMITS=transport.LIMITS
class CarryFeedback(carry.CarryFeedback):
    def command(self,**kwargs):
        original=carry.preview;rotation=yaw_preview.rotation(self.chain,kwargs['qpos'])
        before={k:copy.deepcopy(getattr(self,k)) for k in ('previous_command','last_frame','first_command','stage','handoff_frame')}
        try:
            carry.preview=lambda p,c,dt,r:yaw_preview.preview(p,c,dt,r,rotation)
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
