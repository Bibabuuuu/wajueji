"""Frozen transport commands with rotating-frame preview and transactional state."""
import importlib.util
from runtime import ROOT
import yaw_preview
spec=importlib.util.spec_from_file_location('_frozen_d158_controller',ROOT/'work/contact-fixed-truck-continuation-20260925-v1/controller.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
LIMITS=prior.LIMITS

class TransportController(prior.TransportController):
    def command(self,stage,q,target,desired,velocity):
        reference_rotation=yaw_preview.rotation(self.r.chain,q)
        original=prior.base.preview
        bias=self.tracking.bias.copy();previous=self.previous.copy();first=self.first
        try:
            prior.base.preview=lambda projector,cmd,dt,residual:yaw_preview.preview(projector,cmd,dt,residual,reference_rotation)
            return super().command(stage,q,target,desired,velocity)
        except BaseException:
            self.tracking.bias=bias;self.previous=previous;self.first=first
            raise
        finally:prior.base.preview=original
