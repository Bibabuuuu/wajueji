"""Unchanged v4 transport control and rotating preview, without curl replay."""
import importlib.util,sys
from runtime import ROOT
sys.path.append(str(ROOT/'work/contact-carry-transport-20260924-v2'))
sys.path.append(str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2'))
import yaw_preview
spec=importlib.util.spec_from_file_location('_fixed_route_transport',ROOT/'work/contact-second-finish-unload-20260925-v4/transport_feedback.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMITS=base.LIMITS
class TransportController(base.TransportController):
    def command(self,stage,q,target,desired,velocity):
        original=base.preview;rotation=yaw_preview.rotation(self.r.chain,q)
        bias=self.tracking.bias.copy();previous=self.previous.copy();first=self.first
        try:
            base.preview=lambda p,c,dt,r:yaw_preview.preview(p,c,dt,r,rotation)
            return super().command(stage,q,target,desired,velocity)
        except BaseException:
            self.tracking.bias=bias;self.previous=previous;self.first=first;raise
        finally:base.preview=original
