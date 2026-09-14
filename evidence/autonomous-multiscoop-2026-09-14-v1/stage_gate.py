"""Actual-state confirmation for acquisition; elapsed path time is insufficient."""
import math

class StageGate:
    def __init__(self,stage,entered,reference_duration):
        self.stage=stage;self.entered=entered;self.duration=reference_duration;self.since=None;self.last=None
    def update(self,now,*,reached,fresh,cavity_kg=None,clear=False):
        if not math.isfinite(now) or (self.last is not None and now<=self.last):raise ValueError('stage_clock_not_fresh')
        self.last=now
        if not fresh:raise ValueError('perception_expired')
        if now-self.entered>self.duration+8:raise TimeoutError('stage_tracking_timeout:'+self.stage)
        condition=bool(reached)
        if self.stage in ('CURL','LIFT') and cavity_kg is not None:
            if not math.isfinite(cavity_kg) or cavity_kg<0:raise ValueError('invalid_inventory')
            condition=condition and cavity_kg>=20.  # Existing development acquisition minimum.
        if self.stage=='LIFT':condition=condition and clear
        self.since=(now if self.since is None else self.since) if condition else None
        return self.since is not None and now-self.since>=.25-1e-9
