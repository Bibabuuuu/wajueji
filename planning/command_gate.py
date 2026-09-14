"""Pure automatic-intent gate. Does not publish commands or own manual controls."""
import math
import numpy as np
from perception.core import observation_fresh


class AutomaticIntentGate:
    def __init__(self,speed_limits,command_ttl_s=.05):
        self.limits=np.asarray(speed_limits,float)
        if self.limits.shape!=(4,) or not np.isfinite(self.limits).all() or (self.limits<=0).any() or not math.isfinite(command_ttl_s) or command_ttl_s<=0:raise ValueError('invalid_gate_configuration')
        self.ttl=command_ttl_s;self.enabled=False;self.binding=None;self.last_clock=None;self.last_command=None

    def arm(self,episode,plan_id):
        if not episode or not plan_id:raise ValueError('explicit_binding_required')
        self.binding=(episode,plan_id);self.enabled=True;self.last_command=None;self.last_clock=None

    def stop(self,reason):
        self.enabled=False
        return {'automatic_velocity':[0.,0.,0.,0.],'automatic_enabled':False,'reason':reason,'requires_rearm':True,'published':False}

    def select(self,packet,intent,*,now_sim_s,now_wall_s,manual=False,paused=False,fault=False):
        if fault:return self.stop('fault')
        if manual:return self.stop('manual_override')
        if paused:return self.stop('paused')
        if not self.enabled:return self.stop('not_armed')
        try:
            if not all(math.isfinite(v) for v in (now_sim_s,now_wall_s)):raise ValueError('invalid_clock')
            if self.last_clock and (now_sim_s<self.last_clock[0] or now_wall_s<self.last_clock[1]):raise ValueError('clock_reversed')
            self.last_clock=(now_sim_s,now_wall_s)
            if packet['schema']!='excavation_observation_v1' or packet['episode']!=self.binding[0] or intent['plan_id']!=self.binding[1] or intent['episode']!=self.binding[0]:raise ValueError('binding_mismatch')
            if not observation_fresh(packet,now_sim_s,now_wall_s) or not packet['terrain']['valid']:raise ValueError('observation_invalid_or_expired')
            if packet['frame']!='world' or packet['units']['length']!='m' or packet['units']['time']!='s':raise ValueError('observation_contract')
            q=np.asarray(packet['proprioception']['joints'],float)
            if q.shape!=(19,) or not np.isfinite(q).all():raise ValueError('invalid_proprioception')
            created=float(intent['created_monotonic_s']);seq=int(intent['sequence']);velocity=np.asarray(intent['velocity'],float)
            if not math.isfinite(created) or not 0<=now_wall_s-created<=self.ttl:raise ValueError('command_expired_or_future')
            if seq<0 or seq!=intent['sequence'] or velocity.shape!=(4,) or not np.isfinite(velocity).all() or np.any(np.abs(velocity)>self.limits):raise ValueError('invalid_command')
            key=(seq,created,tuple(velocity))
            if self.last_command and (seq<self.last_command[0] or created<self.last_command[1] or (seq==self.last_command[0] and key!=self.last_command)):raise ValueError('command_reordered_or_changed')
            self.last_command=key
            return {'automatic_velocity':velocity.tolist(),'automatic_enabled':True,'reason':'fresh_bound_intent','requires_rearm':False,'published':False}
        except (KeyError,ValueError,TypeError,IndexError,AttributeError,OverflowError) as error:return self.stop(str(error))
