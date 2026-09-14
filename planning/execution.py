"""Bounded executor for an already checked empty approach in a prescribed-motion simulator."""
import numpy as np
from .command_gate import AutomaticIntentGate


class ApproachExecutor:
    def __init__(self, path, axes, speed_limits, dt, grid_contract):
        self.path=np.array(path,float,copy=True);self.axes=np.asarray(axes,int)
        self.dt=float(dt);self.grid_contract=grid_contract
        self.gate=AutomaticIntentGate(speed_limits);self.cursor=0;self.stop_reason=None
        if self.path.ndim!=2 or self.path.shape[1]!=19 or len(self.path)<2 or not np.isfinite(self.path).all() or self.axes.shape!=(4,) or not 0<self.dt<=.05:
            raise ValueError('invalid_execution_path')
        if np.any(np.abs(np.diff(self.path[:,self.axes],axis=0)/self.dt)>self.gate.limits+1e-10):
            raise ValueError('path_speed_limit')

    def _validate(self, packet, q, sim):
        if q.shape!=(19,) or not np.isfinite(q).all() or np.max(np.abs(q-self.path[self.cursor]))>1e-5:
            raise ValueError('actual_path_tracking_error')
        terrain=packet['terrain']
        contract=(terrain['origin_xy_m'],terrain['resolution_m'],list(np.shape(terrain['height_m'])))
        if contract!=self.grid_contract:raise ValueError('map_grid_changed')
        measured=np.asarray(packet['proprioception']['joints'],float)
        age=sim-float(packet['time']['capture_sim_s'])
        if measured.shape!=(19,) or not np.isfinite(measured).all() or age<0 or np.any(np.abs(measured[self.axes]-q[self.axes])>self.gate.limits*age+1e-5):
            raise ValueError('sensor_pose_tracking_error')

    def arm(self, packet, q, plan_id, *, clock, check_pose):
        self.gate.stop('arming');self.cursor=0;self.stop_reason=None
        try:
            sim,wall=clock();q=np.asarray(q,float);self._validate(packet,q,sim)
            if np.max(np.abs(np.asarray(packet['proprioception']['joints'])-q))>1e-5:raise ValueError('start_observation_pose_mismatch')
            check_pose(q,None,packet)
            self.gate.arm(packet['episode'],plan_id)
            sim,wall=clock()
            intent={'episode':packet['episode'],'plan_id':plan_id,'sequence':0,'created_monotonic_s':wall,'velocity':[0.]*4}
            result=self.gate.select(packet,intent,now_sim_s=sim,now_wall_s=wall)
            if not result['automatic_enabled']:raise ValueError(result['reason'])
        except Exception as error:
            self.stop_reason=str(error);self.gate.stop(self.stop_reason);raise

    def step(self, packet, q, *, clock, advance, check_pose, prepare, fault=False, manual=False, paused=False):
        """prepare returns a physics commit callable. Failed checks never call it.

        Caller advances held soil separately on rejection. No automatic rearm.
        """
        q=np.asarray(q,float);diagnostics={};result=None
        try:
            if self.cursor==len(self.path)-1:raise ValueError('path_complete')
            sim,wall=clock()
            v=(self.path[self.cursor+1,self.axes]-q[self.axes])/self.dt
            intent={'episode':self.gate.binding[0] if self.gate.binding else '',
                    'plan_id':self.gate.binding[1] if self.gate.binding else '',
                    'sequence':self.cursor+1,'created_monotonic_s':wall,'velocity':v.tolist()}
            result=self.gate.select(packet,intent,now_sim_s=sim,now_wall_s=wall,fault=fault,manual=manual,paused=paused)
            if not result['automatic_enabled']:raise ValueError(result['reason'])
            self._validate(packet,q,sim)
            proposed=np.asarray(advance(q.copy(),v.copy(),self.dt),float)
            if proposed.shape!=(19,) or not np.isfinite(proposed).all() or np.max(np.abs(proposed-self.path[self.cursor+1]))>1e-5:raise ValueError('proposed_path_tracking_error')
            diagnostics=check_pose(proposed,q,packet)
            commit=prepare(proposed,v)
            sim,wall=clock()
            result=self.gate.select(packet,intent,now_sim_s=sim,now_wall_s=wall)
            if not result['automatic_enabled']:raise ValueError(result['reason'])
        except (ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError) as error:
            if self.stop_reason is None:self.stop_reason=str(error)
            result=self.gate.stop(self.stop_reason)
            return q.copy(),dict(result,consumed=False,cursor=self.cursor,diagnostics=diagnostics)
        # Propagate a physics error: continuing from an uncertain state is not valid.
        commit();self.cursor+=1
        return proposed,dict(result,consumed=True,cursor=self.cursor,diagnostics=diagnostics)
