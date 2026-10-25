"""Opt-in D152 runtime, strategy/MPM60Hz and native4800Hz."""
import copy,json,time
from pathlib import Path
import numpy as np
import warp as wp
from layered_47_cycle_runtime import CycleRuntime,manual,motion
from shadow import Shadow
from full_response_operator import FullResponseOperator
W=Path(__file__).resolve().parent
class CandidateRuntime(CycleRuntime):
    def __enter__(self):
        super().__enter__()
        try:
            self.shadow=Shadow(self.output,self.output/'runtime/audit.xml')
            if np.any(self.fixture.solver._mpm_model.collider.body_com.numpy()[self.fixture.bucket_body_index]!=0):raise ValueError('bucket_reference_not_origin')
            if self.fixture.solver.collider_velocity_mode!='forward':raise ValueError('D152_requires_forward_collider_velocity')
            self.response=FullResponseOperator(self.fixture.solver,self.fixture.bucket_body_index,int(self.fixture.bucket_collider_ids[0]))
            self.timings=[];self.prediction=None;self.real_step_calls=0
            original=self.fixture.step;real_step=self.bridge.step
            def counted_step(*a,**kw):self.real_step_calls+=1;return real_step(*a,**kw)
            self.bridge.step=counted_step
            self.half_count=0
            native_read=self.bridge.bridge._read_state
            def audited_native_read(*a,**kw):
                state=native_read(*a,**kw)
                expected=self.captured['wrenches']
                for key,native_key in (('force','applied_bucket_mpm_force_world_n'),('torque','applied_bucket_mpm_torque_world_nm')):
                    if not np.array_equal(state[native_key],expected['bucket'][key]):raise ValueError('bucket_wrench_dispatch_changed')
                for rock in state['applied_rocks_mpm']:
                    for key,native_key in (('force','force_world_n'),('torque','torque_world_nm')):
                        if not np.array_equal(rock[native_key],expected[rock['rock_id']][key]):raise ValueError('rock_wrench_dispatch_changed')
                self.half_count+=1
                keys=('time_s','mechanism_qpos','mechanism_qvel','equivalent_drive_load','bucket_integral_effort','bucket_active_curl_effort','applied_bucket_mpm_force_world_n','applied_bucket_mpm_torque_world_nm','contact_substeps','closure_m','forbidden_contact_count')
                manual.append_json(self.output/'native-halves.ndjson',dict(frame=self.frames+1,half=1+(self.half_count-1)%2,dispatch_exact=True,**{k:state[k] for k in keys}))
                return state
            self.bridge.bridge._read_state=audited_native_read
            def full_response_step(dt,origins):
                if dt!=1/60:raise ValueError('D152_60Hz_only')
                state_before=copy.deepcopy(self.bridge.state);pending_before=copy.deepcopy(self.sweep.pending)
                C,free,detail=self.shadow.forecast(self.bridge.state,self.bridge.command,self.sweep.pending)
                if self.bridge.state!=state_before or self.sweep.pending!=pending_before:raise ValueError('shadow_modified_live_state')
                self.response.set(C)
                velocities=self.fixture.state.body_qd.numpy();velocities[self.fixture.bucket_body_index]=free
                self.fixture.state.body_qd.assign(wp.array(velocities,dtype=wp.spatial_vector,device=self.fixture.device))
                before_calls=self.response.calls
                result=original(dt,origins)
                if self.response.calls!=before_calls+1:raise ValueError('full_response_operator_call_count')
                detail['predicted_mechanism_qvel']=[s['mechanism_qvel'] for s in detail.pop('predicted_states')]
                detail.update(frame=self.frames+1,live_state_unchanged=True,pending_unchanged_before_real_step=True,mpm_operator_calls=1)
                self.prediction=detail
                manual.append_json(self.output/'response.ndjson',detail)
                return result
            self.fixture.step=full_response_step
            self.candidate_bindings={str(p):motion.probe.bench.sha256(p) for p in W.glob('*.py')}
            self.candidate_bindings.update({str(p):motion.probe.bench.sha256(p) for p in W.glob('*.cc')})
            return self
        except BaseException:self.close();raise
    def step(self,command):
        start=time.perf_counter();before=self.real_step_calls;before_halves=self.half_count
        row=super().step(command)
        if self.real_step_calls!=before+1 or self.half_count!=before_halves+2:raise ValueError('D152_real_impulse_applied_more_than_once')
        wp.synchronize()
        timing=dict(frame=self.frames,runtime_step_wall_seconds=time.perf_counter()-start,physics_wall_seconds=row['wall_seconds'],mpm_wall_seconds=row['mpm_wall_seconds'],mujoco_wall_seconds=row['mujoco_wall_seconds'],predictor_wall_seconds=self.prediction['wall_seconds'],real_step_calls_this_frame=1)
        self.timings.append(timing);manual.append_json(self.output/'timing.ndjson',timing)
        return row
    def save(self,path,**kw):
        info=super().save(path,**kw);sidecar=Path(str(path)+'.json');meta=json.loads(sidecar.read_text())
        meta.update(base_schema=meta['schema'],schema='layered_47_full_response_candidate_checkpoint_v1',coupling_hz=60,control_hz=60,rigid_hz=4800,candidate_bindings=getattr(self,'candidate_bindings',{}),continuation_qualified=False)
        sidecar.write_text(json.dumps(meta,indent=2)+'\n');return info
    def close(self):
        if getattr(self,'response',None) is not None:self.response.close();self.response=None
        if getattr(self,'shadow',None) is not None:self.shadow.close();self.shadow=None
        super().close()
