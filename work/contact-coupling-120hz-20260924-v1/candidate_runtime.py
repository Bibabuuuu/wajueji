"""Isolated D150 exchange-frequency candidate; production runtime is unchanged."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from layered_47_cycle_runtime import CycleRuntime, manual, motion, controller_snapshot
from contact_bc_ground_guard import BucketGroundGuard


def aggregate(rows):
    result = copy.deepcopy(rows[-1])
    for key in ('wall_seconds','assignment_wall_seconds','mpm_wall_seconds','mujoco_wall_seconds',
                'bucket_mpm_contact_samples','rock_bucket_contact_substeps','external_rigid_support_substeps','forbidden_contact_count'):
        result[key] = sum(row[key] for row in rows)
    for key in ('closure_m','maximum_rock_linear_speed_m_s','maximum_rock_angular_speed_rad_s','maximum_rock_position_step_m'):
        result[key] = max(row[key] for row in rows)
    result['hard_guard_checks'] = {key:all(row['hard_guard_checks'][key] for row in rows) for key in rows[0]['hard_guard_checks']}
    result['hard_guard'] = all(result['hard_guard_checks'].values())
    result['finite'] = all(row['finite'] for row in rows)
    result['stable'] = all(row['stable'] for row in rows)
    result['rigid_contact_substeps'] = {key:sum(row['rigid_contact_substeps'].get(key,0) for row in rows) for key in set().union(*(row['rigid_contact_substeps'] for row in rows))}
    # Full material/sweep state is logged per exchange. Outer row retains the
    # last ledger, with safety covering both exchanges, never sums balances.
    for key in ('hencky_material_guard','moving_bucket_sweep'):
        result[key]['safe'] = all(row[key]['safe'] for row in rows)
    result['coupling_exchange_count'] = len(rows)
    return result


class CandidateRuntime(CycleRuntime):
    coupling_hz = 120

    def __enter__(self):
        super().__enter__()
        self.exchange_config = copy.deepcopy(self.base)
        self.exchange_config['probe'].update(output_dt_s=1/self.coupling_hz,mujoco_internal_substeps=4800//self.coupling_hz)
        self.exchange_ground = BucketGroundGuard(self.assembly,self.bridge.state['mechanism_qpos'])
        self.timings = []
        self.candidate_bindings = {str(Path(__file__).resolve()):motion.probe.bench.sha256(__file__)}
        return self

    def capture(self):
        if self.coupling_hz == 60:
            return super().capture()
        elapsed = controller_snapshot(self.bridge.state)['native_time_s'] - self.metadata['controller_state']['native_time_s']
        return motion.probe.snapshots.capture_arrays(self.fixture,self.bridge.state,float(self.arrays['simulation_time_s'][0])+elapsed)

    def save(self,path,**kwargs):
        info=super().save(path,**kwargs)
        if self.coupling_hz != 60:
            sidecar=Path(str(path)+'.json');meta=json.loads(sidecar.read_text())
            meta.update(base_schema=meta['schema'],schema='layered_47_coupling_candidate_checkpoint_v1',
                        coupling_hz=self.coupling_hz,control_hz=60,rigid_hz=4800,
                        candidate_bindings=self.candidate_bindings,
                        continuation_qualified=False)
            sidecar.write_text(json.dumps(meta,indent=2)+'\n')
        return info

    def step(self,command):
        started=time.perf_counter()
        if self.coupling_hz == 60:
            result=super().step(command)
        else:
            command=np.asarray(command,dtype=float)
            if command.shape!=(4,) or not np.isfinite(command).all() or np.any(np.abs(command)>manual.COMMAND_LIMITS_SI+1e-15):
                raise ValueError('cycle_command_envelope')
            self.bridge.command=command
            rows=[]
            for half in range(2):
                self.guard.prepare_frame()
                row,_=motion.probe.bench._advance_frame(self.fixture,self.bridge,self.exchange_config,self.frames+1)
                row['moving_bucket_sweep']=self.sweep.finish_frame()
                row['hencky_material_guard']=self.guard.summarize(self.fixture.state)
                q=self.fixture.state.particle_q.numpy();v=self.fixture.state.particle_qd.numpy()
                row.update(applied_command_si=command.tolist(),
                    finite=bool(row['finite'] and np.isfinite(q).all() and np.isfinite(v).all() and motion.probe.bench._finite_tree(self.bridge.state)),
                    joint_bounds_ok=self.projector.physical_bounds_ok(np.asarray(self.bridge.state['mechanism_qpos'])),
                    bucket_mpm_wrench_finite=all(np.isfinite(x).all() for x in self.captured['wrenches']['bucket'].values()),
                    rigid_contact_substeps=self.bridge.state.get('contact_substeps',{}),after_state=manual.compact_state(self.bridge.state))
                row['hard_guard_checks']=manual.hard_guard_checks(row,allow_truck=self.truck_enabled)
                row['hard_guard']=all(row['hard_guard_checks'].values())
                row['ground_check']=self.exchange_ground.measure(self.bridge.state['mechanism_qpos'])
                row['exchange_in_frame']=half+1
                manual.append_json(self.output/'exchanges.ndjson',row)
                if not row['hard_guard']:raise RuntimeError('cycle_half_hard_guard:'+str(row['hard_guard_checks']))
                if not row['ground_check']['safe']:raise RuntimeError('cycle_half_ground_guard:'+str(row['ground_check']))
                rows.append(row)
            result=aggregate(rows)
            self.frames+=1
            manual.append_json(self.output/'frames.ndjson',result)
        wp.synchronize()
        elapsed=time.perf_counter()-started
        timing=dict(frame=self.frames,runtime_step_wall_seconds=elapsed,
                    physics_wall_seconds=result['wall_seconds'],mpm_wall_seconds=result['mpm_wall_seconds'],
                    mujoco_wall_seconds=result['mujoco_wall_seconds'])
        self.timings.append(timing)
        manual.append_json(self.output/'timing.ndjson',timing)
        return result
