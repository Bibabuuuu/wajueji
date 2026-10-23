def progress_reward(raw, before, after, terminal):
    """Gamma=1 potential shaping; every contractual terminal has zero potential."""
    return raw + (0. if terminal else after) - before


"""Single-instance Gym environment: saved manual actions plus four axes and pace."""
import datetime
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from manual_trajectory_residual import LIMITS_SI, compose_action, compare_arrays
from manual_trajectory_alignment import ProgressAlignment, CONFIG, VERSION

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'work/bucket-projection-isolation-20260925-v1'
INPUT = ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
SOURCE = ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
INPUT_SHA = 'dd479e6addc20a3b09479dbd3c4798929707af40ccabbf890bbfcd560dcb35e6'
SOURCE_SHA = '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'
OBS_SIZE = 221


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, path, *, seconds=7200, purpose='training_and_evaluation'):
        self.path = Path(path)
        self.seconds = seconds
        self.purpose = purpose
        if seconds <= 0 or not np.isfinite(seconds):
            raise ValueError('physical_budget_duration')
        self.data = json.loads(self.path.read_text()) if self.path.exists() else None

    def start(self):
        if self.data is None:
            now = time.time()
            self.data = dict(seconds=self.seconds, purpose=self.purpose, epoch_start=now, epoch_deadline=now+self.seconds,
                             monotonic_deadline=time.monotonic()+self.seconds,
                             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                             started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(self.path, self.data)
        self.check()

    def remaining(self):
        if self.data is None:
            return float(self.seconds)
        if self.data['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            raise BudgetExpired('machine_rebooted_during_budget')
        return min(self.data['epoch_deadline']-time.time(), self.data['monotonic_deadline']-time.monotonic())

    def check(self):
        if self.remaining() <= 0:
            raise BudgetExpired('physical_budget_exhausted')


def retimed_baseline(commands,cursor,pace):
    """Integrate saved 60Hz velocity holds over reference progress.

    One physical frame receives that reference displacement divided by its
    unchanged dt. Stop at the first axis-speed limit instead of advancing a
    reference faster than the permitted feedforward velocity can execute.
    This is command retiming, not proof of loaded-mechanism path tracking.
    """
    end=min(float(len(commands)),cursor+pace)
    if pace==1. and cursor==int(cursor) and end-cursor==1.:
        return np.asarray(commands[int(cursor)],dtype=float).copy(),1.
    position=cursor;integral=np.zeros(4)
    while position<end-1e-12:
        first=int(position);span=min(end,float(first+1))-position
        velocity=np.asarray(commands[first],dtype=float)
        nonzero=velocity!=0
        permitted=(np.sign(velocity[nonzero])*LIMITS_SI[nonzero]-integral[nonzero])/velocity[nonzero]
        accepted=min(span,max(0.,float(permitted.min()))) if len(permitted) else span
        integral+=velocity*accepted;position+=accepted
        if accepted<span-1e-12:break
    return integral,position-cursor


def command_at(commands, cursor, action, cap_fraction=.1, *, reference_limit=None, wait_target_error=None, waiting_action_mode="signed_residual_v2"):
    if waiting_action_mode not in ("signed_residual_v2", "error_strength_v1"):
        raise ValueError("unknown_waiting_action_mode")
    action = np.asarray(action, dtype=float)
    if action.shape != (5,) or not np.isfinite(action).all() or np.any(abs(action)>1):
        raise ValueError('residual_action_contract')
    endpoint_wait=cursor==len(commands) and reference_limit==cursor
    if not (0<=cursor<len(commands) or endpoint_wait) or not 0<=cap_fraction<=.1:
        raise ValueError('residual_cursor_or_caps')
    allowed=1.+.5*action[4]
    if reference_limit is not None:
        if not np.isfinite(reference_limit) or reference_limit<cursor:
            raise ValueError('alignment_reference_limit')
        allowed=min(allowed,reference_limit-cursor)
    if allowed<=1e-12:
        error=np.zeros(4) if wait_target_error is None else np.asarray(wait_target_error,dtype=float)
        if error.shape!=(4,) or not np.isfinite(error).all():
            raise ValueError('alignment_wait_target_error')
        if waiting_action_mode=="error_strength_v1":
            strength=(action[:4]+1.)*.5
            delta=np.sign(error)*np.minimum(strength*LIMITS_SI*cap_fraction,abs(error)*60)
        else:
            delta=np.asarray(action[:4])*LIMITS_SI*cap_fraction
            delta=np.where(delta*error>0,np.sign(delta)*np.minimum(abs(delta),abs(error)*60),0.)
        return np.zeros(4),delta,delta.copy(),0.
    base,pace=retimed_baseline(commands,cursor,allowed)
    combined, delta = compose_action(base, action[:4], LIMITS_SI*cap_fraction)
    return base, delta, combined, pace


ACTION_HANDOFF_VERSION = 'invalidate_on_mode_change_v1'
TERMINAL_SCORE_VERSION = 'manual_endpoint_hold_no_gap_v1'


def environment_contract(mode='error_strength_v1'):
    if mode not in ('signed_residual_v2','error_strength_v1'):
        raise ValueError('unknown_waiting_action_mode')
    return dict(waiting_action_mode=mode,
                action_handoff_version=ACTION_HANDOFF_VERSION if mode=='error_strength_v1' else 'legacy_per_frame',
                terminal_score_version=TERMINAL_SCORE_VERSION,observation_size=OBS_SIZE,action_size=5,
                gamma=1.,timeout_bootstrap=False,input_sha256=INPUT_SHA,source_sha256=SOURCE_SHA,
                code={n:sha(ROOT/'src/v4'/n) for n in ['manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']})


def require_environment_contract(receipt,mode='error_strength_v1'):
    if receipt.get('environment_contract')!=environment_contract(mode):
        raise ValueError('environment_contract_mismatch')


def control_mode(cursor, alignment, hold):
    """Match the command's physical-frame branch, including endpoint wait."""
    if hold is not None:
        return 'hold'
    return 'wait' if alignment['reference_limit']-cursor<=1e-12 else 'follow'


def retained_score(samples):
    if len(samples)!=3 or not all(x['valid_estimate'] for x in samples):
        return None
    return min(x['fill_percent']/100 for x in samples)


class ManualResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, output_root, budget, *, cap_fraction=.1, max_frames=1560, collect=False, zero_check=False, waiting_action_mode="signed_residual_v2"):
        super().__init__()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        if waiting_action_mode not in ("signed_residual_v2", "error_strength_v1"):
            raise ValueError("unknown_waiting_action_mode")
        self.waiting_action_mode = waiting_action_mode
        self.cap_fraction = cap_fraction
        self.max_frames = max_frames
        self.collect = collect
        self.zero_check = zero_check
        self.runtime = None
        self.counter = 0
        self.completed = []
        if sha(INPUT)!=INPUT_SHA or sha(SOURCE)!=SOURCE_SHA:
            raise ValueError('manual_input_identity')
        records = [json.loads(x) for x in INPUT.read_text().splitlines()][:1170]
        if [x['frame'] for x in records]!=list(range(1,1171)):
            raise ValueError('manual_input_timebase')
        self.commands = np.asarray([x['applied_command_si'] for x in records])
        first = json.loads((OLD/'manual-prefix-1800/display/states.json').read_text())
        tail = json.loads((OLD/'manual-prefix-continuation-1620/display/states.json').read_text())
        self.references = first[:181]+tail[1:991]
        assert len(self.references)==1171
        self.reference_q = np.asarray([x['mechanism_qpos'] for x in self.references])
        self.reference_targets=np.asarray([x['controller_target'] for x in self.references])
        self.action_space = gym.spaces.Box(-1., 1., shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10., 10., shape=(OBS_SIZE,), dtype=np.float32)
        self.transitions = []
        self.cached_positions = None
        if zero_check:
            self.cached_positions = np.load(OLD/'manual-prefix-1800/display/particles.npy', mmap_mode='r')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.budget.start()
        self.counter += 1
        self.directory = self.output_root/f'episode-{self.counter:04d}'
        if self.directory.exists():
            raise ValueError('immutable_episode_already_exists')
        self.directory.mkdir()
        self.report = dict(status='initializing', episode=self.counter, physical_frames=0,
                           input_sha256=INPUT_SHA, source_sha256=SOURCE_SHA,
                           cap_fraction=self.cap_fraction, requested_pace_range=[.5,1.5],
                           actual_pace='speed-feasible reference integral, clipped at trajectory endpoint',
                           retiming_version='velocity_hold_integral_v2',
                           alignment_version=VERSION, alignment_config=CONFIG,
                           waiting_strategy=('freeze_phase_error_directed_strength_v1' if self.waiting_action_mode=='error_strength_v1'
                                             else 'freeze_phase_allow_directed_residual_v2'),
                           waiting_action_mode=self.waiting_action_mode,
                           action_handoff_version=(ACTION_HANDOFF_VERSION if self.waiting_action_mode=='error_strength_v1'
                                                   else 'legacy_per_frame'),
                           alignment_checks=0, alignment_wait_frames=0,
                           maximum_reference_lead_frames=0., maximum_alignment_error_m=0.,
                           control_hz=20, mpm_hz=60, rigid_hz=4800,
                           score_scope='uncalibrated held bucket display estimate, not truck delivery',
                           terminal_score_version=TERMINAL_SCORE_VERSION,
                           gap_role='diagnostic_only',
                           scores=[], gap_checks=[], success=False)
        self.started = time.monotonic()
        try:
            from layered_47_cycle_runtime import CycleRuntime
            from contact_bc_ground_guard import BucketGroundGuard
            from contact_phase_geometry import SolidBucketGap
            self.runtime = CycleRuntime(SOURCE, self.directory/'physics', enable_truck=True,
                                        freeze_passive_jp=True, bucket_motion_consistency=True)
            self.runtime.__enter__()
            exact, errors = compare_arrays(self.runtime.capture(), self.runtime.arrays)
            self.report['zero_step_exact'] = exact
            if not all(v for k,v in exact.items() if k!='simulation_time_s') or errors['simulation_time_s']>1e-12:
                raise ValueError('residual_initial_state_changed')
            self.ground = BucketGroundGuard(self.runtime.assembly, self.runtime.bridge.state['mechanism_qpos'])
            self.geometry = SolidBucketGap(self.runtime.proxy)
            from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
            self.vertices = _cache_rock_vertices(self.runtime.root, self.runtime.records)
            self.estimator = None
            self.cursor = 0.
            poses=[self.runtime.assembly.assemble(q,include_meshes=False)[1]['bucket']
                   for q in self.reference_q]
            self.alignment=ProgressAlignment(self.reference_q,poses)
            self.alignment_state('reset')
            self.hold = None
            self.previous_action = np.zeros(5)
            self.previous_applied = np.zeros(4)
            self.report['runtime_bindings'] = self.runtime.bindings
            self.report['status'] = 'running'
            write(self.directory/'report.json', self.report)
            return self.observation(), {'episode':self.counter}
        except BaseException as error:
            self.report.update(status='failed_initialization', error=str(error))
            write(self.directory/'report.json', self.report)
            self.close()
            raise

    def alignment_state(self, stage):
        q=np.asarray(self.runtime.bridge.state['mechanism_qpos'])
        pose=self.runtime.assembly.assemble(q,include_meshes=False)[1]['bucket']
        result=self.alignment.observe(q,pose,self.cursor)
        # Saved controller targets are reference data, not a replacement for
        # the existing native position loop. Interpolate its reference clock.
        first=min(int(self.cursor),1170);last=min(first+1,1170);fraction=self.cursor-first
        target=self.reference_targets[first]*(1-fraction)+self.reference_targets[last]*fraction
        result['wait_target_error_si']=(target-np.asarray(self.runtime.bridge.state['controller_target'])).tolist()
        result['endpoint_ready']=self.alignment.endpoint_ready(q,pose)
        if self.cursor>=1170 and not result['endpoint_ready'] and getattr(self,'hold',None) is None:
            result['mode']='wait_endpoint'
        with (self.directory/'alignment.ndjson').open('a') as log:
            log.write(json.dumps(dict(frame=self.runtime.frames,stage=stage,
                                      reference_progress=self.cursor,**result))+'\n')
        self.report['alignment_checks']+=1
        self.report['maximum_reference_lead_frames']=max(self.report['maximum_reference_lead_frames'],
                                                        self.cursor-result['actual_progress'])
        self.report['maximum_alignment_error_m']=max(self.report['maximum_alignment_error_m'],
                                                     result['position_error_m'])
        self.report['last_alignment']=result
        return result

    def observation(self):
        state = self.runtime.bridge.state
        q = np.asarray(state['mechanism_qpos'])
        phase = min(int(self.cursor),1170)
        reference = self.reference_q[phase]
        bucket = np.asarray(state['bucket_origin'])
        force = self.runtime.captured.get('wrenches',{}).get('bucket')
        wrench = np.zeros(6) if force is None else np.r_[force['force'],force['torque']]/1e5
        points = [self.runtime.fixture.state.particle_q.numpy()]
        from mixed_material_r0 import quaternion_matrix_wxyz
        for vertices,rock in zip(self.vertices,state['rocks']):
            pose=np.asarray(rock['qpos']);points.append(vertices@quaternion_matrix_wxyz(pose[3:]).T+pose[:3])
        cloud=np.concatenate(points)
        indices=np.floor((cloud[:,:2]-bucket[:2]+.8)/.2).astype(int)
        inside=((indices>=0)&(indices<8)).all(1)
        indices=indices[inside];heights=(cloud[inside,2]-bucket[2])/3
        height=np.full(64,-1.,dtype=np.float32);known=np.zeros(64,dtype=np.float32)
        flat=indices[:,1]*8+indices[:,0]
        np.maximum.at(height,flat,heights);known[np.unique(flat)]=1
        base = np.zeros(4) if self.hold is not None else self.commands[min(phase,1169)]/LIMITS_SI
        values=np.r_[q,np.asarray(state['mechanism_qvel']),q-reference,
                     bucket-[5.85,-.05,.65],state['bucket_linear_velocity_world_m_s'],
                     state['bucket_angular_velocity_world_rad_s'],wrench,base,
                     self.previous_applied/LIMITS_SI,self.previous_action,
                     self.cursor/1170,self.runtime.frames/1560,float(force is not None),height,known,
                     np.asarray(self.report['last_alignment']['wait_target_error_si'])/LIMITS_SI,
                     self.alignment.measured/1170,float(self.report['last_alignment']['mode'].startswith('wait'))]
        if values.shape!=(OBS_SIZE,) or not np.isfinite(values).all():
            raise ValueError('residual_observation_contract')
        return np.clip(values,-10,10).astype(np.float32)

    def measure(self):
        if self.estimator is None:
            from live_bucket_fill import MixedFillEstimator,load_config
            self.estimator=MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),
                        [x['volume_m3'] for x in self.runtime.records],self.vertices)
        state=self.runtime.bridge.state
        pose=self.runtime.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
        gap=self.geometry.observe(self.runtime.fixture.state.particle_q.numpy(),pose)
        score=self.estimator.measure_snapshot(dict(frame=self.runtime.frames,
                soil=self.runtime.fixture.state.particle_q.numpy(),pose=pose,
                rocks=np.asarray([x['qpos'] for x in state['rocks']])))
        self.report['scores'].append(score)
        self.report['gap_checks'].append(dict(frame=self.runtime.frames,**gap))

    def step(self, action):
        if self.runtime is None or self.report['status']!='running':
            raise ValueError('residual_step_outside_episode')
        self.budget.check()
        action=np.asarray(action,dtype=float)
        if action.shape!=(5,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('residual_action_contract')
        before=self.observation() if self.collect else None
        potential_before=float(np.float32(self.alignment.measured/1170.))
        count=0;reward=0.;done=False;truncated=False
        decision_mode=control_mode(self.cursor,self.report['last_alignment'],self.hold)
        action_invalidated=False
        handoff_enabled=self.waiting_action_mode=='error_strength_v1'
        try:
            for _ in range(3):
                self.budget.check()
                if self.runtime.frames>=self.max_frames:
                    truncated=True;reward-=2.;self.finish('timeout');break
                alignment=self.alignment_state('before')
                if not alignment['trusted']:
                    done=True;reward-=2.;self.finish('alignment_stop');break
                execution_mode=control_mode(self.cursor,alignment,self.hold)
                # Once invalidated, never revive this decision even if the mode returns.
                action_invalidated |= handoff_enabled and execution_mode!=decision_mode
                effective_action=action.copy()
                if action_invalidated:
                    effective_action=np.zeros(5)
                    if execution_mode=='wait':effective_action[:4]=-1.
                if self.hold is None:
                    base,delta,request,pace=command_at(self.commands,self.cursor,effective_action,self.cap_fraction,
                                                      reference_limit=max(self.cursor,alignment['reference_limit']),
                                                      wait_target_error=alignment['wait_target_error_si'],
                                                      waiting_action_mode=self.waiting_action_mode)
                    if pace==0:self.report['alignment_wait_frames']+=1
                else:
                    base=delta=request=np.zeros(4);pace=0.
                applied=self.runtime.projector.project(self.runtime.bridge.state['mechanism_qpos'],request)
                with (self.directory/'actions.ndjson').open('a') as log:
                    log.write(json.dumps(dict(frame=self.runtime.frames+1,cursor_before=self.cursor,
                        baseline_si=base.tolist(),residual_si=delta.tolist(),request_si=request.tolist(),
                        applied_si=applied.tolist(),pace=pace,hold_frame=self.hold,
                        alignment=alignment,waiting_action_mode=self.waiting_action_mode,
                        action_handoff_version=(ACTION_HANDOFF_VERSION if handoff_enabled else 'legacy_per_frame'),
                        policy_action=action.tolist(),effective_action=effective_action.tolist(),
                        decision_mode=decision_mode,execution_mode=execution_mode,
                        action_invalidated=bool(action_invalidated),
                        projection_reason=self.runtime.projector.reason))+'\n')
                row=self.runtime.step(applied);count+=1;self.previous_applied=applied.copy()
                check=self.ground.measure(self.runtime.bridge.state['mechanism_qpos'])
                if not check['safe']:
                    self.report['ground_failure']=check;done=True;reward-=2.;self.finish('ground_stop');break
                if self.zero_check and not np.array_equal(self.runtime.fixture.state.particle_q.numpy(),self.cached_positions[self.runtime.frames]):
                    raise ValueError('zero_env_particle_replay_changed')
                if self.hold is None:
                    self.cursor=min(1170.,self.cursor+pace)
                after_alignment=self.alignment_state('after')
                action_invalidated |= handoff_enabled and control_mode(self.cursor,after_alignment,self.hold)!=decision_mode
                if not after_alignment['trusted']:
                    done=True;reward-=2.;self.finish('alignment_stop');break
                if self.hold is None:
                    if self.cursor>=1170 and after_alignment['endpoint_ready']:
                        self.hold=0;self.measure()
                else:
                    self.hold+=1
                    if self.hold==15:self.measure()
                    if self.hold>=26:
                        state=self.runtime.bridge.state
                        pose=self.runtime.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
                        self.report['gap_checks'].append(dict(frame=self.runtime.frames,
                            **self.geometry.observe(self.runtime.fixture.state.particle_q.numpy(),pose)))
                    if self.hold==30:
                        self.measure();done=True
                        score=retained_score(self.report['scores'])
                        self.report['retained_fill_fraction']=score
                        # Ground-connected proxy gaps are diagnostics, not a hold gate.
                        self.report['success']=score is not None
                        reward+=score if self.report['success'] else -2.
                        self.finish('completed' if self.report['success'] else 'invalid_endpoint');break
            reward-=count*.000005+float(np.sum((action[:4]-self.previous_action[:4])**2))*.000001
            self.previous_action=action.copy()
            self.report['physical_frames']=self.runtime.frames
            if self.runtime.frames//60!=(self.runtime.frames-count)//60:
                write(self.directory/'report.json',self.report)
                print('RL_FRAME',self.directory.name,self.runtime.frames,'cursor',round(self.cursor,1),flush=True)
            after=self.observation()
            raw_reward=reward
            reward=progress_reward(raw_reward,potential_before,float(after[-2]),done or truncated)
            info={'terminal_score_version':TERMINAL_SCORE_VERSION,'physical_frames':count,'raw_reward':raw_reward,'potential_before':potential_before,
                  'potential_after':0. if done or truncated else float(after[-2])}
            if done or truncated:info['episode_summary']=dict(self.report,path=str(self.directory))
            if self.collect:self.transitions.append((before,after,action.astype(np.float32),reward,done or truncated,info))
            return after,float(reward),done,truncated,info
        except BaseException as error:
            self.report.update(status='fatal',error=str(error),physical_frames=self.runtime.frames)
            try:self.report['failure_checkpoint']=self.runtime.save(self.directory/'failure.npz',stage='failure')
            except Exception as save_error:self.report['failure_save_error']=str(save_error)
            write(self.directory/'report.json',self.report)
            raise

    def finish(self,status):
        self.report.update(status=status,physical_frames=self.runtime.frames,cursor=self.cursor,
                           simulation_seconds=self.runtime.frames/60,wall_seconds=time.monotonic()-self.started)
        self.report['checkpoint']=self.runtime.save(self.directory/'endpoint.npz',stage='hold' if self.report['success'] else 'failure')
        self.report['source_unchanged']=sha(INPUT)==INPUT_SHA and sha(SOURCE)==SOURCE_SHA
        write(self.directory/'report.json',self.report)
        self.completed.append(dict(self.report,path=str(self.directory)))

    def close(self):
        if self.runtime is not None:
            if self.report.get('status')=='running':
                self.report.update(status='closed_early',physical_frames=self.runtime.frames,
                                   wall_seconds=time.monotonic()-self.started)
                write(self.directory/'report.json',self.report)
            self.runtime.close();self.runtime=None
