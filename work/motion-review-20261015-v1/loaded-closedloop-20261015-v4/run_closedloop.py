"""Real closed-loop full lift/swing; saved source and reference, no training."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/motion-review-20261015-v1/loaded-closedloop-20261015-v4'
OUT=ROOT/'runs/loaded-closedloop-20261015-v4'
OLD=ROOT/'runs/motion-physical-short-20261015-v2/loaded'
sys.path[:0]=[str(WORK),str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import CycleRuntime,FixedRouteClearance,PhaseView,TransportController,manual
import layered_47_loaded_transport_support as policy
import layered_mound_checkpoint_runtime as generator
from observe_bridge import instrument as observe
from damping_candidate import instrument_observation
from planning.cycle_geometry import tool_pose,MOUTH
from layered_47_cycle_geometry import bucket_mask


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')


protocol=json.loads((WORK/'protocol.json').read_text());budget=json.loads((WORK/'budget.json').read_text())
assert budget['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and not (WORK/'closure.json').exists()
assert all(sha(p)==h for p,h in protocol['bindings'].items())
source=ROOT/'runs/edge-first-two-scoop-20261015-continuation/policy-1/episode-0001/endpoint.npz'
assert sha(source)=='46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a'
oldstates=json.loads((OLD/'display/states.json').read_text());cloud=np.load(OLD/'display/particles.npy',mmap_mode='r')
prefix=[json.loads(x)['command'] for x in (OLD/'dispatch.ndjson').read_text().splitlines()[:3]]
oldactions={x['frame']:x for x in map(json.loads,(OLD/'actions.ndjson').read_text().splitlines())}
refs=np.load(WORK/'reference.npz');OUT.mkdir(exist_ok=False)
result=dict(status='starting',physics_steps=0,native_initializations=0,inference_calls=0,training_updates=0,prefix_checks=[],phases=[],policy=policy.POLICY,activation_clock=protocol['support_start_s'],deactivation_clock=protocol['support_end_s'],qualification=False)
recorded=[];errors=[];r=None;all_actions=[];all_dispatch=[];restore_cache={'active':[],'inactive':[]}


class ObservedRuntime(policy.LoadedTransportRuntime):
    def _initialize_fixture(self,*args):
        original=generator.render_capacity_bridge;patch=policy.instrument
        def render(*a):
            s,m=original(*a);(OUT/'bridge-original.cc').write_text(s)
            logged,back=observe(s,OUT/'substeps.csv');assert back==s
            return logged,m
        def instrument(s,start,end):
            changed=instrument_observation(patch(s,start,end));(OUT/'bridge-candidate.cc').write_text(changed)
            return changed
        generator.render_capacity_bridge=render;policy.instrument=instrument
        try:return super()._initialize_fixture(*args)
        finally:generator.render_capacity_bridge=original;policy.instrument=patch
    def step(self,command):
        if min(budget['epoch_deadline']-time.time(),budget['monotonic_deadline']-time.monotonic())<=120:raise TimeoutError('save_reserve')
        if self.frames>=protocol['max_frames']:raise ValueError('frame_ceiling')
        return super().step(command)


def persist():
    result['physics_steps']=r.frames if r else 0
    result['metrics']={}
    for phase in ['CLEAR_PILE','ALIGN','SETTLE_OFF']:
        e=[x['pitch_error_deg'] for x in errors if x['phase']==phase]
        if e:result['metrics'][phase]=dict(frames=len(e),peak_error_deg=float(np.max(np.abs(e))),rms_error_deg=float(np.sqrt(np.mean(np.array(e)**2))),last_error_deg=e[-1])
    write(OUT/'report.json',result)


try:
    with ObservedRuntime(source,OUT/'physics',support_start_s=protocol['support_start_s'],support_end_s=protocol['support_end_s'],enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
        result['native_initializations']=1
        assert len(r.capture())==24 and all(np.array_equal(v,r.arrays[k]) for k,v in r.capture().items())
        assert max(r.restored['cycle_errors'].values())<=1e-12
        assert manual.compact_state(r.bridge.state)==oldstates[0] and np.array_equal(r.fixture.state.particle_q.numpy(),cloud[0])
        recorded.append(manual.compact_state(r.bridge.state))
        def step(command,phase):
            before=manual.compact_state(r.bridge.state)
            row=r.step(command)
            assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']
            recorded.append(row['after_state']);all_dispatch.append(dict(frame=r.frames,phase=phase,command=np.asarray(command)))
            for label,first in [('active',121),('inactive',protocol['max_frames']-119)]:
                if first<=r.frames<=first+2:
                    restore_cache[label].append(dict(command=np.asarray(command),before_state=before,after_state=row['after_state']))
                    np.save(OUT/(label+'-restore-'+str(r.frames-first)+'.npy'),r.fixture.state.particle_q.numpy())
            if r.frames==120:result['active_checkpoint']=r.save(OUT/'active120.npz',stage='transport')
            if r.frames==protocol['max_frames']-120:result['inactive_checkpoint']=r.save(OUT/'inactive.npz',stage='transport')
            if r.frames%120==0:persist();print('PROGRESS',r.frames,phase,result['metrics'],flush=True)
            return row
        for command in prefix:
            row=step(command,'RESTORE_PREFIX3')
            c=dict(frame=r.frames,state_exact=row['after_state']==oldstates[r.frames],particles_exact=np.array_equal(r.fixture.state.particle_q.numpy(),cloud[r.frames]));result['prefix_checks'].append(c);assert c['state_exact'] and c['particles_exact']
        _,body=r.assembly.assemble(np.asarray(r.bridge.state['mechanism_qpos']),include_meshes=False)
        initial_mask=bucket_mask(r.fixture.state.particle_q.numpy(),body['bucket'])
        checker=FixedRouteClearance(PhaseView(r));controller=TransportController(PhaseView(r),checker,prefix[-1])
        for phase,q,v,holds in [('CLEAR_PILE',refs['CLEAR_PILE_qpos'],refs['CLEAR_PILE_velocity'],30),('ALIGN',refs['ALIGN_qpos'],refs['ALIGN_velocity'],60),('SETTLE_OFF',refs['ALIGN_qpos'][-1:],np.empty((0,4)),120)]:
            begin=r.frames
            for desired,velocity in list(zip(q[1:],v,strict=True))+[(q[-1],np.zeros(4))]*holds:
                before=np.asarray(r.bridge.state['mechanism_qpos']);physical_phase='ALIGN' if phase=='SETTLE_OFF' else phase
                controller.check(before,physical_phase,physical=True)
                command,detail=controller.command(physical_phase,before,r.bridge.state['controller_target'],desired,velocity)
                assert command[3]==0
                frame=r.frames+1
                actual_pitch=tool_pose(r.assembly,before,MOUTH)[1];desired_pitch=tool_pose(r.assembly,desired,MOUTH)[1]
                error=float(np.rad2deg(np.arctan2(np.sin(actual_pitch-desired_pitch),np.cos(actual_pitch-desired_pitch))))
                errors.append(dict(frame=frame,phase=phase,pitch_error_deg=error))
                all_actions.append(dict(frame=frame,phase=phase,desired=desired,velocity=velocity,command=command,detail=detail))
                row=step(command,phase);actual=np.asarray(row['after_state']['mechanism_qpos'])
                controller.check(actual,physical_phase,physical=True)
                assert np.max(abs(actual[r.chain.axes]-desired[r.chain.axes]))<=.025
            endpoint=float(np.max(abs(np.asarray(r.bridge.state['mechanism_qpos'])[r.chain.axes]-q[-1][r.chain.axes])))
            if endpoint>.015:raise ValueError('endpoint15mm')
            result['phases'].append(dict(name=phase,start_frame=begin,end_frame=r.frames,endpoint_error=endpoint))
        _,body=r.assembly.assemble(np.asarray(r.bridge.state['mechanism_qpos']),include_meshes=False);final_mask=bucket_mask(r.fixture.state.particle_q.numpy(),body['bucket'])
        result.update(status='full_closedloop_lift_swing_completed',initial_cavity_particles=int(initial_mask.sum()),initial_cohort_still_in_cavity=int((initial_mask&final_mask).sum()),final_cavity_particles=int(final_mask.sum()),final_checkpoint=r.save(OUT/'final.npz',stage='transport'))
except BaseException as exc:
    result.update(status='failed',exception=repr(exc),traceback=traceback.format_exc());raise
finally:
    persist();write(OUT/'states.json',recorded);write(OUT/'pitch-errors.json',errors);write(OUT/'actions.json',all_actions);write(OUT/'dispatch.json',all_dispatch);write(OUT/'restore-cache.json',restore_cache)
    write(WORK/'closure.json',dict(status=result['status'],physics_steps=result['physics_steps'],finished_epoch=time.time(),remaining_seconds=max(0,budget['epoch_deadline']-time.time()),old_budget_reused=False));print(result['status'],result['physics_steps'],flush=True)
