"""Single-source causal replay of isolated loaded-port damping."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/motion-review-20261015-v1/loaded-support-20261015-v3'
OUT=ROOT/'runs/loaded-support-20261015-v3'
OLD=ROOT/'runs/motion-physical-short-20261015-v2/loaded'
sys.path[:0]=[str(WORK),str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from observe_bridge import instrument as observe
from support_candidate import instrument as damp, instrument_observation
from continuous_scoop import CycleRuntime, FixedRouteClearance, PhaseView, TransportController, manual
from planning.cycle_geometry import tool_pose, MOUTH
import layered_mound_checkpoint_runtime as generator


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')


budget=json.loads((WORK/'budget.json').read_text())
assert budget['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not (WORK/'closure.json').exists()
OUT.mkdir(exist_ok=False)
states=json.loads((OLD/'display/states.json').read_text())
cloud=np.load(OLD/'display/particles.npy',mmap_mode='r')
dispatch=[json.loads(x)['command'] for x in (OLD/'dispatch.ndjson').read_text().splitlines()]
actions={x['frame']:x for x in map(json.loads,(OLD/'actions.ndjson').read_text().splitlines())}
activation=states[0]['time_s']+3/60
source=ROOT/'runs/edge-first-two-scoop-20261015-continuation/policy-1/episode-0001/endpoint.npz'
frozen=json.loads((ROOT/'work/motion-physical-short-20261015-v2/preflight.json').read_text())['bindings_sha256']
assert all(sha(p)==h for p,h in frozen.items()) and sha(source)=='46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a'
result=dict(status='starting',physics_steps=0,native_initializations=0,training_updates=0,inference_calls=0,
            source_sha256=sha(source),activation_clock=activation,control_scope='Native neutral measured-load support with saved bounded accumulator plus dissipative velocity feedback; original commands, targets, models and guards preserved.',
            script_hashes={p.name:sha(p) for p in WORK.glob('*.py')},prefix_checks=[],metrics={},qualification=False)
write(OUT/'report.json',result)


class DampedRuntime(CycleRuntime):
    def _initialize_fixture(self,*args):
        original=generator.render_capacity_bridge
        def render(*a):
            source_code,manifest=original(*a)
            logged,recovered=observe(source_code,OUT/'substeps.csv')
            assert recovered==source_code
            changed=instrument_observation(damp(logged,activation))
            (OUT/'bridge-original.cc').write_text(source_code)
            (OUT/'bridge-candidate.cc').write_text(changed)
            return changed,manifest
        generator.render_capacity_bridge=render
        try:return super()._initialize_fixture(*args)
        finally:generator.render_capacity_bridge=original
    def step(self,command):
        if min(budget['epoch_deadline']-time.time(),budget['monotonic_deadline']-time.monotonic())<=120:raise TimeoutError('save_reserve')
        if self.frames>=887:raise ValueError('frame_ceiling')
        return super().step(command)


r=None;recorded=[];errors=[]
def metrics():
    out={}
    for phase in ('CLEAR_PILE','ALIGN'):
        selected=[x for x in errors if x['phase']==phase]
        if selected:
            current=np.array([x['pitch_error_deg'] for x in selected]);old=np.array([x['old_pitch_error_deg'] for x in selected])
            out[phase]=dict(frames=len(selected),peak_error_deg=float(np.max(abs(current))),rms_error_deg=float(np.sqrt(np.mean(current**2))),
                            baseline_peak_error_deg=float(np.max(abs(old))),baseline_rms_error_deg=float(np.sqrt(np.mean(old**2))))
    return out


try:
    with DampedRuntime(source,OUT/'physics',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
        result['native_initializations']=1
        assert len(r.capture())==24 and all(np.array_equal(v,r.arrays[k]) for k,v in r.capture().items())
        assert max(r.restored['cycle_errors'].values())<=1e-12
        assert manual.compact_state(r.bridge.state)==states[0]
        assert np.array_equal(r.fixture.state.particle_q.numpy(),cloud[0])
        recorded.append(manual.compact_state(r.bridge.state))
        view=PhaseView(r);checker=FixedRouteClearance(view);controller=TransportController(view,checker,dispatch[2])
        _,body=r.assembly.assemble(np.array(r.bridge.state['mechanism_qpos']),include_meshes=False)
        from layered_47_cycle_geometry import bucket_mask
        initial_mask=bucket_mask(r.fixture.state.particle_q.numpy(),body['bucket'])
        for i,command in enumerate(dispatch,1):
            phase=actions[i]['phase'] if i>3 else 'RESTORE_PREFIX3'
            before=manual.compact_state(r.bridge.state)
            if i>3:
                controller.check(np.asarray(before['mechanism_qpos']),phase,physical=True)
                x=actions[i];actual_pitch=tool_pose(r.assembly,np.array(before['mechanism_qpos']),MOUTH)[1]
                desired_pitch=tool_pose(r.assembly,np.array(x['desired']),MOUTH)[1]
                error=lambda a:float(np.rad2deg(np.arctan2(np.sin(a-desired_pitch),np.cos(a-desired_pitch))))
                errors.append(dict(frame=i,phase=phase,pitch_error_deg=error(actual_pitch),old_pitch_error_deg=error(x['detail']['pitch_diagnostic_rad'])))
            row=r.step(command);result['physics_steps']=r.frames
            assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']
            if i<=3:
                check=dict(frame=i,state_exact=row['after_state']==states[i],particles_exact=np.array_equal(r.fixture.state.particle_q.numpy(),cloud[i]))
                result['prefix_checks'].append(check);assert check['state_exact'] and check['particles_exact']
                if i==3:
                    _,initial_body=r.assembly.assemble(np.array(row['after_state']['mechanism_qpos']),include_meshes=False)
                    initial_mask=bucket_mask(r.fixture.state.particle_q.numpy(),initial_body['bucket'])
            else:
                controller.check(np.array(row['after_state']['mechanism_qpos']),phase,physical=True)
                assert np.max(abs(np.array(row['after_state']['mechanism_qpos'])[[7,9,12,15]]-np.array(actions[i]['desired'])[[7,9,12,15]]))<=.025
            recorded.append(row['after_state'])
            if i==120:
                m=metrics()['CLEAR_PILE'];result['short_metrics']=m
                result['short_checkpoint']=r.save(OUT/'short120.npz',stage='transport')
                if m['peak_error_deg']>.6*m['baseline_peak_error_deg'] or m['rms_error_deg']>.6*m['baseline_rms_error_deg']:
                    result['status']='short_gain_insufficient';break
            if i%120==0:result['metrics']=metrics();write(OUT/'report.json',result);print('PROGRESS',i,result['metrics'],flush=True)
        else:result['status']='loaded_887_frame_damping_completed'
        _,body=r.assembly.assemble(np.array(r.bridge.state['mechanism_qpos']),include_meshes=False)
        final_mask=bucket_mask(r.fixture.state.particle_q.numpy(),body['bucket'])
        result.update(final_checkpoint=r.save(OUT/'final.npz',stage='transport'),initial_cavity_particles=int(initial_mask.sum()),initial_cohort_still_in_cavity=int((initial_mask&final_mask).sum()),final_cavity_particles=int(final_mask.sum()))
        write(OUT/'final.npz.control-policy.json',dict(checkpoint_sha256=result['final_checkpoint']['sha256'],policy='loaded_measured_support_damping_v3',activation_clock=activation,policy_sha256=sha(WORK/'support_candidate.py'),independent_restore_verified=False))
except BaseException as exc:
    result.update(status='failed',exception=repr(exc),traceback=traceback.format_exc());raise
finally:
    result['physics_steps']=r.frames if r else 0;result['metrics']=metrics();result['finished_epoch']=time.time()
    write(OUT/'states.json',recorded);write(OUT/'pitch-errors.json',errors);write(OUT/'report.json',result)
    write(WORK/'closure.json',dict(status=result['status'],physics_steps=result['physics_steps'],finished_epoch=time.time(),remaining_seconds=max(0,budget['epoch_deadline']-time.time()),old_budget_reused=False))
    print(result['status'],result['physics_steps'],result['metrics'],flush=True)


