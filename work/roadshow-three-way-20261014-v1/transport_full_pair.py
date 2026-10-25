"""Complete source-bound CLEAR_PILE/ALIGN/DUMP pair; requires a new full-pair budget."""
import argparse,copy,hashlib,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
CPU=WORK/'transport-endpoint-final-cpu-v1'
SOURCE_SHA={
    'better_manual':'cae7f9e1502e7c7c08ee663ed8170976cad01cbaf4140f9f8c159baf9c528701',
    'frozen_residual':'8600050504489dfb06afa97d086d3d1cbbe36ec4f4205b4f73728ebfa3cbc2b7',
}
sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT/'work/contact-second-fixed-route-20260925-v1'),str(ROOT)]
from layered_47_cycle_runtime import CycleRuntime
from controller import TransportController
from clearance import FixedRouteClearance
from layered_47_cycle_geometry import bucket_mask
from view_contact_bc_probe import ProbeDisplayCache
from terrain.truck_receiver import TruckReceiver,CAVITY
import run_layered_47_sparse_manual_contact_episode as manual

def encode(x):return json.dumps(manual._clean(x),default=lambda v:v.item(),ensure_ascii=False,indent=2,allow_nan=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(encode(value)+'\n')

class Budget:
    def __init__(self,path):
        self.path=Path(path);self.data=json.loads(self.path.read_text())
        d=self.data
        if self.path.with_suffix(self.path.suffix+'.closed.json').exists():raise ValueError('budget_already_closed')
        if d['purpose']!='d165_transport_full_pair' or d['status']!='authorized':raise ValueError('new_budget_required')
        if d['seconds'] not in (1800,2700) or abs(d['epoch_deadline']-d['epoch_start']-d['seconds'])>1e-6:raise ValueError('fixed_new_budget_required')
        expected=2697
        if d['per_case_max_frames']!=expected or d['training_updates_allowed']!=0:raise ValueError('budget_scope_mismatch')
    def check(self):
        if time.time()+self.data['reserve_seconds']>=self.data['epoch_deadline']:raise TimeoutError('fixed_budget_reserve_reached')
    @property
    def frames(self):return self.data['per_case_max_frames']

class BoundRuntime(CycleRuntime):
    def __init__(self,source,output,identity,budget):
        budget.check()
        if sha(source)!=identity['sha256'] or sha(str(source)+'.json')!=identity['metadata_sha256']:
            raise ValueError('source_identity_changed')
        self.budget=budget;self.step_authorized=False
        super().__init__(source,output,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
        if self.metadata['stage']!='hold' or self.metadata['cycle_index']!=1 or self.metadata.get('training_reset'):
            raise ValueError('original_hold_source_required')
    def __enter__(self):
        self.budget.check();return super().__enter__()
    def step(self,command):
        if not self.step_authorized:raise ValueError('zero_restore_must_pass_before_step')
        self.budget.check()
        if self.frames>=self.budget.frames:raise ValueError('physical_frame_ceiling')
        return super().step(command)

def self_check():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir=WORK) as temp:
        path=Path(temp)/'budget.json';now=time.time()
        good=dict(purpose='d165_transport_full_pair',status='authorized',seconds=1800,
            epoch_start=now,epoch_deadline=now+1800,reserve_seconds=180,per_case_max_frames=2697,training_updates_allowed=0)
        write(path,good);budget=Budget(path);budget.check();assert budget.frames==2697
        write(path,dict(good,per_case_max_frames=120))
        try:Budget(path)
        except ValueError as e:assert str(e)=='budget_scope_mismatch'
        else:raise AssertionError('scope expansion accepted')
        write(path,dict(good,epoch_start=now-1800,epoch_deadline=now))
        try:Budget(path).check()
        except TimeoutError:pass
        else:raise AssertionError('expired budget accepted')
        runtime=BoundRuntime.__new__(BoundRuntime);runtime.step_authorized=False
        try:runtime.step(np.zeros(4))
        except ValueError as e:assert str(e)=='zero_restore_must_pass_before_step'
        else:raise AssertionError('unrestored step accepted')
        write(path,good);budget=Budget(path)
        source=Path(temp)/'source.npz';source.write_bytes(b'not_an_approved_source')
        try:BoundRuntime(source,Path(temp)/'must_not_exist',{'sha256':'0'*64,'metadata_sha256':'0'*64},budget)
        except ValueError as e:assert str(e)=='source_identity_changed'
        else:raise AssertionError('changed source accepted')
        assert not (Path(temp)/'must_not_exist').exists()
    print('CPU guards passed; native_initializations=0 physical_steps=0')

def validate_inputs():
    audit=json.loads((CPU/'read-only-audit.json').read_text())
    if audit['status']!='passed' or not all(audit['checks'].values()):raise ValueError('cpu_gate_not_passed')
    for relative,digest in json.loads((CPU/'manifest.json').read_text()).items():
        if sha(CPU/relative)!=digest:raise ValueError('cpu_artifact_changed:'+relative)
    for path,digest in json.loads((CPU/'dependency-sha256.json').read_text()).items():
        if sha(path)!=digest:raise ValueError('original_dependency_changed:'+path)
    identities=json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources']
    for name,expected in SOURCE_SHA.items():
        identity=identities[name];plan=json.loads((CPU/(name+'-plan/report.json')).read_text())
        if plan['source']!=identity['source'] or identity['sha256']!=expected or sha(identity['source'])!=expected:
            raise ValueError('source_does_not_match_cpu_plan:'+name)
        if sha(identity['source']+'.json')!=identity['metadata_sha256']:raise ValueError('source_metadata_changed:'+name)
    short=ROOT/'runs/d165-transport-native-probe-20261014-v1'
    verified=json.loads((short/'read-only-audit.json').read_text())
    if verified['status']!='passed' or not all(verified['checks'].values()):raise ValueError('native_short_gate_not_passed')
    if json.loads((short/'closure.json').read_text())['status']!='closed_early':raise ValueError('prior_budget_not_sealed')
    return identities

def run_case(name,identity,budget,root):
    budget.check();out=root/name;out.mkdir()
    source=Path(identity['source']);plan_dir=CPU/(name+'-plan')
    plan=json.loads((plan_dir/'report.json').read_text());paths=np.load(plan_dir/'plan.npz')
    report=dict(status='initializing',source=identity,frames=0,zero_restore_passed=False,stages=[],
        max_physical_frames=budget.frames,training_updates=0,formal_mixed_volume_qualified=False,
        particle_terrain_check_enabled=False,cpu_plan_sha256=sha(plan_dir/'plan.npz'))
    started=time.monotonic();cache=None;control=None;r=None;phase='INITIALIZE';step_index=0;fatal=None
    def persist():write(out/'report.json',report)
    def save(label,stage):
        result=r.save(out/label,stage=stage)
        write(out/(label+'.control.json'),dict(phase=phase,stage_step=step_index,frame=r.frames,
            previous_command=None if control is None else control.previous.tolist(),
            load_bias=None if control is None else control.tracking.bias.tolist(),
            source_sha256=identity['sha256'],checkpoint_sha256=result['sha256'],continuation_qualified=False))
        return result
    persist()
    try:
        with BoundRuntime(source,out/'physics',identity,budget) as r:
            try:
                exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()}
                errors=r.restored['cycle_errors']
                sweep_exact=r.sweep.diagnostic_state()==r.metadata['moving_bucket_sweep_state']
                report.update(zero_arrays_exact=exact,native_restore_errors=errors,sweep_restore_exact=sweep_exact)
                if len(exact)!=24 or not all(exact.values()) or max(errors.values())>1e-12 or not sweep_exact:
                    raise ValueError('native_zero_restore_inexact')
                report['zero_restore_passed']=True;report['zero_checkpoint']=save('zero.npz','hold');persist()
                actual=json.loads((source.parent/'actions.ndjson').read_text().splitlines()[-1])
                previous=np.array(actual['applied_si'])
                if actual['hold_frame']!=29 or np.any(previous):raise ValueError('handoff_actual_command_not_held_zero')
                checker=FixedRouteClearance(r);control=TransportController(r,checker,previous)
                report['initial_controller_bias']=control.tracking.bias.tolist()
                pose=r.assembly.assemble(r.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
                payload=bucket_mask(r.arrays['particle_q'],pose)
                receiver=TruckReceiver(json.loads((ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text()))
                cache=ProbeDisplayCache(out/'display',budget.frames,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
                r.step_authorized=True;report['status']='running';persist()
                def advance(desired,velocity):
                    budget.check();q=np.array(r.bridge.state['mechanism_qpos']);control.check(q,phase,physical=True)
                    command,detail=control.command(phase,q,r.bridge.state['controller_target'],desired,velocity)
                    manual.append_json(out/'actions.ndjson',dict(frame=r.frames+1,phase=phase,stage_step=step_index,
                        before_state=manual.compact_state(r.bridge.state),desired=desired.tolist(),velocity=velocity.tolist(),command=command.tolist(),detail=detail))
                    row=r.step(command);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy())
                    report.update(frames=r.frames,phase=phase,stage_step=step_index)
                    if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
                    q=np.array(row['after_state']['mechanism_qpos']);measured=control.check(q,phase,physical=True)
                    error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
                    manual.append_json(out/'clearance.ndjson',dict(frame=r.frames,phase=phase,axis_error=error,**measured))
                    if error>.025:raise ValueError('fixed_route_tracking_error:'+str(error))
                    if r.frames==120:
                        report['short_gate_checkpoint']=save('short-120.npz','transport');report['short_gate_passed']=True;persist()
                    if r.frames%360==0:
                        report['periodic_checkpoint']=save('periodic-%05d.npz'%r.frames,'transport');persist()
                    if r.frames%120==0:print(name,phase,r.frames,flush=True)
                for segment in plan['stages']:
                    phase=segment['name'];qs=paths[phase+'_qpos'];vs=paths[phase+'_velocity'];goal=qs[-1]
                    commands=list(zip(qs[1:],vs,strict=True))+[(goal,np.zeros(4))]*(30+(180 if phase=='DUMP' else 0))
                    for step_index,(desired,velocity) in enumerate(commands,1):
                        if r.frames>=budget.frames:break
                        advance(desired,velocity)
                    if r.frames>=budget.frames and budget.frames==120:
                        report['status']='short_probe_passed';break
                    error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-goal[r.chain.axes])))
                    if error>.015:raise ValueError('fixed_route_stage_endpoint:'+phase)
                    if phase=='CLEAR_PILE':control.check(np.array(r.bridge.state['mechanism_qpos']),'ALIGN',physical=True)
                    if phase=='DUMP':
                        particles=r.fixture.state.particle_q.numpy();pose=r.assembly.assemble(np.array(r.bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
                        remaining=payload&bucket_mask(particles,pose);received=payload&(receiver.classify(particles)==CAVITY)
                        report['unload_inventory']=dict(source_particles=int(payload.sum()),received_source_particles=int(received.sum()),remaining_source_particles=int(remaining.sum()),scope='Development soil centers only; not stable mixed net volume')
                        if remaining.sum()>max(50,.1*payload.sum()) or received.sum()<.5*payload.sum():raise ValueError('original_development_unload_gate')
                    report['stages'].append(dict(name=phase,frames=r.frames,checkpoint=save(phase.lower()+'.npz','unloaded' if phase=='DUMP' else 'transport')));persist()
                else:report['status']='transport_unload_development_passed'
            except TimeoutError as e:report.update(status='budget_stopped',failure=repr(e))
            except BaseException as e:report.update(status='failed',failure=repr(e));fatal=e
            finally:
                report['frames']=r.frames
                try:report['final_checkpoint']=save('final.npz' if report['status']!='failed' else 'recovery.npz','failure' if report['status']=='failed' else 'unloaded' if report['status']=='transport_unload_development_passed' else 'transport')
                except Exception as e:report['save_error']=repr(e)
                persist()
    except BaseException as e:
        report.update(status='failed',failure=repr(e));fatal=e
    finally:
        if cache is not None:
            try:
                display=cache.finish(report['status'])
                display.update(capture_origin='native_transport_full_pair',label='D165 TRANSPORT FULL PAIR / '+name)
                write(out/'display/reconstruction.json',display);report['display']=display
            except Exception as e:report['display_error']=repr(e)
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(source)==identity['sha256']);persist();paths.close()
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true');parser.add_argument('--budget');parser.add_argument('--output');args=parser.parse_args()
    if args.self_check:self_check();return
    if not args.budget or not args.output:parser.error('Explicit new budget and new output directory required')
    budget=Budget(args.budget);identities=validate_inputs();budget.check()
    root=Path(args.output);root.mkdir(exist_ok=False)
    batch=dict(status='running',budget=dict(path=args.budget,sha256=sha(args.budget),**budget.data),cases={},training_updates=0,runner_sha256=sha(__file__))
    write(root/'report.json',batch)
    for name in ['better_manual','frozen_residual']:
        try:budget.check()
        except TimeoutError as e:batch.update(status='budget_stopped',reason=repr(e));break
        case=run_case(name,identities[name],budget,root);batch['cases'][name]=case
        write(root/'report.json',batch)
        if case['status'] not in ('short_probe_passed','transport_unload_development_passed'):
            batch['status']=case['status'];break
    else:batch['status']='completed'
    batch['physical_frames']=sum(x['frames'] for x in batch['cases'].values());write(root/'report.json',batch)
    print(encode({k:batch[k] for k in ['status','physical_frames','training_updates']}),flush=True)

if __name__=='__main__':main()
