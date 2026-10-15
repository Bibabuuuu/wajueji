"""Bounded paired native continuation, identical geometric withdrawal, 5s hold."""
from exit_common import *
from layered_47_cycle_bridge import controller_snapshot
from view_contact_bc_probe import ProbeDisplayCache
import argparse,time,traceback

class Budget:
    def __init__(self,path):
        self.path=Path(path);self.data=read(path);self.total=0
        d=self.data
        if Path(str(path)+'.closed.json').exists():raise ValueError('budget_closed')
        if d['purpose']!='d165_exit_settle_pair' or d['status']!='authorized' or d['seconds']!=2700 or d['training_updates_allowed']!=0:
            raise ValueError('new_scope_budget_required')
        if abs(d['epoch_deadline']-d['epoch_start']-2700)>1e-6 or d['max_physical_frames']!=5160:
            raise ValueError('budget_identity')
    def check(self):
        if time.time()+self.data['reserve_seconds']>=self.data['epoch_deadline']:raise TimeoutError('budget_reserve_reached')

class BoundRuntime(CycleRuntime):
    def __init__(self,info,output,budget,limit):
        self.budget=budget;self.limit=limit;self.authorized=False;budget.check()
        if sha(info['source'])!=info['sha256'] or sha(info['source']+'.json')!=info['metadata_sha256'] or sha(info['source']+'.control.json')!=info['control_sha256']:
            raise ValueError('changed_source')
        super().__init__(info['source'],output,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
    def step(self,command):
        self.budget.check()
        if not self.authorized:raise ValueError('zero_restore_required')
        if self.frames>=self.limit or self.budget.total>=self.budget.data['max_physical_frames']:raise ValueError('physical_frame_ceiling')
        before=self.frames
        try:return super().step(command)
        finally:self.budget.total+=self.frames-before

def zero_gate(r):
    actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()}
    errors=r.restored['cycle_errors'];sweep=r.sweep.diagnostic_state()==r.metadata['moving_bucket_sweep_state']
    if len(exact)!=24 or not all(exact.values()) or max(errors.values())>1e-12 or not sweep:raise ValueError('zero_restore_inexact')
    r.authorized=True
    return dict(arrays_exact=exact,native_errors=errors,sweep_exact=sweep)

def array_digest(arrays):
    return dict(hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items() if k!='simulation_time_s'},simulation_time_s=float(arrays['simulation_time_s'][0]))

def compare_capture(actual,expected):
    a=array_digest(actual)
    return a['hashes']==expected['hashes'] and abs(a['simulation_time_s']-expected['simulation_time_s'])<1e-10

def save_control(r,c,path,phase):
    saved=r.save(path,stage='unloaded')
    write(str(path)+'.control.json',dict(previous_command=c.previous.tolist(),load_bias=c.tracking.bias.tolist(),first=c.first,
        checkpoint_sha256=saved['sha256'],phase=phase,continuation_qualified=False))
    return saved

def commands(name,plan):
    paths=np.load(CPU/(name+'-plan.npz'))
    for stage in ['CLEAR_TRUCK','WITHDRAW_SWING']:
        qs,vs=paths[stage+'_qpos'],paths[stage+'_velocity']
        for index,(q,v) in enumerate(zip(qs[1:],vs,strict=True),1):yield stage,index,q,v
        for index in range(30):yield stage,len(vs)+index+1,qs[-1],np.zeros(4)
    for index in range(300):yield 'SETTLE',index+1,qs[-1],np.zeros(4)

def advance(r,c,phase,desired,velocity,path):
    before=manual.compact_state(r.bridge.state);q=np.array(before['mechanism_qpos']);c.check(q,phase,physical=True)
    cmd,detail=c.command(phase,q,before['controller_target'],desired,velocity)
    manual.append_json(path/'actions.ndjson',dict(frame=r.frames+1,phase=phase,before_state=before,
        desired=desired.tolist(),velocity=velocity.tolist(),command=cmd.tolist(),detail=detail))
    row=r.step(cmd)
    if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
    measured=c.check(np.array(row['after_state']['mechanism_qpos']),phase,physical=True)
    error=float(np.max(abs(np.array(row['after_state']['mechanism_qpos'])[r.chain.axes]-desired[r.chain.axes])))
    manual.append_json(path/'clearance.ndjson',dict(frame=r.frames,phase=phase,axis_error=error,**measured))
    if error>.025:raise ValueError('original_tracking_guard:'+str(error))
    return row

def prefix_gate(name,case,budget,out):
    dest=out/'prefix-continuity';dest.mkdir();info=case['prefix_source'];state=read(info['source']+'.control.json')
    result=dict(status='checking',source=info,frames=0,reference_steps=[]);r=None;c=None
    try:
        with BoundRuntime(info,dest/'physics',budget,180) as r:
            result['zero']=zero_gate(r);c=controller(r,state)
            old=[json.loads(s) for s in (PREVIOUS/name/'actions.ndjson').read_text().splitlines()][2520:]
            assert len(old)==177
            for a in old:
                row=advance(r,c,a['phase'],np.array(a['desired']),np.array(a['velocity']),dest)
                if not np.array_equal(row['applied_command_si'],a['command']):raise ValueError('replayed_command_not_exact')
            with np.load(case['source']['source']) as data:
                expected={k:data[k] for k in data.files};exact={k:np.array_equal(v,expected[k]) for k,v in r.capture().items()}
                time_error=abs(float(r.capture()['simulation_time_s'][0])-float(data['simulation_time_s'][0]))
            native=read(case['source']['source']+'.json')
            result.update(original_final_arrays_exact=exact,original_final_time_error_s=time_error,
                original_final_native_exact=controller_snapshot(r.bridge.state)==native['controller_state'],
                original_final_sweep_exact=r.sweep.diagnostic_state()==native['moving_bucket_sweep_state'])
            expected_control=read(case['source']['source']+'.control.json')
            result['high_controller_exact']=np.array_equal(c.previous,expected_control['previous_command']) and np.array_equal(c.tracking.bias,expected_control['load_bias'])
            if not all(v for k,v in exact.items() if k!='simulation_time_s') or time_error>=1e-10 or not result['original_final_native_exact'] or not result['original_final_sweep_exact'] or not result['high_controller_exact']:
                raise ValueError('prefix_does_not_reproduce_original_final')
            for phase,index,q,v in list(commands(name,case))[:3]:
                advance(r,c,phase,q,v,dest)
                result['reference_steps'].append(dict(arrays=array_digest(r.capture()),native=controller_snapshot(r.bridge.state),
                    sweep=r.sweep.diagnostic_state(),previous_command=c.previous.tolist(),load_bias=c.tracking.bias.tolist()))
            result['reference_checkpoint']=save_control(r,c,dest/'reference-3.npz','CLEAR_TRUCK')
            result.update(status='passed',frames=r.frames)
    except BaseException as e:
        result.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),frames=0 if r is None else r.frames)
        write(dest/'report.json',result);raise
    write(dest/'report.json',result);return result

def run_case(name,case,budget,root):
    out=root/name;out.mkdir();result=dict(status='preparing',source=case['source'],frames=0,stages=[],training_updates=0,formal_bulk_volume_qualified=False)
    r=None;c=None;cache=None;phase='INITIALIZE';started=time.monotonic()
    def persist():write(out/'report.json',result)
    persist()
    try:
        gate=prefix_gate(name,case,budget,out);result['prefix_gate_report_sha256']=sha(out/'prefix-continuity/report.json');persist()
        with BoundRuntime(case['source'],out/'physics',budget,2400) as r:
            try:
                result['zero']=zero_gate(r);c=controller(r,read(case['source']['source']+'.control.json'))
                result['zero_checkpoint']=save_control(r,c,out/'zero.npz',phase)
                mass=r.fixture.model.particle_mass.numpy().copy();np.save(out/'particle-mass.npy',mass)
                result['soil_mass_inventory']=dict(count=len(mass),total_kg=float(mass.sum(dtype=np.float64)),minimum_kg=float(mass.min()),maximum_kg=float(mass.max()),sha256=sha(out/'particle-mass.npy'))
                cache=ProbeDisplayCache(out/'display',case['frames'],r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());result['status']='running';persist()
                previous_stage=None
                for phase,index,desired,velocity in commands(name,case):
                    if phase!=previous_stage:
                        if previous_stage is not None:
                            result['stages'].append(dict(name=previous_stage,frames=r.frames,checkpoint=save_control(r,c,out/(previous_stage.lower()+'.npz'),previous_stage)))
                        previous_stage=phase;persist()
                    row=advance(r,c,phase,desired,velocity,out);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy());result['frames']=r.frames
                    if r.frames<=3:
                        expected=gate['reference_steps'][r.frames-1]
                        passed=compare_capture(r.capture(),expected['arrays']) and controller_snapshot(r.bridge.state)==expected['native'] and r.sweep.diagnostic_state()==expected['sweep'] and np.array_equal(c.previous,expected['previous_command']) and np.array_equal(c.tracking.bias,expected['load_bias'])
                        result.setdefault('restored_next_steps_exact',[]).append(bool(passed))
                        if not passed:raise ValueError('restored_withdraw_step_discontinuous')
                    if r.frames==120:result['short_gate_checkpoint']=save_control(r,c,out/'short-120.npz',phase);result['short_gate_passed']=True;persist()
                    if phase=='SETTLE' and index==1:
                        gap=bucket_opening_gap(r,np.array(row['after_state']['mechanism_qpos']));result['withdrawn_opening_gap_m']=gap
                        if gap<=.1:raise ValueError('bucket_not_withdrawn')
                    if r.frames%360==0:result['periodic_checkpoint']=save_control(r,c,out/('periodic-%05d.npz'%r.frames),phase);persist()
                    if r.frames%120==0:print(name,phase,r.frames,'physical_total',budget.total,flush=True)
                error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-desired[r.chain.axes])))
                if error>.015:raise ValueError('original_endpoint_tracking_guard')
                result['stages'].append(dict(name=phase,frames=r.frames,checkpoint=save_control(r,c,out/(phase.lower()+'.npz'),phase)))
                result.update(status='withdraw_and_settle_completed',settle_frames=300)
            except BaseException as e:
                result.update(status='failed',failure=repr(e),traceback=traceback.format_exc());raise
            finally:
                result['frames']=r.frames
                try:result['final_checkpoint']=save_control(r,c,out/('final.npz' if result['status']=='withdraw_and_settle_completed' else 'recovery.npz'),phase)
                except Exception as e:result['save_error']=repr(e)
                persist()
    except BaseException as e:
        result.update(status='failed',failure=repr(e));persist()
    finally:
        if cache is not None:
            try:
                d=cache.finish(result['status']);d.update(capture_origin='native_exit_settle_pair',label='D165 WITHDRAW + SETTLE / '+name);write(out/'display/reconstruction.json',d);result['display']=d
            except Exception as e:result['display_error']=repr(e)
        result['wall_seconds']=time.monotonic()-started;persist()
    return result

def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'budget.json';now=time.time();good=dict(purpose='d165_exit_settle_pair',status='authorized',seconds=2700,epoch_start=now,epoch_deadline=now+2700,reserve_seconds=300,training_updates_allowed=0,max_physical_frames=5160)
        write(p,good);Budget(p).check();write(str(p)+'.closed.json',{})
        try:Budget(p)
        except ValueError as e:assert str(e)=='budget_closed'
        else:raise AssertionError('closed budget reused')
        r=BoundRuntime.__new__(BoundRuntime);r.budget=type('B',(),{'check':lambda self:None})();r.authorized=False
        try:r.step(np.zeros(4))
        except ValueError as e:assert str(e)=='zero_restore_required'
        else:raise AssertionError('zero gate bypass')
    a={'x':np.array([1.]),'simulation_time_s':np.array([2.])};assert compare_capture(a,array_digest(a));a['x'][0]=3;assert not compare_capture(a,array_digest({'x':np.array([1.]),'simulation_time_s':np.array([2.])}))
    print('CPU runner checks passed; native_initializations=0 physical_steps=0')

def main():
    p=argparse.ArgumentParser();p.add_argument('--budget',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check();return
    budget=Budget(a.budget);budget.check();cpu=read(CPU/'report.json');assert cpu['status']=='cpu_passed'
    for name,digest in read(CPU/'dependency-sha256.json').items():
        if sha(name)!=digest:raise ValueError('CPU_dependency_changed:'+name)
    for name,case in cpu['cases'].items():
        if not all(case['checks'].values()) or sha(CPU/(name+'-plan.npz'))!=case['plan_sha256']:raise ValueError('CPU_case_changed')
    a.output.mkdir(exist_ok=False);report=dict(status='running',cases={},training_updates=0,physical_frames=0,budget_sha256=sha(a.budget),runner_sha256=sha(__file__),cpu_sha256=sha(CPU/'report.json'))
    try:
        for name,case in cpu['cases'].items():
            report['cases'][name]=run_case(name,case,budget,a.output);report['physical_frames']=budget.total;write(a.output/'report.json',report)
            if report['cases'][name]['status']!='withdraw_and_settle_completed':raise ValueError('case_failed:'+name)
        report['status']='completed'
    except BaseException as e:report.update(status='failed',failure=repr(e),physical_frames=budget.total)
    finally:write(a.output/'report.json',report);print(encode({'status':report['status'],'physical_frames':budget.total,'training_updates':0}),flush=True)

if __name__=='__main__':main()
