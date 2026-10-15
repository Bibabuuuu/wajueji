"""One native instance: safe setup, approach, frozen residual dig, unload, return."""
from continuous_scoop import *
from view_contact_bc_probe import ProbeDisplayCache
from stable_baselines3 import SAC
import argparse,traceback,torch

class Budget:
    def __init__(self,path):
        self.path=Path(path);self.data=read(path);d=self.data
        if Path(str(path)+'.closed.json').exists():raise ValueError('closed_budget')
        if d['purpose']!='continuous_single_scoop' or d['seconds']!=5400 or d['max_frames']!=15000 or d['training_updates_allowed']!=0:
            raise ValueError('explicit_new_scope_required')
        if abs(d['epoch_deadline']-d['epoch_start']-5400)>1e-6:raise ValueError('fixed_budget_identity')
    def start(self):self.check()
    def remaining(self):return self.data['epoch_deadline']-time.time()
    def check(self):
        if self.remaining()<=self.data['reserve_seconds']:raise TimeoutError('physical_budget_reserve_reached')

class BoundRuntime(CycleRuntime):
    def __init__(self,output,budget):
        budget.check();self.budget=budget;self.step_authorized=False;self.native_initializations=0
        if sha(SOURCE)!=SOURCE_SHA:raise ValueError('initial_source_changed')
        super().__init__(SOURCE,output,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
    def __enter__(self):
        if self.native_initializations:raise RuntimeError('second_native_initialization_forbidden')
        self.native_initializations+=1;return super().__enter__()
    def step(self,command):
        self.budget.check()
        if not self.step_authorized:raise ValueError('native_zero_gate_required')
        if self.frames>=self.budget.data['max_frames']:raise ValueError('whole_episode_frame_ceiling')
        return super().step(command)

def digest(r):
    a=r.capture()
    return dict(arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()},
        native=sha_json(controller_snapshot(r.bridge.state)),sweep=sha_json(r.sweep.diagnostic_state()),
        global_frame=r.frames,simulation_time_s=float(a['simulation_time_s'][0]),fixture_id=id(r.fixture),bridge_id=id(r.bridge))

def sha_json(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--budget',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    budget=Budget(a.budget);budget.check();cpu=read(W/'cpu-v2/report.json');assert cpu['status']=='cpu_passed'
    protocol=read(W/'protocol.json')
    for path,value in protocol['code_sha256'].items():
        if sha(path)!=value:raise ValueError('frozen_code_changed:'+path)
    if sha(MODEL)!=MODEL_SHA or sha(RECEIPT)!=protocol['receipt_sha256']:raise ValueError('model_or_contract_changed')
    require_environment_contract(read(RECEIPT),'error_strength_v1');torch.set_num_threads(1)
    model=SAC.load(MODEL,device='cpu');updates=model._n_updates
    out=a.output;out.mkdir(exist_ok=False);started=time.monotonic();report=dict(status='initializing',frames=0,native_initializations=0,phases=[],handoffs=[],
        model_sha256=MODEL_SHA,neural_updates=0,source_sha256=SOURCE_SHA,source_metadata_sha256=sha(str(SOURCE)+'.json'),budget_sha256=sha(a.budget),
        protocol_sha256=sha(W/'protocol.json'),cpu_report_sha256=sha(W/'cpu-v2/report.json'),episode_scope='One current fixed-target scoop, including real safe-start preparation and approach; not3 scoops or formal competition acceptance')
    r=None;cache=None;environment=None;phase='INITIALIZE';previous=np.zeros(4);frame_phases=[]
    def persist():write(out/'report.json',report)
    def save(name,stage):
        result=r.save(out/(name+'.npz'),stage=stage)
        write(out/(name+'.control.json'),dict(global_frame=r.frames,phase=phase,previous_command=previous.tolist(),checkpoint_sha256=result['sha256'],continuation_qualified=False))
        return result
    def handoff(label,factory):
        before=digest(r);value=factory();after=digest(r);exact=before==after
        report['handoffs'].append(dict(label=label,before=before,after=after,physical_and_native_unchanged=exact));persist()
        if not exact:raise ValueError('handoff_mutated_physics:'+label)
        return value
    def geometry(name,path,checker,hold=30):
        nonlocal phase
        phase=name;begin=r.frames
        c=handoff(name,lambda:TransportController(PhaseView(r),checker,previous))
        for desired,velocity in list(zip(path['qpos'][1:],path['velocity'],strict=True))+[(path['qpos'][-1],np.zeros(4))]*hold:
            q=np.array(r.bridge.state['mechanism_qpos']);c.check(q,name,physical=True)
            command,detail=c.command(name,q,r.bridge.state['controller_target'],desired,velocity)
            manual.append_json(out/'geometry-actions.ndjson',dict(global_frame=r.frames+1,phase=name,before_state=manual.compact_state(r.bridge.state),desired=desired.tolist(),velocity=velocity.tolist(),command=command.tolist(),detail=detail))
            row=r.step(command);q=np.array(row['after_state']['mechanism_qpos']);measured=c.check(q,name,physical=True)
            error=float(np.max(abs(q[r.chain.axes]-desired[r.chain.axes])))
            manual.append_json(out/'geometry-clearance.ndjson',dict(global_frame=r.frames,phase=name,axis_error=error,**measured))
            if error>.025:raise ValueError('original_tracking_guard:'+str(error))
        # A genuine zero-command stop hands the native target/velocity over intact.
        for _ in range(30):
            row=r.step(np.zeros(4));c.check(np.array(row['after_state']['mechanism_qpos']),name,physical=True)
        error=float(np.max(abs(np.array(r.bridge.state['mechanism_qpos'])[r.chain.axes]-path['qpos'][-1][r.chain.axes])))
        if error>.015:raise ValueError('original_endpoint_guard:'+name)
        checkpoint=save(name.lower(),'returned' if name=='RETURN_SAFE' else 'prepoint' if name=='APPROACH_PREPOINT' else 'unloaded' if name in ('DUMP','CLEAR_TRUCK','WITHDRAW_SWING') else 'transport')
        report['phases'].append(dict(name=name,start_frame=begin,end_frame=r.frames,endpoint_axis_error=error,checkpoint=checkpoint));persist()
    persist()
    try:
        with BoundRuntime(out/'physics',budget) as r:
            try:
                actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()};time_error=abs(float(actual['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0]))
                if not all(v for k,v in exact.items() if k!='simulation_time_s') or time_error>1e-12 or max(r.restored['cycle_errors'].values())>1e-12:
                    raise ValueError('original_zero_restore_inexact')
                report.update(zero_arrays_exact=exact,zero_time_error_s=time_error,native_restore_errors=r.restored['cycle_errors'],native_initializations=r.native_initializations,
                    initial_simulation_time_s=float(actual['simulation_time_s'][0]),zero_checkpoint=save('zero','prepoint'))
                cache=ProbeDisplayCache(out/'display',budget.data['max_frames'],r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());frame_phases.append('INITIAL')
                raw_step=r.step
                def tracked_step(command):
                    nonlocal previous
                    label='RESIDUAL_HOLD' if phase=='RESIDUAL' and environment is not None and environment.hold is not None else phase
                    manual.append_json(out/'dispatch.ndjson',dict(global_frame=r.frames+1,phase=label,command=np.asarray(command).tolist()))
                    try:row=raw_step(command)
                    finally:report['frames']=r.frames
                    if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
                    previous=np.array(command,copy=True);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy());frame_phases.append(label)
                    if r.frames%600==0:report['periodic_checkpoint']=save('periodic-%05d'%r.frames,'transport');persist()
                    if r.frames%120==0:print('GLOBAL',r.frames,label,flush=True);persist()
                    return row
                r.step=tracked_step;r.step_authorized=True;report['status']='running';persist()
                paths=np.load(W/'cpu-v2/approach/plan.npz');first=read(W/'cpu-v2/approach/report.json');checker=ApproachClearance(PhaseView(r))
                for stage in first['stages'][:3]:
                    name=stage['name'];geometry(name,dict(qpos=paths[name+'_qpos'],velocity=paths[name+'_velocity']),checker)
                # Fresh observed-state planning at dispatch, still within this instance.
                view=PhaseView(r);checker=ApproachClearance(view);start=project(view,view.arrays['mechanism_qpos']);live=[];directory=out/'approach-dispatch';directory.mkdir()
                for stage in first['stages'][3:]:
                    name=stage['name'];goal=project(view,paths[name+'_qpos'][-1]);path=segment(view,start,goal,checker);live.append((name,path));start=goal
                np.savez(directory/'plan.npz',**{n+'_qpos':d['qpos'] for n,d in live},**{n+'_velocity':d['velocity'] for n,d in live})
                write(directory/'observation.json',checker.fresh_observation)
                for name,path in live:geometry(name,path,checker)
                phase='RESIDUAL';begin=r.frames
                environment=AttachedResidualEnv(out/'policy',budget,waiting_action_mode='error_strength_v1',collect=False)
                obs=handoff('RESIDUAL',lambda:environment.attach(r,out/'policy/episode-0001',previous))
                while True:
                    local=environment.runtime.frames;global_frame=r.frames;action=model.predict(obs,deterministic=True)[0]
                    manual.append_json(out/'policy-decisions.ndjson',dict(global_frame_before=global_frame,local_frame_before=local,observation=obs.tolist(),action=action.tolist(),model_sha256=MODEL_SHA))
                    obs,_,done,truncated,info=environment.step(action)
                    if done or truncated:break
                report['residual_result']=info['episode_summary']
                if not environment.report['success'] or environment.report['status']!='completed':raise ValueError('residual_dig_failed:'+environment.report['status'])
                report['phases'].append(dict(name='RESIDUAL',start_frame=begin,end_frame=r.frames,checkpoint=environment.report['checkpoint']));persist()
                handoff('RESIDUAL_RELEASE',lambda:environment.close())
                view=PhaseView(r)
                from transport_endpoint_candidate_v2 import build_plan
                stages,checker,planned=build_plan(view,out/'policy/episode-0001/endpoint.npz',out/'transport-plan')
                report['transport_plan']=planned;persist()
                for stage in stages:geometry(stage['name'],stage['path'],checker,hold=30+(180 if stage['name']=='DUMP' else 0))
                view=PhaseView(r);stages,checker=exit_and_return_plan(view,np.array(first['safe_return_goal']),out/'exit-return-plan')
                for stage in stages:geometry(stage['name'],stage['path'],checker)
                phase='FINAL_HOLD';begin=r.frames
                for _ in range(180):
                    row=r.step(np.zeros(4));checker.check(np.array(row['after_state']['mechanism_qpos']),physical=True)
                report['phases'].append(dict(name=phase,start_frame=begin,end_frame=r.frames,checkpoint=save('returned','returned')))
                report.update(status='continuous_single_scoop_completed',frames=r.frames,native_initializations=r.native_initializations,
                    final_simulation_time_s=float(r.capture()['simulation_time_s'][0]),final_opening_gap_m=bucket_opening_gap(r,np.array(r.bridge.state['mechanism_qpos'])))
                if sha(MODEL)!=MODEL_SHA or model._n_updates!=updates:raise ValueError('frozen_policy_changed')
                report['final_checkpoint']=save('final','returned');persist()
            except BaseException as e:
                report.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),frames=r.frames,phase=phase)
                try:report['recovery_checkpoint']=save('recovery','failure')
                except Exception as s:report['save_error']=repr(s)
                persist();raise
    except BaseException as e:
        report.update(status='failed',failure=repr(e));persist()
    finally:
        if cache is not None:
            try:
                d=cache.finish(report['status']);d.update(capture_origin='single_native_continuous_scoop',label='CONTINUOUS SINGLE SCOOP / FROZEN SAC DIG + GEOMETRY',native_initializations=report['native_initializations'])
                write(out/'display/reconstruction.json',d);write(out/'display/phases.json',frame_phases);report['display']=d
            except Exception as e:report['display_error']=repr(e)
        report.update(wall_seconds=time.monotonic()-started,neural_updates=model._n_updates-updates,remaining_budget_seconds=budget.remaining());persist()
        print(encode({k:report[k] for k in ['status','frames','native_initializations','wall_seconds','neural_updates']}),flush=True)

if __name__=='__main__':main()
