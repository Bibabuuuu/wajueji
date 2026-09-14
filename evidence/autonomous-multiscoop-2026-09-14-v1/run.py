"""Continuous observed-terrain excavation cycles in one physical MPM instance."""
from common import *
import argparse,signal
from mechanism import Mechanism
from reference_tracking import ReferenceTracker
from retention_observer import RetentionObserver
from stage_gate import StageGate
from receipt_observer import ReceiptObserver
from work_envelope import WorkEnvelope
from runtime_plan import build_cycle_paths
from planning.cycle import validate_cut
from planning_retry import MAX_PLANNING_ATTEMPTS,replan_allowed


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--mode',choices=('empty','contact'),required=True)
    ap.add_argument('--empty-report',type=Path)
    ap.add_argument('--cycles',type=int,default=3)
    ap.add_argument('--max-wall',type=float,default=10800)
    ap.add_argument('--max-sim',type=float,default=540)
    a=ap.parse_args()
    if not 1<=a.cycles<=3:raise ValueError('cycles_must_be_1_to_3')
    a.run.mkdir(exist_ok=False);stop=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:stop.append(s))
    report={'status':'building','mode':a.mode,'sources_before':sources(),'steps':0,
        'mechanism_steps':0,'logged_steps':0,'mpm_steps':0,'stages':{},'events':[],
        'cycles_requested':a.cycles,'cycles_completed':0,'cycles':[],'gui':False,
        'scope':'continuous observed-terrain acquisition, delivery and return in one MPM instance',
        'planning_replan_limit':MAX_PLANNING_ATTEMPTS,'planning_attempts':{},
        'reaction_lag_steps':1,'extra_projection_impulses_in_feedback':False,
        'state_continuity':{'same_mpm_instance':a.mode=='contact','particle_reset_between_cycles':False,
            'terrain_reobserved_each_cycle':True,'controller_state_reset_each_cycle':True},
        'timing_wall_s':{'sensor':0.,'selection':0.,'runtime_planning':0.,'observer':0.,
                         'mechanism':0.,'mpm':0.,'logging':0.}}
    started=time.perf_counter();soil=None;observer=None;receipt=None;mech=None
    fault_phase='initialization';fault_record=None;stage='INITIALIZE';poses=None
    log=None;perception_log=None;terrain=None;arrays=None;meta=None;capture_sim=0.;sequence=0

    def save():
        report.update(wall_s=time.perf_counter()-started)
        if mech is not None:report.update(last_mechanism=mech.state,sim_s=mech.state['time_s'])
        if soil is not None:report.update(mpm_steps=soil.steps,last_inventory=soil.last)
        p=a.run/'report.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.run/'report.json')

    def observe(mark=None):
        began=time.perf_counter()
        result=observer.sample(soil.state.particle_q.numpy(),soil.state.particle_qd.numpy(),
            poses['bucket'],time_s=soil.time,stage=stage,mark=mark)
        report['retention_last']=result
        if receipt is not None:
            report['receipt_last']=receipt.sample(soil.state.particle_q.numpy(),soil.state.particle_qd.numpy(),
                poses['bucket'],soil.time,stage,mark)
        report['timing_wall_s']['observer']+=time.perf_counter()-began
        return result

    def capture(use_sequence=None):
        nonlocal terrain,arrays,meta,capture_sim,sequence
        if use_sequence is None:sequence+=1
        else:sequence=int(use_sequence)
        now=0. if mech is None else float(mech.state['time_s'])
        particles=soil.state.particle_q.numpy() if soil is not None else initial_particles
        began=time.perf_counter();terrain,arrays,meta=sensor.capture(particles,poses,sequence)
        report['timing_wall_s']['sensor']+=time.perf_counter()-began
        capture_sim=now;meta['capture_sim_s']=now
        if perception_log is not None:perception_log.write(json.dumps(meta)+'\n')
        return terrain

    def acquired_since(record,start_mark):
        prior=record['cohorts'][start_mark]['still_in_cavity_kg']
        return max(0.,record['cavity_kg']-prior)

    save()
    try:
        geometry=read(a.geometry/'report.json')
        assert geometry['status']=='passed' and geometry['sources_unchanged']
        assert all(sha(p)==h for p,h in geometry['sources_after'].items()),'geometry_source_changed'
        assert all(sha(a.geometry/(s.lower()+'.npz'))==geometry['path_sha256'][s] for s in STAGES)
        report['geometry_report_sha256']=sha(a.geometry/'report.json')
        report['reference_paths_sha256']=geometry['path_sha256'];report['selection']=geometry['selection']
        build=read(M/'build.json')
        assert build['returncode']==0 and build['original_unchanged'] and sha(M/'mpm_bridge')==build['binary_sha256']
        unit=read(A/'unit-tests.json');probe=read(A/'probe/report.json')
        assert unit['passed'] and unit['gate_sha256']==sha(A/'stage_gate.py')
        assert probe['passed'] and probe['binary_sha256']==sha(M/'mpm_bridge')
        report['unit_tests_sha256']=sha(A/'unit-tests.json');report['probe_sha256']=sha(A/'probe/report.json')
        assert sha(A/'objective_strain.py')==read(A/'objectivity-source.json')['candidate_sha256']
        assert sha(Path(read(A/'objectivity-source.json')['installed_source']))==read(A/'objectivity-source.json')['installed_sha256']
        assert all(sha(B/n)==h for n,h in read(B/'manual-20260911-090834-launch.json')['source_sha256'].items())
        if a.mode=='contact':
            empty=read(a.empty_report);audit=read(a.empty_report.parent/'analysis.json')
            assert empty['status']=='passed' and empty['mode']=='empty' and empty['cycles_requested']==a.cycles
            assert empty['sources_before']==report['sources_before'] and empty['sources_unchanged']
            assert empty['geometry_report_sha256']==report['geometry_report_sha256']
            assert audit['passed'] and audit['inputs_sha256']['report.json']==sha(a.empty_report)
            assert audit['inputs_sha256']['tracking.jsonl']==sha(a.empty_report.parent/'tracking.jsonl')
            report['empty_report_sha256']=sha(a.empty_report)

        import warp as wp
        wp.init();wp.set_device('cuda:0')
        asm,chain,initial,geom,poses,scene,bounds,parts=setup();report['target_bounds']=bounds.tolist()
        envelope=WorkEnvelope(scene.tri)
        report['work_envelope']={'original_xy':[[1,-4],[9,4]],
            'truck_xy':[envelope.truck_low[:2].tolist(),envelope.truck_high[:2].tolist()],
            'z_bounds':[-.05,5.],'combination':'union of footprints with unchanged height bounds',
            'source':'actual truck_parts CAD triangles'}
        report['maximum_scene_outlier_count']=0
        initial_paths={s:np.load(a.geometry/(s.lower()+'.npz'))['qpos'] for s in STAGES}
        for s in STAGES:
            for q in initial_paths[s]:scene.check(asm.assemble(q,include_meshes=False)[1],s)
        initial_particles=np.load(B/'manual-20260911-090834/initial-bank.npz')['q']
        if a.mode=='contact':
            import newton,bank_runtime
            from bank_soil import Soil
            from moving_scene import to_pose
            from coupled import velocity
            from objective_strain import install as install_objective
            install_objective();newton.use_coord_layout_targets=True
            model,bucket,initial_poses,soil_asm,names=bank_runtime.build(T/'scene',R/'manual-comparison-20260908-v1')
            assert np.allclose(initial_poses['bucket'],poses['bucket'],rtol=0,atol=1e-10)
            cfg=read(B/'manual-20260911-090834/soil-config.json')
            fields=('friction','young_modulus','poisson_ratio','yield_pressure','yield_stress','tensile_yield_ratio','hardening','hardening_rate','softening_rate','dilatancy','viscosity','damping')
            settings={'material':{k:cfg['material'][k] for k in fields},'voxel':cfg['voxel_m'],
                      'numerics':{'max_iterations':250},'static_sweep':True}
            soil=Soil(model,bucket,read(B/'inventory-contract.json'),a.run,
                      spacing=cfg['particle_spacing_m'],settings=settings,speed_limit_enforced=False)
            report['initial_particle_q_exact']=bool(np.array_equal(initial_particles,soil.state.particle_q.numpy()))
            report['initial_particle_v_zero']=bool(np.all(soil.state.particle_qd.numpy()==0))
            actual=read(a.run/'soil-config.json')
            report['original_soil_config_match']={k:actual[k]==cfg[k] for k in
                ('particles','material','voxel_m','particle_spacing_m','initial_mass_kg','numerics','static_continuous_collision','body_map')}
            assert report['initial_particle_q_exact'] and report['initial_particle_v_zero'] and all(report['original_soil_config_match'].values())
            soil.state.body_q.assign(wp.array([to_pose(poses[n]) for n in names],dtype=wp.transform))
            soil.state.body_qd.zero_();soil.sample(poses['bucket'])
            observer=RetentionObserver(a.run,soil.masses,read(B/'inventory-contract.json'),
                                       np.load(A/'retention-front-rim.npz')['front'])
            receipt=ReceiptObserver(a.run,soil.masses,read(A/'receiver-contract.json'),observer)
            observe('initial')

        sensor=Sensor(geom,parts)
        sensor.capture(initial_particles,poses,0)
        capture(0)
        log=(a.run/'tracking.jsonl').open('w');perception_log=(a.run/'perception.jsonl').open('w')
        force=np.zeros(3);torque=np.zeros(3);point=poses['bucket'][:3,3].copy();peak=np.zeros(4)
        previous_entries=[]

        with Mechanism(M,initial,a.run/'mechanism.stderr') as mech:
            report['status']='running'
            for cycle_number in range(1,a.cycles+1):
                cycle_key=f'cycle-{cycle_number:02d}';cycle_dir=a.run/cycle_key;cycle_dir.mkdir(exist_ok=False)
                stage='PLANNING';state=mech.state;start_q=np.array(state['qpos'])
                if cycle_number>1:capture()
                if observer:
                    start_mark=cycle_key+'-start';start_inventory=observe(start_mark)
                    start_bucket_kg=start_inventory['cavity_kg']
                    truck_baseline=report['receipt_last']['stable_net_kg']
                else:
                    start_mark=None;start_bucket_kg=0.;truck_baseline=0.
                planning_attempts=[];report['planning_attempts'][cycle_key]=planning_attempts
                selection_wall=0.
                for attempt in range(1,MAX_PLANNING_ATTEMPTS+1):
                    attempt_state=mech.state
                    attempt_record={'attempt':attempt,'status':'building',
                        'start_sim_s':attempt_state['time_s'],
                        'start_particle_step':0 if soil is None else soil.steps}
                    planning_attempts.append(attempt_record)
                    began=time.perf_counter();selection=select_cut(terrain,excluded_entry_xy=previous_entries)
                    elapsed=time.perf_counter()-began;selection_wall+=elapsed
                    report['timing_wall_s']['selection']+=elapsed;attempt_record['selection']=selection
                    suffix='' if attempt==1 else f'-attempt-{attempt:02d}'
                    planning_observation=cycle_dir/f'planning-observation{suffix}.npz'
                    np.savez_compressed(planning_observation,**arrays)
                    attempt_record['planning_observation']=planning_observation.name
                    if cycle_number==1:
                        assert selection==geometry['selection'],'live_initial_selection_changed'
                        paths=initial_paths;path_report={'status':'passed','wall_s':0.,'source':'frozen_initial_geometry'}
                        report['live_initial_selection_match']=True
                    else:
                        began=time.perf_counter();path_dir=cycle_dir/('paths' if attempt==1 else f'paths-attempt-{attempt:02d}')
                        paths,path_report=build_cycle_paths(asm,chain,start_q,selection,
                            read(B/'inventory-contract.json')['truck'],scene,bounds,path_dir)
                        report['timing_wall_s']['runtime_planning']+=time.perf_counter()-began
                        path_report['directory']=path_dir.name
                    attempt_record['path_plan']=path_report
                    capture();dispatch_observation=cycle_dir/f'dispatch-observation{suffix}.npz'
                    np.savez_compressed(dispatch_observation,**arrays)
                    attempt_record['dispatch_observation']=dispatch_observation.name
                    try:
                        dispatch_validation=validate_cut(terrain,selection,excluded_entry_xy=previous_entries)
                    except ValueError as error:
                        attempt_record.update(status='rejected',error=repr(error));save()
                        if not replan_allowed(error,cycle_number,attempt):raise
                        continue
                    attempt_record.update(status='passed',dispatch_validation=dispatch_validation)
                    break
                previous_entries.append(selection['entry_xy_m'])
                cycle_report={'cycle':cycle_number,'status':'running','selection':selection,
                    'selection_wall_s':selection_wall,'path_plan':path_report,
                    'dispatch_validation':dispatch_validation,'planning_attempts':planning_attempts,
                    'start_sim_s':state['time_s'],
                    'start_particle_step':0 if soil is None else soil.steps,
                    'start_bucket_kg':start_bucket_kg,'truck_baseline_stable_net_kg':truck_baseline,
                    'stages':{},'events':[]}
                report['cycles'].append(cycle_report);save()

                index=0;stage=STAGES[index];tracker=None;gate=None;hold_start=None
                hold_peak=np.zeros(4);wait_start=None
                while True:
                    fault_phase='before_step';fault_record=None
                    if stop or (a.run/'STOP').exists():raise InterruptedError('cooperative_stop')
                    if time.perf_counter()-started>a.max_wall:raise TimeoutError('wall_budget')
                    state=mech.state;now=state['time_s'];q=np.array(state['qpos'])
                    qd=np.array(state['qvel'])[state['axis_dof_address']]
                    if now>a.max_sim:raise TimeoutError('simulation_budget')
                    if time.perf_counter()-meta['capture_wall_s']>=.15 or now-capture_sim>=.15:capture()
                    age_wall=time.perf_counter()-meta['capture_wall_s'];age_sim=now-capture_sim
                    fresh=terrain['valid'] and 0<=age_wall<=.3 and 0<=age_sim<=.3
                    if not fresh:raise ValueError('perception_expired')
                    clearance=scene.check(poses,stage)
                    if tracker is None:
                        path=paths[stage]
                        tracker=ReferenceTracker(path,COMMAND_LIMITS,target_bounds=bounds,
                            initial_bias=np.array(state['target'])-path[0,[7,9,12,15]])
                        gate=StageGate(stage,now,tracker.duration)
                        cycle_report['stages'][stage]={'status':'running','start_s':now,
                            'reference_duration_s':tracker.duration}
                        report['stages'][cycle_key+'/'+stage]=cycle_report['stages'][stage]
                        stage_peak=np.zeros(4)
                    execution_stage=stage
                    command=tracker.command(time_s=now,qpos=q,target=state['target'],axis_velocity=qd)
                    reference=tracker.reference().copy();reference_phase=tracker.phase
                    err=abs(q[tracker.axes]-tracker.reference()[tracker.axes])
                    stage_peak=np.maximum(stage_peak,err);peak=np.maximum(peak,err)
                    reached=(tracker.done and np.all(abs(q[tracker.axes]-tracker.path[-1,tracker.axes])<=tracker.finish_tolerance)
                             and np.all(abs(qd)<=tracker.finish_velocity))
                    if stage in ('HOLD','SETTLE','FINAL_HOLD'):
                        hold_peak=np.maximum(hold_peak,abs(q[tracker.axes]-tracker.path[-1,tracker.axes]))
                        if not reached:raise ValueError('hold_pose_or_velocity')
                        if stage=='HOLD' and clearance['bucket_min_z_m']<scene.bank_hi[2]+.10:raise ValueError('hold_not_above_bank')
                        if stage=='HOLD' and now-hold_start>=10-1e-9:
                            if observer:
                                hold_end_mark=cycle_key+'-hold-end';result=observe(hold_end_mark)
                                acquired=acquired_since(result,start_mark)
                                if acquired<20:raise ValueError('empty_hold')
                                acquired_mark=cycle_key+'-acquired'
                                cohort_mass=observer.derive_cohort(acquired_mark,hold_end_mark,start_mark)
                                if abs(cohort_mass-acquired)>1e-7:raise ValueError('acquisition_cohort_mass_mismatch')
                                receipt.set_active_cohort(acquired_mark)
                                cycle_report['hold_end_newly_acquired_kg']=acquired
                                cycle_report['acquisition_cohort']=acquired_mark
                            cycle_report['hold_duration_s']=now-hold_start
                            cycle_report['hold_max_axis_error']=hold_peak.tolist()
                            event={'cycle':cycle_number,'from':'HOLD','to':'TURN','sim_s':now,
                                   'actual_reached':bool(reached),'confirmation_s':now-hold_start}
                            report['events'].append(event);cycle_report['events'].append(event)
                            index=STAGES.index('TURN');stage='TURN';tracker=None
                        elif stage in ('SETTLE','FINAL_HOLD'):
                            elapsed=now-wait_start
                            if elapsed>30:raise TimeoutError('receipt_stability_timeout:'+stage)
                            if observer and report['steps']%30==0:observe()
                            stable_increment=(report.get('receipt_last',{}).get('stable_net_kg',0.)-truck_baseline)
                            qualified=(receipt is None or (stable_increment>=10 and
                                report['receipt_last']['stable_acquired_kg']>=10))
                            if elapsed>=10-1e-9 and qualified:
                                if observer:observe(cycle_key+('-unloaded' if stage=='SETTLE' else '-returned'))
                                if stage=='FINAL_HOLD':
                                    cycle_report['final_hold_duration_s']=elapsed
                                    cycle_report['stable_delivery_increment_kg']=report.get('receipt_last',{}).get('stable_net_kg',0.)-truck_baseline
                                    cycle_report['final_bucket_kg']=report.get('retention_last',{}).get('cavity_kg',0.)
                                    cycle_report['end_sim_s']=now;cycle_report['end_particle_step']=0 if soil is None else soil.steps
                                    cycle_report['status']='passed';report['cycles_completed']=cycle_number;save();break
                                cycle_report['unload_wait_s']=elapsed
                                event={'cycle':cycle_number,'from':stage,'to':'CLEAR_TRUCK','sim_s':now,
                                       'actual_reached':bool(reached),'confirmation_s':elapsed,
                                       'stable_delivery_increment_kg':stable_increment}
                                report['events'].append(event);cycle_report['events'].append(event)
                                index=STAGES.index('CLEAR_TRUCK');stage='CLEAR_TRUCK';tracker=None
                    else:
                        acquisition=None
                        if observer and reached and stage in ('CURL','LIFT'):
                            acquisition=acquired_since(observe(),start_mark)
                        if gate.update(now,reached=reached,fresh=fresh,cavity_kg=acquisition,
                                       clear=clearance['bucket_min_z_m']>=scene.bank_hi[2]+.10):
                            current=cycle_report['stages'][stage]
                            current.update(status='passed',end_s=now,maximum_axis_error=stage_peak.tolist(),
                                endpoint_error=abs(q[tracker.axes]-tracker.path[-1,tracker.axes]).tolist(),
                                endpoint_velocity=qd.tolist())
                            old=stage
                            if stage=='LIFT':
                                stage='HOLD';hold_start=now;cycle_report['endpoint_arrival_s']=now
                                if observer:observe(cycle_key+'-hold-start')
                            elif stage in ('DUMP','RETURN'):
                                stage='SETTLE' if stage=='DUMP' else 'FINAL_HOLD';wait_start=now
                                if observer:observe(cycle_key+('-dump-arrival' if stage=='SETTLE' else '-return-arrival'))
                            else:
                                index+=1;stage=STAGES[index];tracker=None
                            event={'cycle':cycle_number,'from':old,'to':stage,'sim_s':now,
                                'actual_reached':bool(reached),'newly_acquired_kg':acquisition,
                                'confirmation_s':now-gate.since}
                            report['events'].append(event);cycle_report['events'].append(event);save()

                    if time.perf_counter()-meta['capture_wall_s']>=.15:capture()
                    age_wall=time.perf_counter()-meta['capture_wall_s'];age_sim=now-capture_sim
                    if not terrain['valid'] or not 0<=age_wall<=.3 or not 0<=age_sim<=.3:
                        raise ValueError('perception_expired_at_dispatch')
                    applied_force=force.copy();applied_torque=torque.copy();applied_point=point.copy()
                    fault_phase='mechanism_step';fault_record={'command':command.tolist(),
                        'time_before_s':now,'stage':execution_stage,'cycle':cycle_number}
                    began=time.perf_counter();state=mech.step(command,force,torque,point)
                    report['timing_wall_s']['mechanism']+=time.perf_counter()-began
                    report['mechanism_steps']+=1;fault_phase='post_mechanism_step';fault_record['mechanism']=state
                    if np.any(np.array(state['target'])<bounds[:,0]-1e-10) or np.any(np.array(state['target'])>bounds[:,1]+1e-10):
                        raise ValueError('internal_target_outside_autonomous_8mm')
                    newposes=asm.assemble(np.array(state['qpos']),include_meshes=False)[1];scene.check(newposes,stage)
                    if np.linalg.norm(np.array(state['bucket_origin'])-newposes['bucket'][:3,3])>1e-8:
                        raise ValueError('dynamic_geometry_alignment')
                    if soil:
                        soil.state.body_q.assign(wp.array([to_pose(poses[n]) for n in names],dtype=wp.transform))
                        soil.state.body_qd.assign(wp.array([velocity(poses[n],newposes[n],1/120) for n in names],dtype=wp.spatial_vector))
                        soil.state.body_f.zero_();fault_phase='soil_step';began=time.perf_counter()
                        soil.step(1/120,newposes['bucket'],record=(report['steps']+1)%12==0)
                        report['timing_wall_s']['mpm']+=time.perf_counter()-began
                        report['mpm_steps']=soil.steps;fault_phase='post_soil_step';fault_record['soil_steps']=soil.steps
                        impulse,position,ids=soil.solver.collect_collider_impulses(soil.state)
                        point=newposes['bucket'][:3,3].copy()
                        force,torque=soil.diagnostics.force(impulse,position,ids,soil.bucket_collider,point,1/120)
                        if not np.isfinite(force).all() or not np.isfinite(torque).all():raise ValueError('nonfinite_wrench')
                        soil.state.body_q.assign(wp.array([to_pose(newposes[n]) for n in names],dtype=wp.transform))
                    poses=newposes;report['steps']+=1
                    began=time.perf_counter()
                    log.write(json.dumps({'step':report['steps'],'cycle':cycle_number,'stage':execution_stage,
                        'time_s':state['time_s'],'reference':reference.tolist(),'reference_phase_s':reference_phase,
                        'command':command.tolist(),'mechanism':state,'applied_force_n':applied_force.tolist(),
                        'applied_torque_nm':applied_torque.tolist(),'applied_point_world_m':applied_point.tolist(),
                        'measured_force_n':force.tolist(),'measured_torque_nm':torque.tolist(),
                        'perception_sequence':sequence,'perception_age_wall_s':age_wall,
                        'perception_age_sim_s':age_sim,'error_axes':err.tolist(),
                        'bucket_min_z_m':clearance['bucket_min_z_m']})+'\n')
                    report['timing_wall_s']['logging']+=time.perf_counter()-began
                    report['logged_steps']+=1;fault_phase='post_log'
                    if soil and report['steps']%12==0:
                        outside=envelope.outside(soil.state.particle_q.numpy());count=int(outside.sum())
                        report['maximum_scene_outlier_count']=max(report['maximum_scene_outlier_count'],count)
                        with (a.run/'scene-envelope.jsonl').open('a') as ef:
                            ef.write(json.dumps({'time_s':soil.time,'step':soil.steps,
                                'legacy_acquisition_outlier_count':soil.last['outlier_count'],
                                'scene_outlier_count':count,'scene_outlier_ids':np.flatnonzero(outside).tolist()})+'\n')
                        if count:raise ValueError('particle_outside_full_work_envelope')
                    if observer and report['steps']%30==0:observe()
                    if report['steps']%120==0:
                        report['maximum_axis_error']=peak.tolist();log.flush();perception_log.flush();save()
                        latest_acquired=None if not observer else acquired_since(report['retention_last'],start_mark)
                        print(json.dumps({'mode':a.mode,'cycle':cycle_number,'stage':stage,
                            'sim_s':state['time_s'],'force_n':float(np.linalg.norm(force)),
                            'newly_acquired_kg':latest_acquired,'wall_s':report['wall_s']}),flush=True)
            report['status']='passed'
        report['maximum_axis_error']=peak.tolist()
    except Exception as error:
        report.update(status='failed',error=repr(error),failed_stage=stage,fault_phase=fault_phase)
        if report['cycles'] and report['cycles'][-1]['status']=='running':
            report['cycles'][-1].update(status='failed',error=repr(error),failed_stage=stage)
        if fault_record is not None:
            (a.run/'fault-step.json').write_text(json.dumps({'phase':fault_phase,**fault_record},indent=2)+'\n')
    finally:
        if observer:
            try:observe('final')
            except Exception as error:report['final_observer_error']=repr(error)
        if soil:
            try:soil.close()
            except Exception as error:report.update(status='failed',cleanup_error=repr(error))
        if log:log.close()
        if perception_log:perception_log.close()
        report['sources_after']=sources();report['sources_unchanged']=report['sources_before']==report['sources_after'];save()
        print(json.dumps({k:v for k,v in report.items() if k in
            ('status','error','failed_stage','steps','cycles_requested','cycles_completed','sim_s','wall_s',
             'retention_last','receipt_last','sources_unchanged','timing_wall_s')},indent=2),flush=True)
    return 0 if report['status']=='passed' and report['cycles_completed']==a.cycles and report['sources_unchanged'] else 1


if __name__=='__main__':raise SystemExit(main())
