"""CPU-only evidence audit; preserve task failures separately from integrity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
RUN=ROOT/'runs/d165-transport-full-pair-20261014-v1'
sys.path[:0]=[str(WORK),str(ROOT/'src/v4'),str(ROOT)]
from transport_cpu_candidate_v3 import CpuTransportRuntime,TransportController,encode
from clearance import FixedRouteClearance
from layered_47_cycle_geometry import bucket_mask
from terrain.truck_receiver import TruckReceiver,OUTSIDE,CAVITY,BOUNDARY,ABOVE_RIM,OTHER_OWNER
from live_bucket_fill import MixedFillEstimator,load_config
from mixed_material_r0 import quaternion_matrix_wxyz
import run_layered_47_sparse_manual_contact_episode as manual

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
batch=read(RUN/'report.json');budget_path=ROOT/'runs/d165-transport-full-pair-20261014-v1-budget.json'
budget=read(budget_path);protocol=read(WORK/'transport-full-pair-protocol-v1.json')
checks={'scope':batch['physical_frames']<=5394 and budget['seconds']==2700 and budget['per_case_max_frames']==2697,
    'budget_identity':sha(budget_path)==protocol['budget_sha256'],
    'runner_identity':sha(WORK/'transport_full_pair.py')==protocol['runner_sha256']==batch['runner_sha256'],
    'no_training':batch['training_updates']==0,
    'case_order':list(batch['cases']) in [['better_manual'],['better_manual','frozen_residual']],
    'stop_after_first_failure':len(batch['cases'])==1 if batch['cases']['better_manual']['status']!='transport_unload_development_passed' else True}
cases={};receiver=TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))
classes={'outside':OUTSIDE,'core':CAVITY,'boundary_uncertain':BOUNDARY,'above_rim_unverified':ABOVE_RIM,'other_owner':OTHER_OWNER}
for name,case in batch['cases'].items():
    out=RUN/name;source=Path(case['source']['source']);actions=rows(out/'actions.ndjson');frames=rows(out/'physics/frames.ndjson');clear=rows(out/'clearance.ndjson')
    c={'source_unchanged':sha(source)==case['source']['sha256'] and sha(str(source)+'.json')==case['source']['metadata_sha256'],
       'reported_frames':len(frames)==case['frames'],
       'contiguous_frames':[f['frame'] for f in frames]==list(range(1,len(frames)+1)),
       'request_execution_count':len(actions) in [len(frames),len(frames)+1],
       'commands_match_dispatch':all(a['command']==f['applied_command_si'] for a,f in zip(actions,frames)),
       'hard_checks_independently_recomputed':all(manual.hard_guard_checks(f,allow_truck=True)==f['hard_guard_checks'] for f in frames),
       'no_formal_volume_claim':case['formal_mixed_volume_qualified'] is False,
       'frame_ceiling':case['frames']<=2697}
    if case['zero_restore_passed']:
        c['zero_24_arrays_exact']=len(case['zero_arrays_exact'])==24 and all(case['zero_arrays_exact'].values())
        c['native_restore_exact']=max(case['native_restore_errors'].values())<=1e-12 and case['sweep_restore_exact']
        with np.load(out/'zero.npz') as a,np.load(source) as b:c['saved_zero_exact']=len(a.files)==24 and all(np.array_equal(a[k],b[k]) for k in a.files)
    else:
        c['failed_restore_did_not_move']=case['frames']==0 and case['status']=='failed'
    failed_guards=[f['frame'] for f in frames if not(f['hard_guard'] and f['hencky_material_guard']['safe'] and f['moving_bucket_sweep']['safe'])]
    c['guard_failure_stops_execution']=not failed_guards or failed_guards==[len(frames)] and case['status']=='failed'
    details=dict(status=case['status'],failure=case.get('failure'),checks=c,frames=case['frames'],failed_guard_frames=failed_guards,
        task_complete=case['status']=='transport_unload_development_passed',completed_stages=[s['name'] for s in case['stages']])
    if frames and (out/'display/states.json').exists():
        states=read(out/'display/states.json');particles=np.load(out/'display/particles.npy',mmap_mode='r')
        c['cache_count']=len(states)==len(particles) and len(states) in [len(frames),len(frames)+1]
        c['cache_states_match_frames']=all(s==f['after_state'] for s,f in zip(states[1:],frames))
        c['cache_hashes_match']=all(sha(out/'display'/k)==v for k,v in case['display']['cache_sha256'].items())
        if 'final_checkpoint' in case:
            final=Path(case['final_checkpoint']['path']);c['final_hash']=sha(final)==case['final_checkpoint']['sha256']
            with np.load(final) as data:
                c['final_simulation_clock']=abs(float(data['simulation_time_s'][0])-float(np.load(source)['simulation_time_s'][0])-case['frames']/60)<1e-10
                if len(states)==len(frames)+1:c['final_cache_particles_match']=np.array_equal(particles[-1],data['particle_q'])
        r=CpuTransportRuntime(source,RUN/('audit-cpu-'+name),case['source']['sha256']);checker=FixedRouteClearance(r)
        control=TransportController(r,checker,np.zeros(4));exact=[];bias=[]
        for a in actions[:len(frames)]:
            state=a['before_state'];command,detail=control.command(a['phase'],np.array(state['mechanism_qpos']),state['controller_target'],np.array(a['desired']),np.array(a['velocity']))
            exact.append(np.array_equal(command,a['command']));bias.append(np.array_equal(detail['load_bias'],a['detail']['load_bias']))
        c['all_executed_commands_recomputed_exact']=all(exact);c['all_load_biases_recomputed_exact']=all(bias)
        c['loaded_commands_do_not_open']=all(abs(a['command'][3])<1e-10 for a in actions if a['phase'] in ['CLEAR_PILE','ALIGN'])
        if len(frames)>=120:
            c['new_short_gate_passed']=case['short_gate_passed'] and sha(out/'short-120.npz')==case['short_gate_checkpoint']['sha256']
        if details['task_complete']:
            c['complete_three_stages']=details['completed_stages']==['CLEAR_PILE','ALIGN','DUMP'] and case['frames']==2697
            c['all_protections_passed']=not failed_guards
            c['all_tracking_checks']=len(clear)==2697 and max(x['axis_error'] for x in clear)<=.025
            c['cache_has_every_frame']=len(states)==2698
        scene=read(out/'display/scene.json')
        est=MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in scene['records']],manual._cache_rock_vertices(Path(scene['root']),scene['records']))
        def observed(frame):
            state=states[frame];pose=r.assembly.assemble(np.array(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
            cloud=np.array(particles[frame]);labels=receiver.classify(cloud)
            rock_liters=0.;rock_centers=0
            for (_,samples,weight),rock in zip(est.rocks,state['rocks'],strict=True):
                q=np.array(rock['qpos']);world=samples@quaternion_matrix_wxyz(q[3:]).T+q[:3]
                rock_liters+=float(np.count_nonzero(receiver.classify(world)==CAVITY)*weight*1000)
                rock_centers+=int(receiver.classify(q[None,:3])[0]==CAVITY)
            display=est.measure_snapshot(dict(frame=frame,soil=cloud,pose=pose,rocks=np.array([x['qpos'] for x in state['rocks']])))
            return dict(frame=frame,bucket_display=display,all_soil_centers_in_truck=int(np.count_nonzero(labels==CAVITY)),
                soil_receiver_partition={key:int(np.count_nonzero(labels==value)) for key,value in classes.items()},receiver_boundary_band_m=receiver.band,
                rock_centers_in_truck=rock_centers,rock_reference_sample_volume_liters_in_truck=rock_liters,
                scope='Instant CAD interior classification; rock reference sample volume; not calibrated stable bulk mixed volume')
        sample_frames=sorted({0,120,*[s['frame'] if 'frame' in s else s['frames'] for s in case['stages']],max(0,len(states)-31),max(0,len(states)-16),len(states)-1})
        sample_frames=[f for f in sample_frames if f<len(states)]
        observations=[observed(f) for f in sample_frames];details['observations']=observations
        if len(states)==len(frames)+1:
            pose0=r.assembly.assemble(np.array(states[0]['mechanism_qpos']),include_meshes=False)[1]['bucket']
            pose1=r.assembly.assemble(np.array(states[-1]['mechanism_qpos']),include_meshes=False)[1]['bucket']
            payload=bucket_mask(np.array(particles[0]),pose0);final_cloud=np.array(particles[-1])
            received=payload&(receiver.classify(final_cloud)==CAVITY);remaining=payload&bucket_mask(final_cloud,pose1)
            inventory=dict(source_particles=int(payload.sum()),received_source_particles=int(received.sum()),remaining_source_particles=int(remaining.sum()))
            details['source_cavity_inventory_recomputed']=inventory
            if 'unload_inventory' in case:c['development_inventory_recomputed']=all(case['unload_inventory'][k]==v for k,v in inventory.items())
            labels=receiver.classify(final_cloud)
            partition={key:int(np.count_nonzero(payload&(labels==value))) for key,value in classes.items()}
            details['source_cavity_receiver_partition']=partition
            c['source_receiver_partition_sums']=sum(partition.values())==inventory['source_particles']
            upper=payload&(labels==ABOVE_RIM)
            if upper.any() and 'final_checkpoint' in case:
                with np.load(case['final_checkpoint']['path']) as data:speeds=np.linalg.norm(data['particle_qd'][upper],axis=1)
                local=(final_cloud[upper]-receiver.pose[:3,3])@receiver.pose[:3,:3]
                rim=np.interp(local[:,0],receiver.rim[:,0],receiver.rim[:,3])
                earlier=np.array(particles[max(0,len(states)-31)])
                movement=np.linalg.norm(final_cloud[upper]-earlier[upper],axis=1)
                details['above_rim_diagnostics']=dict(count=int(upper.sum()),speed_p50_p95_max_m_s=np.quantile(speeds,[.5,.95,1]).tolist(),
                    displacement_last_half_second_p50_p95_max_m=np.quantile(movement,[.5,.95,1]).tolist(),
                    height_above_rim_min_median_max_m=np.quantile(local[:,2]-rim,[0,.5,1]).tolist(),
                    scope='Motion and height diagnostics only; does not validate support, ownership or effective heap volume')
        details['origin_delta_m']=(np.array(states[-1]['bucket_origin_world_m'])-np.array(states[0]['bucket_origin_world_m'])).tolist()
        details['observed_max_tracking_error_m_or_rad']=max((x['axis_error'] for x in clear),default=None)
        details['source_sha256']=case['source']['sha256']
        print(name,'recomputed',len(exact),'commands','task_complete',details['task_complete'],flush=True)
    cases[name]=details;checks.update({name+'_'+k:bool(v) for k,v in c.items()})
checks['frozen_model_unchanged']=sha(ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip')=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
result=dict(status='passed' if all(checks.values()) else 'failed',check_count=len(checks),checks=checks,cases=cases,
    full_pair_task_complete=batch['status']=='completed' and len(cases)==2 and all(c['task_complete'] for c in cases.values()),
    physical_frames=batch['physical_frames'],training_updates=0,audit_new_physical_frames=0,
    budget_remaining_seconds=budget['epoch_deadline']-time.time(),scope='Evidence integrity audit; task outcome and any protection stop recorded separately; not official truck volume')
(RUN/'read-only-audit.json').write_text(encode(result)+'\n')
print(encode({k:result[k] for k in ['status','check_count','full_pair_task_complete','physical_frames','budget_remaining_seconds']}))
assert all(checks.values())
