"""Independent CPU audit and fixed-window receipt quantities; no new physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from exit_common import *
from live_bucket_fill import MixedFillEstimator,load_config
from mixed_material_r0 import quaternion_matrix_wxyz
import time
RUN=ROOT/'runs/d165-exit-settle-pair-20261014-v1'
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]

def receipt_label(x):return (x==CAVITY)|(x==ABOVE_RIM)

def main():
    started=time.monotonic();batch=read(RUN/'report.json');protocol=read(WORK/'exit-settle-protocol-v1.json');cpu=read(CPU/'report.json');measurement=protocol['measurement']
    assert batch['status']=='completed'
    checks=dict(no_training=batch['training_updates']==0,budget_identity=sha(ROOT/'runs/d165-exit-settle-pair-20261014-v1-budget.json')==protocol['budget_sha256']==batch['budget_sha256'],
        runner_identity=sha(WORK/'run_exit_settle.py')==protocol['runner_sha256']==batch['runner_sha256'],common_identity=sha(WORK/'exit_common.py')==protocol['common_sha256'],
        cpu_identity=sha(CPU/'report.json')==protocol['cpu_report_sha256']==batch['cpu_sha256'],correct_case_order=list(batch['cases'])==['better_manual','frozen_residual'],frame_count=batch['physical_frames']==4880)
    cases={};truck=receiver()
    for name,report in batch['cases'].items():
        out=RUN/name;case=cpu['cases'][name];prefix=read(out/'prefix-continuity/report.json')
        c=dict(task_complete=report['status']=='withdraw_and_settle_completed',frames=report['frames']==case['frames']==2260,
            source_unchanged=source_info(name)==case['source'],prefix_source_unchanged=source_info(name,'periodic-02520')==case['prefix_source'],
            prefix_reproduces_final=all(v for k,v in prefix['original_final_arrays_exact'].items() if k!='simulation_time_s') and prefix['original_final_time_error_s']<1e-10,
            prefix_native_and_high_control_exact=prefix['original_final_native_exact'] and prefix['original_final_sweep_exact'] and prefix['high_controller_exact'],
            prefix_gate_identity=sha(out/'prefix-continuity/report.json')==report['prefix_gate_report_sha256'],
            next_step_and_three_frame_gate=report['restored_next_steps_exact']==[True]*3,
            new120gate=report['short_gate_passed'] and sha(out/'short-120.npz')==report['short_gate_checkpoint']['sha256'])
        for label_name,source,history in [('prefix',case['prefix_source'],out/'prefix-continuity'),('main',case['source'],out)]:
            r=CpuRuntime(source['source'],RUN/('audit-'+name+'-'+label_name),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
            control=controller(r,read(source['source']+'.control.json'));actions=rows(history/'actions.ndjson');frames=rows(history/'physics/frames.ndjson');clear=rows(history/'clearance.ndjson')
            c[label_name+'_counts_and_order']=len(actions)==len(frames)==len(clear)==(180 if label_name=='prefix' else 2260) and [f['frame'] for f in frames]==list(range(1,len(frames)+1))
            exact=[];bias=[]
            for a,f in zip(actions,frames,strict=True):
                state=a['before_state'];cmd,detail=control.command(a['phase'],np.array(state['mechanism_qpos']),state['controller_target'],np.array(a['desired']),np.array(a['velocity']))
                exact.append(np.array_equal(cmd,a['command']) and np.array_equal(cmd,f['applied_command_si']));bias.append(np.array_equal(detail['load_bias'],a['detail']['load_bias']))
            c[label_name+'_all_commands_recomputed_exact']=all(exact);c[label_name+'_all_biases_recomputed_exact']=all(bias)
            c[label_name+'_hard_guards_recomputed']=all(manual.hard_guard_checks(f,allow_truck=True)==f['hard_guard_checks'] and f['hard_guard'] and f['hencky_material_guard']['safe'] and f['moving_bucket_sweep']['safe'] for f in frames)
            c[label_name+'_tracking_guard']=max(x['axis_error'] for x in clear)<=.025
            if label_name=='main':
                c['zero_24_arrays_exact']=len(report['zero']['arrays_exact'])==24 and all(report['zero']['arrays_exact'].values())
                with np.load(out/'zero.npz') as z:c['zero_saved_exact']=all(np.array_equal(z[k],v) for k,v in r.arrays.items()) and len(z.files)==24
                c['zero_native_sweep_exact']=max(report['zero']['native_errors'].values())<=1e-12 and report['zero']['sweep_exact']
                c['main_stages']= [s['name'] for s in report['stages']]==['CLEAR_TRUCK','WITHDRAW_SWING','SETTLE'] and report['settle_frames']==300
        states=read(out/'display/states.json');cloud=np.load(out/'display/particles.npy',mmap_mode='r');scene=read(out/'display/scene.json')
        c['cache_frames_and_states']=len(cloud)==len(states)==2261 and all(s==f['after_state'] for s,f in zip(states[1:],frames,strict=True))
        c['cache_hashes']=all(sha(out/'display'/k)==v for k,v in report['display']['cache_sha256'].items())
        c['final_checkpoint_hash']=sha(out/'final.npz')==report['final_checkpoint']['sha256']
        with np.load(out/'final.npz') as f:
            final={k:f[k].copy() for k in f.files}
        c['final_cache_particles_exact']=np.array_equal(cloud[-1],final['particle_q'])
        with np.load(out/'prefix-continuity/reference-3.npz') as f:
            c['three_frame_particles_and_mechanism_independent']=np.array_equal(cloud[3],f['particle_q']) and np.array_equal(states[3]['mechanism_qpos'],f['mechanism_qpos']) and np.array_equal(np.array([rock['qpos'] for rock in states[3]['rocks']]),f['rigid_qpos'])
        c['final_time']=abs(float(final['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0])-2260/60)<1e-10
        mass=np.load(out/'particle-mass.npy');spacing=float(r.base['mpm']['particle_spacing_m']);expected_mass=np.float32(r.base['mpm']['particle_density_kg_m3']*spacing**3)
        c['mass_inventory_bound']=mass.shape==(256441,) and np.all(mass==expected_mass) and sha(out/'particle-mass.npy')==report['soil_mass_inventory']['sha256']
        q=final['mechanism_qpos'];obs,labels,distance=soil_observation(r,final['particle_q'],q,final['particle_qd'],np.array(cloud[-31]))
        c['bucket_fully_withdrawn']=obs['bucket_outside_opening_aabb_gap_m']>.1
        c['no_receipt_soil_near_bucket']=not np.any(receipt_label(labels)&(distance<=.05))
        # Fixed identities, every frame in the last3s, separate stock and static subset.
        first=len(states)-181;ids=np.flatnonzero(receipt_label(labels));initial=np.array(cloud[first,ids]);persistent=np.ones(len(ids),bool);movement=np.zeros(len(ids))
        for index in range(first,len(states)):
            p=np.array(cloud[index,ids]);persistent&=receipt_label(truck.classify(p));movement=np.maximum(movement,np.linalg.norm(p-initial,axis=1))
        speed=np.linalg.norm(final['particle_qd'][ids],axis=1)
        stable=persistent&(movement<=measurement['soil_max_displacement_m'])&(speed<=measurement['soil_final_speed_m_s'])
        stable_ids=ids[stable];stable_mask=np.zeros(256441,bool);stable_mask[stable_ids]=True
        quant=dict(soil=dict(receipt_stock_count=len(ids),persistent_3s_count=int(persistent.sum()),stable_count=int(stable.sum()),
            core_stable_count=int(np.count_nonzero(stable_mask&(labels==CAVITY))),above_rim_stable_count=int(np.count_nonzero(stable_mask&(labels==ABOVE_RIM))),
            unsteady_or_nonpersistent_count=int((~stable).sum()),stable_mass_kg=float(mass[stable_ids].sum(dtype=np.float64)),
            stable_original_reference_volume_liters=float(len(stable_ids)*spacing**3*1000),
            maximum_displacement_p50_p95_max_m=np.quantile(movement,[.5,.95,1]).tolist(),final_speed_p50_p95_max_m_s=np.quantile(speed,[.5,.95,1]).tolist()),rocks=[])
        groups={};pre=case['diagnostic'];before=np.load(case['source']['source'])['particle_q']
        for group in ['near50','far50']:
            cohort=np.array(pre['source_above_'+group+'_ids'],int);groups[group]=dict(initial_count=len(cohort),final_receipt_count=int(receipt_label(labels[cohort]).sum()),
                final_stable_receipt_count=int(stable_mask[cohort].sum()),final_near_bucket_count=int((distance[cohort]<=.05).sum()),
                median_world_displacement_m=float(np.median(np.linalg.norm(final['particle_q'][cohort]-before[cohort],axis=1))))
        est=MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in scene['records']],manual._cache_rock_vertices(Path(scene['root']),scene['records']))
        rock_liters=0.;rock_mass=0.;rock_stock_liters=0.
        for i,((_,samples,weight),record) in enumerate(zip(est.rocks,scene['records'],strict=True)):
            def points(index):
                pose=np.array(states[index]['rocks'][i]['qpos']);return samples@quaternion_matrix_wxyz(pose[3:]).T+pose[:3]
            end=points(-1);membership=receipt_label(truck.classify(end));stock=float(membership.sum()*weight*1000);rock_stock_liters+=stock
            if not membership.any():continue
            start=points(first);persistent_samples=membership.copy();maximum=0.
            for index in range(first,len(states)):
                p=points(index);persistent_samples&=receipt_label(truck.classify(p));maximum=max(maximum,float(np.linalg.norm(p-start,axis=1).max()))
            linear=float(np.linalg.norm(final['rigid_qvel'][i,:3]));angular=float(np.linalg.norm(final['rigid_qvel'][i,3:]))
            motion=maximum<=measurement['rock_max_sample_displacement_m'] and linear<=.01 and angular<=.01
            accepted=int(persistent_samples.sum()) if motion else 0;liters=float(accepted*weight*1000);fraction=accepted/len(samples);accepted_mass=float(record['mass_kg']*fraction)
            rock_liters+=liters;rock_mass+=accepted_mass
            quant['rocks'].append(dict(rock_id=record['rock_id'],stock_sample_volume_liters=stock,stable_sample_volume_liters=liters,stable_sample_mass_kg=accepted_mass,
                sample_persistence_fraction=float(persistent_samples.mean()),motion_criterion_passed=bool(motion),maximum_sample_displacement_m=maximum,linear_speed_m_s=linear,angular_speed_rad_s=angular))
        quant.update(stable_rock_reference_volume_liters=rock_liters,stable_rock_sample_mass_kg=rock_mass,rock_stock_reference_volume_liters=rock_stock_liters,
            stable_mixed_reference_volume_liters=quant['soil']['stable_original_reference_volume_liters']+rock_liters,
            stable_mixed_sample_mass_kg=quant['soil']['stable_mass_kg']+rock_mass,
            scope='Fixed3s/1cm/1cm-s development screening; soil physical mass and original material reference volume + rock reference sample fractions. Not calibrated bulk volume or official acceptance.',formal_bulk_volume_qualified=False)
        timeseries=[]
        for index in [0,1220,1960,2080,2140,2200,2260]:
            lab=truck.classify(np.array(cloud[index]));timeseries.append(dict(frame=index,core=int((lab==CAVITY).sum()),above_rim=int((lab==ABOVE_RIM).sum()),boundary=int((lab==BOUNDARY).sum()),receipt_stock=int(receipt_label(lab).sum())))
        details=dict(checks=c,quantities=quant,initial_above_rim_cohorts=groups,final_soil_observation=obs,
            final_adjacency_sensitivity=graph_diagnostic(final['particle_q'],labels),receipt_stock_time_series=timeseries,
            max_tracking_error_m_or_rad=max(x['axis_error'] for x in clear),source=case['source'],frames=2260,prefix_frames=180)
        cases[name]=details;checks.update({name+'_'+k:bool(v) for k,v in c.items()});print(name,'audit checks',len(c),'stable reference liters',quant['stable_mixed_reference_volume_liters'],flush=True)
    checks['frozen_model_unchanged']=sha(ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip')=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
    a,b=[cases[k]['quantities'] for k in ['better_manual','frozen_residual']]
    comparison=dict(stable_reference_volume_difference_liters=b['stable_mixed_reference_volume_liters']-a['stable_mixed_reference_volume_liters'],
        stable_reference_volume_difference_percent=(b['stable_mixed_reference_volume_liters']/a['stable_mixed_reference_volume_liters']-1)*100,
        stable_sample_mass_difference_kg=b['stable_mixed_sample_mass_kg']-a['stable_mixed_sample_mass_kg'],
        scope='One paired development run; fixed reference quantities, not statistical significance or calibrated bulk volume')
    result=dict(status='passed' if all(checks.values()) else 'failed',check_count=len(checks),checks=checks,cases=cases,comparison=comparison,measurement=measurement,
        physical_frames=batch['physical_frames'],training_updates=0,audit_physical_frames=0,wall_seconds=time.monotonic()-started)
    write(RUN/'read-only-audit.json',result);print(encode({k:result[k] for k in ['status','check_count','comparison','wall_seconds']}),flush=True);assert all(checks.values())

if __name__=='__main__':main()
