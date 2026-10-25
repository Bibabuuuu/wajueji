"""CPU-only independent verification of native short transport probes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
RUN=ROOT/'runs/d165-transport-native-probe-20261014-v1'
sys.path[:0]=[str(WORK),str(ROOT/'src/v4'),str(ROOT)]
from transport_cpu_candidate_v3 import CpuTransportRuntime,TransportController,encode
from clearance import FixedRouteClearance
from live_bucket_fill import MixedFillEstimator,load_config
import run_layered_47_sparse_manual_contact_episode as manual

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
report=read(RUN/'report.json');budget=read(ROOT/'runs/d165-transport-native-probe-20261014-v1-budget.json')
protocol=read(WORK/'transport-native-probe-protocol-v1.json')
checks={'two_cases_complete':report['status']=='completed' and set(report['cases'])=={'better_manual','frozen_residual'},
        'physical_scope':report['physical_frames']==240 and budget['seconds']==1200 and budget['per_case_max_frames']==120,
        'budget_identity':sha(ROOT/'runs/d165-transport-native-probe-20261014-v1-budget.json')==protocol['budget_sha256'],
        'runner_identity':sha(WORK/'transport_physical_probe.py')==report['runner_sha256']==protocol['runner_sha256'],
        'no_training':report['training_updates']==0}
cases={}
for name in ['better_manual','frozen_residual']:
    out=RUN/name;case=read(out/'report.json');source=Path(case['source']['source'])
    actions=rows(out/'actions.ndjson');frames=rows(out/'physics/frames.ndjson');clear=rows(out/'clearance.ndjson')
    states=read(out/'display/states.json');particles=np.load(out/'display/particles.npy',mmap_mode='r')
    source_sha=sha(source);control_meta=read(str(source)+'.json')['controller_state']
    c={'status_short_passed':case['status']=='short_probe_passed' and case['frames']==120,
       'zero_24_arrays_exact':len(case['zero_arrays_exact'])==24 and all(case['zero_arrays_exact'].values()),
       'native_internal_state_exact':max(case['native_restore_errors'].values())<=1e-12,
       'sweep_internal_state_exact':case['sweep_restore_exact'],
       'frame_count_and_order':len(actions)==len(frames)==len(clear)==120 and [f['frame'] for f in frames]==list(range(1,121)),
       'only_initial_lift_phase':all(a['phase']=='CLEAR_PILE' for a in actions),
       'all_hard_guards':all(f['hard_guard'] and all(f['hard_guard_checks'].values()) for f in frames),
       'all_material_guards':all(f['hencky_material_guard']['safe'] for f in frames),
       'all_sweep_guards':all(f['moving_bucket_sweep']['safe'] for f in frames),
       'commands_equal_dispatched':all(a['command']==f['applied_command_si'] for a,f in zip(actions,frames)),
       'no_active_opening_command':all(abs(a['command'][3])<1e-10 for a in actions),
       'track_guard_preserved':max(c['axis_error'] for c in clear)<=.025,
       'cache_count':len(states)==particles.shape[0]==121,
       'cache_state_matches_frames':all(s==f['after_state'] for s,f in zip(states[1:],frames)),
       'source_unchanged':source_sha==case['source']['sha256'] and sha(str(source)+'.json')==case['source']['metadata_sha256'],
       'not_full_transport':not case['stages'] and not case['formal_mixed_volume_qualified'],
       'short_gate_saved':case['short_gate_passed'] and sha(out/'short-120.npz')==case['short_gate_checkpoint']['sha256'],
       'final_sha':sha(out/'final.npz')==case['final_checkpoint']['sha256']}
    with np.load(out/'zero.npz') as a,np.load(source) as b:
        c['saved_zero_all_arrays_exact']=len(a.files)==24 and all(np.array_equal(a[k],b[k]) for k in a.files)
        c['cache_zero_particles_exact']=np.array_equal(particles[0],b['particle_q'])
    with np.load(out/'short-120.npz') as a,np.load(out/'final.npz') as b:
        c['final_matches_short_24_arrays']=len(a.files)==24 and all(np.array_equal(a[k],b[k]) for k in a.files)
        c['cache_final_particles_exact']=np.array_equal(particles[120],b['particle_q'])
    c['all_cache_hashes']=all(sha(out/'display'/k)==v for k,v in case['display']['cache_sha256'].items())
    r=CpuTransportRuntime(source,RUN/('audit-cpu-'+name),case['source']['sha256'])
    checker=FixedRouteClearance(r);control=TransportController(r,checker,np.zeros(4));commands_exact=[];bias_exact=[]
    for a in actions:
        before=a['before_state'];q=np.array(before['mechanism_qpos'])
        command,detail=control.command(a['phase'],q,before['controller_target'],np.array(a['desired']),np.array(a['velocity']))
        commands_exact.append(np.array_equal(command,a['command']));bias_exact.append(np.array_equal(detail['load_bias'],a['detail']['load_bias']))
    c['all_commands_independently_recomputed_exact']=all(commands_exact)
    c['all_load_biases_independently_recomputed_exact']=all(bias_exact)
    c['initial_bias_inherited']=np.array_equal(case['initial_controller_bias'],np.array(control_meta['controller_target'])-r.arrays['mechanism_qpos'][r.chain.axes])
    min_cylinder=min(s['mechanism_qpos'][15] for s in states);maximum_cylinder=max(s['mechanism_qpos'][15] for s in states)
    c['actual_retention_preserved']=min_cylinder>=control.minimum_cylinder and maximum_cylinder<=r.chain.limits[15][1]
    origin_delta=np.array(states[-1]['bucket_origin_world_m'])-np.array(states[0]['bucket_origin_world_m'])
    scene=read(out/'display/scene.json')
    estimator=MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),
        [x['volume_m3'] for x in scene['records']],manual._cache_rock_vertices(Path(scene['root']),scene['records']))
    displays=[]
    for frame in [0,120]:
        s=states[frame];pose=r.assembly.assemble(np.array(s['mechanism_qpos']),include_meshes=False)[1]['bucket']
        displays.append(estimator.measure_snapshot(dict(frame=frame,soil=np.array(particles[frame]),pose=pose,rocks=np.array([x['qpos'] for x in s['rocks']]))))
    cases[name]=dict(checks=c,origin_delta_m=origin_delta.tolist(),minimum_actual_cylinder_m=min_cylinder,
        maximum_actual_cylinder_m=maximum_cylinder,max_axis_error_m=float(max(x['axis_error'] for x in clear)),
        diagnostic_display=displays,display_scope='Two instantaneous diagnostics during motion; not stationary retention or net truck volume',
        physical_frames=120,source_sha256=source_sha)
    checks.update({name+'_'+k:bool(v) for k,v in c.items()})
checks['model_unchanged']=sha(ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip')=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
audit=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),cases=cases,
    physical_frames=240,training_updates=0,audit_new_physical_frames=0,budget_remaining_seconds=budget['epoch_deadline']-time.time())
write=lambda p,v:Path(p).write_text(encode(v)+'\n')
write(RUN/'read-only-audit.json',audit)
print(encode({'status':audit['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],
    'cases':{k:{x:v[x] for x in ['origin_delta_m','max_axis_error_m','minimum_actual_cylinder_m','diagnostic_display']} for k,v in cases.items()}}))
assert all(checks.values())
