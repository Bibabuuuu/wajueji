"""D151 CPU-only local response probes using the unmodified frozen bridge."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from layered_47_cycle_bridge import native_restore_tokens

W=Path(__file__).resolve().parent;ROOT=W.parents[1]
OUT=ROOT/'runs/contact-effective-response-20260924-v1'
BASE=ROOT/'runs/contact-geometric-probe-20260924-v9-handoff/contact1200'
NEW=ROOT/'runs/contact-coupling-120hz-20260924-v1/candidate120'
BUNDLE=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
DT=1/60;LENGTH=.6
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]

def initial_line(arrays,meta):
    c=meta['controller_state']
    values=[*arrays['mechanism_qpos'],*arrays['rigid_qpos'].ravel(),*arrays['mechanism_qvel'],*arrays['rigid_qvel'].ravel(),
            *c['controller_target'],*c['controller_target_velocity'],c['bucket_integral_effort'][3],*native_restore_tokens(c,47)]
    return 'init_cycle '+' '.join(format(float(x),'.17g') for x in values)

def step_line(t,command,wrenches):
    values=['step',format(t,'.17g'),format(1/120,'.17g'),'40',*[format(float(x),'.17g') for x in command],str(len(wrenches))]
    for name,w in wrenches.items():
        values.append(name)
        for key in ('force','torque','point'):values.extend(format(float(x),'.17g') for x in w[key])
    return ' '.join(values)

def probe(arrays,meta,command,wrenches,label):
    inputs='\n'.join([initial_line(arrays,meta),step_line(meta['controller_state']['native_time_s'],command,wrenches),
        step_line(meta['controller_state']['native_time_s']+1/120,command,wrenches),'close',''])
    env=dict(os.environ,EXCAVATOR_MUJOCO_MESH_DIR=str(BUNDLE/'excavator-resources/mesh'),EXCAVATOR_MUJOCO_TEXTURE_DIR=str(BUNDLE/'excavator-resources/texture'))
    started=time.perf_counter()
    p=subprocess.run([str(BASE/'physics/runtime/capacity-bridge'),str(BASE/'physics/runtime/audit.xml')],input=inputs,text=True,capture_output=True,env=env,timeout=10)
    (OUT/(label+'.stdout')).write_text(p.stdout);(OUT/(label+'.stderr')).write_text(p.stderr)
    states=[json.loads(x[7:]) for x in p.stdout.splitlines() if x.startswith('@STATE ')]
    if p.returncode or len(states)!=3:raise RuntimeError(f'native_probe_failed:{label}:{p.returncode}:{p.stderr}')
    start,half,end=states
    c=meta['controller_state']
    checks=dict(qpos=np.array_equal(start['mechanism_qpos'],arrays['mechanism_qpos']),qvel=np.array_equal(start['mechanism_qvel'],arrays['mechanism_qvel']),
       rocks_qpos=np.array_equal([x['qpos'] for x in start['rocks']],arrays['rigid_qpos']),rocks_qvel=np.array_equal([x['qvel'] for x in start['rocks']],arrays['rigid_qvel']),
       target=start['controller_target']==c['controller_target'],target_velocity=start['controller_target_velocity']==c['controller_target_velocity'],
       integral=start['bucket_integral_effort']==c['bucket_integral_effort'],active=start['bucket_active_curl_effort']==c['bucket_active_curl_effort'],
       warmstart=start['cycle_native_warmstart']==c['native_warmstart'],qacc=start['cycle_native_qacc']==c['native_qacc'],ctrl=start['cycle_native_ctrl']==c['native_ctrl'],
       last_time=start['cycle_controller_last_time']==c['native_last_time_s'],native_time=start['time_s']==c['native_time_s'],dt=start['cycle_native_timestep']==c['native_timestep_s'],
       elapsed=abs(end['time_s']-start['time_s']-DT)<1e-9,finite=all(np.isfinite(s['mechanism_qvel']).all() for s in states),
       closure=all(s['closure_m']<=.001 for s in states),forbidden=all(s['forbidden_contact_count']==0 for s in states))
    if not all(checks.values()):raise ValueError('probe_restore_or_guard:'+str(checks))
    twist=np.r_[end['bucket_linear_velocity_world_m_s'],LENGTH*np.asarray(end['bucket_angular_velocity_world_rad_s'])]
    return dict(label=label,wall_seconds=time.perf_counter()-started,checks=checks,twist_scaled=twist.tolist(),states=states)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=read(W/'protocol.json')
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('input_changed:'+item['path'])
    all_results=[];started=time.monotonic()
    oldw=rows(ROOT/'runs/contact-response-diagnosis-20260924-v2/observed1200/wrench-split.ndjson')[-1]['applied']
    contexts=[('old972_zero',BASE,'zero'),('old972_held_previous_load',BASE,'held_previous'),('new1160_zero',NEW,'zero')]
    for name,run,context in contexts:
        with np.load(run/'physics/failure.npz') as data:arrays={k:data[k].copy() for k in data.files}
        meta=read(run/'physics/failure.npz.json');command=rows(run/'actions.ndjson')[-1]['applied_command_si']
        origin=rows(run/'physics/frames.ndjson')[-1]['after_state']['bucket_origin_world_m']
        if context=='zero':wrenches={'bucket':dict(force=[0.]*3,torque=[0.]*3,point=origin)}
        else:wrenches=copy.deepcopy(oldw)
        # Convert bucket wrench to current origin without changing its wrench.
        bucket=wrenches['bucket'];bucket['torque']=(np.asarray(bucket['torque'])+np.cross(np.asarray(bucket['point'])-origin,bucket['force'])).tolist();bucket['point']=origin
        trials=[]
        for repeat in range(2):trials.append(probe(arrays,meta,command,wrenches,name+f'-baseline-{repeat}'))
        if trials[0]['states']!=trials[1]['states']:raise ValueError('native_baseline_repeat_not_exact')
        matrices={}
        for amplitude in (100.,200.):
            matrix=np.zeros((6,6))
            for axis in range(6):
                paired=[]
                for sign in (-1,1):
                    if time.monotonic()-started>180:raise TimeoutError('D151_CPU_probe_budget')
                    w=copy.deepcopy(wrenches)
                    if axis<3:w['bucket']['force'][axis]+=sign*amplitude
                    else:w['bucket']['torque'][axis-3]+=sign*amplitude*LENGTH
                    t=probe(arrays,meta,command,w,name+f'-a{amplitude:g}-axis{axis}-sign{sign}')
                    paired.append(np.asarray(t['twist_scaled']));trials.append(t)
                matrix[:,axis]=(paired[1]-paired[0])/(2*amplitude*DT)
            matrices[str(amplitude)]=matrix.tolist()
        result=dict(context=name,external_load_context=context,source=str(run/'physics/failure.npz'),command=command,
             origin_world_m=origin,dt_s=DT,characteristic_length_m=LENGTH,
             response_definition='C=d[v_origin,L*omega]/d[F,torque_origin/L]/dt; all world coordinates. Endpoint reported body velocity has original last-substep timing.',
             matrices=matrices,probe_count=len(trials),trials=trials)
        all_results.append(result)
        (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('CONTEXT_DONE',name,'probes',len(trials),'elapsed',time.monotonic()-started,flush=True)
    result=dict(status='CPU_response_measured',contexts=all_results,wall_seconds=time.monotonic()-started,
        full_scene_MPM_steps=0,native_diagnostic_steps=80*26*3,
        inputs_unchanged=all(sha(x['path'])==x['sha256'] for x in protocol['inputs']),
        limitations=['Failure snapshots used only for native diagnostic copies, never admitted as trajectory continuation.',
            '972 held-load context holds its previous MPM wrench, not a recomputed future MPM force.',
            '1160 last applied MPM wrenches were not saved; zero-external context does not reproduce its loaded future.',
            'No new MPM, soil deformation, policy inference or trained-model changes.'])
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',result['wall_seconds'],flush=True)
if __name__=='__main__':main()
