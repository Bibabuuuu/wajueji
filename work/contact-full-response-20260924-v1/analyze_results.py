"""Timing, half-frame response, independent exact endpoint kinematics; no dynamics."""
import json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parents[1];RUN=ROOT/'runs/contact-full-response-20260924-v1/candidate60'
read=lambda p:json.loads(Path(p).read_text());rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def window(halves,actions,lo,hi):
    h=[x for x in halves if lo<=x['frame']<=hi]
    if not h:return None
    v=np.array([x['mechanism_qvel'][14] for x in h]);commands={x['frame']:x['applied_command_si'][3] for x in actions};i=int(np.argmax(np.abs(v)));j=int(np.argmax(np.abs(np.diff(v)))) if len(v)>1 else 0
    return dict(range=[lo,hi],half_count=len(h),maximum_absolute_speed_m_s=float(max(abs(v))),maximum_adjacent_half_jump_m_s=float(max(abs(np.diff(v)))) if len(v)>1 else 0.,
        peak=dict(frame=h[i]['frame'],half=h[i]['half'],velocity=float(v[i]),command=commands[h[i]['frame']]),
        largest_jump=dict(from_frame=h[j]['frame'],from_half=h[j]['half'],to_frame=h[min(j+1,len(h)-1)]['frame'],to_half=h[min(j+1,len(h)-1)]['half'],before=float(v[j]),after=float(v[min(j+1,len(h)-1)])),
        direction_reversals_over_1mm_s=int(sum((v[:-1]*v[1:]<0)&(abs(v[:-1])>.001)&(abs(v[1:])>.001))))
def main():
    report=read(RUN/'report.json');t=rows(RUN/'physics/timing.ndjson');ft=rows(RUN/'frame-timing.ndjson');halves=rows(RUN/'physics/native-halves.ndjson');responses=rows(RUN/'physics/response.ndjson');actions=rows(RUN/'actions.ndjson');n=len(t)
    timings={}
    for count in sorted({min(120,n),n}):
        sim=count/60;timings[str(count)]=dict(simulated_seconds=sim,loop_seconds_per_sim_second=sum(x['wall_seconds'] for x in ft[:count])/sim,
        **{k:sum(x[k] for x in t[:count])/sim for k in ('physics_wall_seconds','runtime_step_wall_seconds','mpm_wall_seconds','mujoco_wall_seconds','predictor_wall_seconds')})
        timings[str(count)]['mpm_excluding_shadow_seconds_per_sim_second']=timings[str(count)]['mpm_wall_seconds']-timings[str(count)]['predictor_wall_seconds']
    # Kinematics at actual frame endpoints, fixed start wrench origin; rock poses
    # don't affect the bucket tree Jacobian, so use saved rock poses/zero rock qvel.
    with np.load(report['source']['path']) as initial:rq=initial['rigid_qpos'].ravel();rv=np.zeros_like(initial['rigid_qvel']).ravel()
    lines=[]
    for action,end in zip(actions[:n],halves[1:2*n:2],strict=True):
        values=[*action['before_state']['bucket_origin_world_m'],*end['mechanism_qpos'],*rq,*end['mechanism_qvel'],*rv]
        lines.append(' '.join(format(x,'.17g') for x in values))
    helper=ROOT/'work/contact-effective-response-20260924-v1/endpoint-velocity';model=RUN/'physics/runtime/audit.xml'
    p=subprocess.run([str(helper),str(model)],input='\n'.join(lines)+'\n',text=True,capture_output=True,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr)
    actual=np.array([json.loads(x) for x in p.stdout.splitlines()]);prediction=[];forces=[]
    for response,end in zip(responses,halves[1:2*n:2],strict=True):
        F=np.r_[end['applied_bucket_mpm_force_world_n'],end['applied_bucket_mpm_torque_world_nm']];pending=response['pending_force_predicted'];known=np.r_[pending['force'],pending['torque']]
        prediction.append(np.array(response['free_twist'])+np.array(response['response_world'])@(F-known)/60);forces.append(np.linalg.norm(F[:3]))
    prediction=np.array(prediction);scale=np.array([1,1,1,.6,.6,.6]);error=(actual-prediction)*scale;loaded=np.array(forces)>100
    model_check=dict(definition='endpoint twist = private free prediction (includes pending reaction) + C*(total bucket wrench - pending wrench)*dt',linear_speed_error_max_m_s=float(np.linalg.norm(error[:,:3],axis=1).max()),angular_speed_error_max_rad_s=float(np.linalg.norm(error[:,3:]/.6,axis=1).max()),loaded_frames=int(loaded.sum()),limitation='Full nonlinear scene may couple rock MPM loads through direct native contacts; these cross-body response blocks are not in bucket C.')
    if loaded.any():model_check.update(loaded_scaled_twist_rms_error=float(np.sqrt(np.mean(error[loaded]**2))),loaded_scaled_actual_rms=float(np.sqrt(np.mean((actual[loaded]*scale)**2))))
    (RUN/'endpoint-response-check.json').write_text(json.dumps(dict(actual=actual.tolist(),prediction=prediction.tolist(),model_check=model_check),indent=2)+'\n')
    result=dict(status=report['status'],failure=report.get('failure'),frames=n,wall_seconds=report['wall_seconds'],startup_wall_seconds=report['startup_wall_seconds'],timing=timings,
        windows={name:window(halves,actions,lo,hi) for name,lo,hi in [('all',1,n),('old_failure',940,972),('old_120hz_late',1118,min(n,1160)),('late',1200,n)]},
        final_phase=report.get('final_phase'),events=report.get('events'),same_instance_gates=report.get('same_instance_gates'),source_unchanged=report['source_unchanged'],full_loop_overhead_first120=(timings['120']['loop_seconds_per_sim_second']/11.636036604998168-1) if n>=120 else None,
        endpoint_response_check=model_check,physics_steps_added=0)
    (W/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
