"""CPU-only source audit, cached above-rim diagnosis, and paired withdrawal plan."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from exit_common import *
from planning.cycle_geometry import MOUTH,tool_pose,solve_pose
from planning.approach import sampled_approach
import time,signal

def main():
    CPU.mkdir(exist_ok=False);started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU preflight 420s ceiling')));signal.alarm(420)
    old=read(PREVIOUS/'read-only-audit.json');assert old['status']=='passed' and old['full_pair_task_complete']
    report=dict(status='checking',cases={},new_physical_frames=0,training_updates=0,original_audit_sha256=sha(PREVIOUS/'read-only-audit.json'),
        recipe=dict(stages=['CLEAR_TRUCK','WITHDRAW_SWING','SETTLE'],clear_mouth_z_m=3.5,clear_pitch_rad=1.,exit_yaw_rad=0.,settle_frames=300,
        speed_limits_si=[.1,.018,.018,.018],dt_s=1/60,validation_substeps=2),formal_bulk_volume_qualified=False)
    try:
        for name in ['better_manual','frozen_residual']:
            info=source_info(name);r=CpuRuntime(info['source'],CPU/(name+'-cpu'),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
            original={k:v.copy() for k,v in r.arrays.items()};q=r.arrays['mechanism_qpos'];cloud=r.arrays['particle_q']
            cache=np.load(PREVIOUS/name/'display/particles.npy',mmap_mode='r')
            obs,classes,distance=soil_observation(r,cloud,q,r.arrays['particle_qd'],np.array(cache[-31]))
            obs['adjacency_sensitivity']=graph_diagnostic(cloud,classes)
            obs['source_above_near50_ids']=np.flatnonzero((classes==ABOVE_RIM)&(distance<=.05)).tolist()
            obs['source_above_far50_ids']=np.flatnonzero((classes==ABOVE_RIM)&(distance>.05)).tolist()
            del cache
            state=read(str(info['source'])+'.control.json');c=controller(r,state)
            checks=dict(native_not_started=r.bridge is None,source_full_native_history=r.metadata['controller_state']['native_state_kind']=='full',
                high_controller_previous_preserved=np.array_equal(c.previous,state['previous_command']),high_controller_bias_preserved=np.array_equal(c.tracking.bias,state['load_bias']))
            # Project only the CPU nominal passive linkage; physical arrays stay unchanged.
            r.chain.q=q.copy();start=r.chain.advance(np.zeros(4),1/60)[0]
            truck=receiver();ycenter=(truck.rim[:,1].min()+truck.rim[:,2].max())/2
            center=truck.pose@np.array([0.,ycenter,0.,1.])
            goal=solve_pose(r.assembly,r.chain,start,np.r_[center[:2],3.5],1.,MOUTH)
            turn=goal.copy();turn[7]=0.
            segments=[];checker=FixedRouteClearance(r)
            for stage,target in [('CLEAR_TRUCK',goal),('WITHDRAW_SWING',turn)]:
                def advance(q,v,dt):
                    r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
                path=sampled_approach(start,target,axes=r.chain.axes,speed_limits=np.array([.1,.018,.018,.018]),
                    advance=advance,check_pose=checker.check,dt=1/60,max_steps=2400,validation_substeps=2)
                diagnostic=path.pop('diagnostics');segments.append(dict(name=stage,path=path,
                    minimum_clearance={k:min(d[k] for d in diagnostic) for k in diagnostic[0] if diagnostic[0][k] is not None}))
                start=target.copy();print(name,stage,path['steps'],flush=True)
            count=sum(s['path']['steps']+30 for s in segments)+300
            checks['bounded_exit_frames']=count<=2400;checks['whole_bucket_outside_opening']=bucket_opening_gap(r,start)>.1
            before_arrays=all(np.array_equal(v,r.arrays[k]) for k,v in original.items());checks['source_arrays_unchanged']=before_arrays
            # Check every nominal command with the same controller and original guards.
            current=q.copy();native_target=np.array(r.metadata['controller_state']['controller_target']);maximum=0.;commands=0
            for s in segments:
                for desired,velocity in list(zip(s['path']['qpos'][1:],s['path']['velocity'],strict=True))+[(s['path']['qpos'][-1],np.zeros(4))]*30:
                    c.check(current,s['name'],physical=True);prev=c.previous.copy();cmd,_=c.command(s['name'],current,native_target,desired,velocity)
                    maximum=max(maximum,float(np.max(abs((cmd-prev)/LIMITS))));commands+=1
                    current=desired.copy();native_target=current[r.chain.axes]+c.tracking.bias
            for _ in range(300):
                prev=c.previous.copy();cmd,_=c.command('SETTLE',current,native_target,current,np.zeros(4));maximum=max(maximum,float(np.max(abs((cmd-prev)/LIMITS))));commands+=1
                native_target=current[r.chain.axes]+c.tracking.bias
            checks['all_nominal_commands_checked']=commands==count;checks['original_slew_preserved']=maximum<=2/60+1e-9
            path=CPU/(name+'-plan.npz');np.savez(path,**{s['name']+'_qpos':s['path']['qpos'] for s in segments},**{s['name']+'_velocity':s['path']['velocity'] for s in segments})
            case=dict(source=info,prefix_source=source_info(name,'periodic-02520'),diagnostic=obs,checks=checks,
                frames=count,plan_sha256=sha(path),final_opening_gap_m=bucket_opening_gap(r,start),maximum_normalized_command_change=maximum,
                stages=[dict(name=s['name'],frames=s['path']['steps'],minimum_clearance=s['minimum_clearance']) for s in segments])
            report['cases'][name]=case;write(CPU/'report.json',report);assert all(checks.values())
        report['status']='cpu_passed'
    except BaseException as e:report.update(status='failed',failure=repr(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;write(CPU/'report.json',report)
        print(encode({'status':report['status'],'wall_seconds':report['wall_seconds'],'failure':report.get('failure')}),flush=True)
    write(CPU/'dependency-sha256.json',{str(p):sha(p) for p in [WORK/'exit_common.py',WORK/'prepare_exit_settle.py',ROOT/'src/v4/layered_47_cycle_runtime.py',ROOT/'src/v4/layered_47_cycle_checkpoint.py',ROOT/'terrain/truck_receiver.py',ROOT/'work/contact-second-fixed-route-20260925-v1/controller.py',ROOT/'work/contact-second-finish-unload-20260925-v4/transport_feedback.py']})

if __name__=='__main__':main()
