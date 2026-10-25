"""CPU-only endpoint, whole-path and nominal-command verification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,signal,hashlib,shutil
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path.insert(0,str(WORK))
from transport_cpu_candidate_v3 import CpuTransportRuntime,TransportController,encode
from transport_endpoint_candidate_v2 import build_plan,locked_lift_pose,bind_endpoint_solver,original_plan
from controller import LIMITS

def main():
    out=WORK/'transport-endpoint-preflight-v3';out.mkdir(exist_ok=False)
    sources=json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources']
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('CPU preflight 180-second limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    report=dict(status='checking',sources={},new_physical_frames=0,training_updates=0,native_restore_tested=False,
        physical_transport_tested=False,scope='CPU nominal paths/commands only; fixed-route static geometry and original guards; no dynamic tracking proof')
    try:
        for name in ['frozen_residual']:
            source=sources[name];r=CpuTransportRuntime(source['source'],out/name,source['sha256'])
            original={k:v.copy() for k,v in r.arrays.items()};q0=original['mechanism_qpos']
            limits_before=dict(r.chain.limits);checks={};entry=dict(checks=checks,nominal_commands=0,phase='endpoint_self_checks')
            report['sources'][name]=entry
            bad=q0.copy();bad[15]=r.chain.limits[15][1]+.001
            try:locked_lift_pose(r.assembly,r.chain,bad,np.array([3.9,0,3.3]))
            except ValueError as e:checks['physical_overrun_rejected']=str(e)=='held_cylinder_outside_physical_bounds'
            else:raise AssertionError('physical boundary bypassed')
            old_solver=original_plan.solve_pose;old_source=original_plan.SOURCE
            try:
                with bind_endpoint_solver(r,Path(source['source'])):raise RuntimeError('test_cleanup')
            except RuntimeError as e:assert str(e)=='test_cleanup'
            checks['solver_scope_restores_after_exception']=original_plan.solve_pose is old_solver and original_plan.SOURCE==old_source
            entry['phase']='build_plan'
            prior=WORK/'transport-endpoint-preflight-v2'
            previous=json.loads((prior/'report.json').read_text())
            assert previous['error'].startswith("TimeoutError") and previous['sources']['better_manual']['phase']=='complete'
            path_report=previous['sources'][name]['plan']
            assert path_report['source']==source['source']
            plan_path=prior/(name+'-plan')/'plan.npz'
            reused_sha=hashlib.sha256(plan_path.read_bytes()).hexdigest()
            target_plan=out/(name+'-plan');target_plan.mkdir()
            shutil.copyfile(plan_path,target_plan/'plan.npz')
            with np.load(target_plan/'plan.npz') as data:
                segments=[dict(name=s['name'],path={'qpos':data[s['name']+'_qpos'].copy(),'velocity':data[s['name']+'_velocity'].copy()}) for s in path_report['stages']]
            from clearance import FixedRouteClearance
            checker=FixedRouteClearance(r)
            path_report['reused_completed_geometric_plan']={'path':str(plan_path),'sha256':reused_sha,'reason':'Previous CPU command check hit its 180-second limit; geometry had already completed'}
            entry['plan']=path_report
            control=TransportController(r,checker,np.zeros(4));maximum=0.;count=0
            for segment in segments:
                qs=segment['path']['qpos'];vs=segment['path']['velocity'];stage=segment['name'];entry['phase']='commands_'+stage
                checks[stage+'_physical_joint_bounds']=all(control.r.projector.physical_bounds_ok(q) for q in qs)
                if stage in ('CLEAR_PILE','ALIGN'):checks[stage+'_cylinder_held']=bool(np.max(abs(qs[:,15]-q0[15]))<1e-8)
                for index in range(len(vs)+30+(180 if stage=='DUMP' else 0)):
                    q=r.arrays['mechanism_qpos'] if count==0 else qs[min(index,len(qs)-1)]
                    desired=qs[min(index+1,len(qs)-1)];velocity=vs[index] if index<len(vs) else np.zeros(4)
                    target=r.metadata['controller_state']['controller_target'] if count==0 else q[r.chain.axes]+control.tracking.bias
                    before=control.previous.copy();command,detail=control.command(stage,q,target,desired,velocity)
                    maximum=max(maximum,float(np.max(abs((command-before)/LIMITS))));count+=1;entry['nominal_commands']=count
                print(name,stage,count,flush=True)
            checks['all_nominal_commands_checked']=count==sum(len(s['path']['velocity'])+30+(180 if s['name']=='DUMP' else 0) for s in segments)
            checks['slew_preserved']=maximum<=2/60+1e-9
            checks['arrays_unchanged']=all(np.array_equal(v,r.arrays[k]) for k,v in original.items())
            checks['native_not_started']=r.bridge is None
            checks['source_bound']=path_report['source']==source['source'] and hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['sha256']
            checks['limits_unchanged']=r.chain.limits==limits_before
            checks['solver_scope_restored']=original_plan.solve_pose is old_solver and original_plan.SOURCE==old_source
            entry.update(phase='complete',maximum_normalized_step=maximum)
            assert all(checks.values())
            (out/(name+'-plan')/'report.json').write_text(encode(path_report,ensure_ascii=False,indent=2)+'\n')
        report['status']='endpoint_nominal_cpu_passed'
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started
        (out/'report.json').write_text(encode(report,ensure_ascii=False,indent=2)+'\n')
        print(encode({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],'error':report.get('error')}),flush=True)

if __name__=='__main__':main()
