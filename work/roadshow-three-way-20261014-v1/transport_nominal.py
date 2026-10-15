"""Bounded CPU plan/nominal command checks, with native/physical entry prohibited."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,signal,hashlib
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path.insert(0,str(WORK))
from transport_cpu_candidate_v3 import CpuTransportRuntime,TransportController,encode
import plan
from controller import LIMITS

def main():
    out=WORK/'transport-nominal-cpu-v1';out.mkdir(exist_ok=False)
    sources=json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources']
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('CPU nominal preflight 180-second limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(180)
    report=dict(status='checking',sources={},new_physical_frames=0,training_updates=0,native_restore_tested=False,
        physical_transport_tested=False,scope='Ideal nominal path CPU checks, not actual loaded tracking; fixed transport terrain exclusion is disclosed; physical particles unchanged')
    try:
        for name in ['better_manual','frozen_residual']:
            source=sources[name];r=CpuTransportRuntime(source['source'],out/name,source['sha256'])
            original={k:v.copy() for k,v in r.arrays.items()}
            # SOURCE is used solely in the old builder's report. Bind it to the
            # actual D165 source; do not call or change the old source loader.
            prior_source=plan.SOURCE;plan.SOURCE=Path(source['source'])
            try:segments,checker,path_report=plan.build_plan(r,out/(name+'-plan'))
            finally:plan.SOURCE=prior_source
            control=TransportController(r,checker,np.zeros(4));count=0;maximum=0.;checks={}
            for segment in segments:
                qs=segment['path']['qpos'];vs=segment['path']['velocity'];stage=segment['name']
                for index in range(len(vs)+30+(180 if stage=='DUMP' else 0)):
                    q=r.arrays['mechanism_qpos'] if count==0 else qs[min(index,len(qs)-1)]
                    desired=qs[min(index+1,len(qs)-1)];velocity=vs[index] if index<len(vs) else np.zeros(4)
                    target=r.metadata['controller_state']['controller_target'] if count==0 else q[r.chain.axes]+control.tracking.bias
                    before=control.previous.copy();command,detail=control.command(stage,q,target,desired,velocity)
                    maximum=max(maximum,float(np.max(abs((command-before)/LIMITS))));count+=1
                print(name,stage,count,flush=True)
            checks['all_nominal_commands_checked']=count==sum(len(s['path']['velocity'])+30+(180 if s['name']=='DUMP' else 0) for s in segments)
            checks['slew_preserved']=maximum<=2/60+1e-9
            checks['arrays_unchanged']=all(np.array_equal(v,r.arrays[k]) for k,v in original.items())
            checks['native_not_started']=r.bridge is None
            checks['source_bound']=path_report['source']==source['source'] and hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['sha256']
            report['sources'][name]=dict(checks=checks,plan=path_report,nominal_commands=count,maximum_normalized_step=maximum)
            assert all(checks.values())
        report['status']='nominal_cpu_passed'
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started
        (out/'report.json').write_text(encode(report,ensure_ascii=False,indent=2)+'\n')
        print(encode({'status':report['status'],'elapsed_seconds':report['elapsed_seconds'],'error':report.get('error')}),flush=True)

if __name__=='__main__':main()
