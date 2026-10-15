"""Source-bound D165 transport candidate. CPU-only; native entry and step blocked."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT/'work/contact-second-fixed-route-20260925-v1'),str(ROOT)]
from layered_47_cycle_runtime import CycleRuntime
from clearance import FixedRouteClearance
from controller import TransportController
from run_layered_47_sparse_manual_contact_episode import _clean
def encode(value, **options):
    return json.dumps(_clean(value), **options)

class CpuTransportRuntime(CycleRuntime):
    def __init__(self, source, output, expected_sha):
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('source_sha_mismatch')
        super().__init__(source,output,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
        if self.metadata['stage']!='hold' or self.metadata['cycle_index']!=1 or self.metadata.get('training_reset') is not None:
            raise ValueError('not_original_first_hold')
    def __enter__(self):
        raise RuntimeError('cpu_candidate_native_entry_forbidden')
    def step(self,command):
        raise RuntimeError('cpu_candidate_physical_step_forbidden')

def main():
    out=WORK/'transport-cpu-candidate-v2';out.mkdir(exist_ok=False)
    started=time.monotonic()
    sources=json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources']
    report=dict(status='checking',sources={},new_physical_frames=0,training_updates=0,
        native_restore_tested=False,full_path_tested=False,physical_transport_tested=False,
        scope='CPU source binding, controller/actual command handoff and original 150ms preview only; no production changes')
    try:
        for name in ['better_manual','frozen_residual']:
            source=sources[name];r=CpuTransportRuntime(source['source'],out/name,source['sha256'])
            original={k:v.copy() for k,v in r.arrays.items()}
            actions=Path(source['source']).parent/'actions.ndjson'
            last=json.loads(actions.read_text().splitlines()[-1]);previous=np.array(last['applied_si'])
            if last['hold_frame']!=29 or not np.array_equal(previous,np.zeros(4)):
                raise ValueError('hold_handoff_history')
            checker=FixedRouteClearance(r)
            control=TransportController(r,checker,previous)
            q=r.arrays['mechanism_qpos'].copy()
            target=np.array(r.metadata['controller_state']['controller_target'])
            checks={'native_not_started':r.bridge is None,'full_source_arrays_exact':all(np.array_equal(v,r.arrays[k]) for k,v in original.items())}
            static=control.check(q,'CLEAR_PILE',physical=True)
            before=control.previous.copy()
            command,detail=control.command('CLEAR_PILE',q,target,q,np.zeros(4))
            checks['handoff_retains_bucket']=abs(command[3])<1e-10
            checks['handoff_slew_preserved']=float(np.max(abs((command-before)/np.array([.12,.0225,.0225,.0225]))))<=2/60+1e-9
            checks['load_bias_inherited']=np.allclose(control.tracking.bias,target-q[r.chain.axes],atol=1e-10,rtol=0)
            for method,expected in [(lambda:r.__enter__(),'cpu_candidate_native_entry_forbidden'),(lambda:r.step(command),'cpu_candidate_physical_step_forbidden')]:
                try:method()
                except RuntimeError as e:checks[expected]=str(e)==expected
                else:raise AssertionError('CPU guard bypassed')
            checks['arrays_unchanged']=all(np.array_equal(v,r.arrays[k]) for k,v in original.items())
            report['sources'][name]=dict(source=source,checks=checks,first_command=command.tolist(),
                controller_detail=detail,static_measurement=static,actual_cylinder_m=float(q[15]),
                minimum_original_cylinder_m=control.minimum_cylinder,controller_target=target.tolist(),
                actual_axes=q[r.chain.axes].tolist(),prior_actual_command=previous.tolist())
            print(encode({'source':name,'checks':checks,'first_command':command.tolist()}),flush=True)
            if not all(checks.values()):raise AssertionError('CPU handoff checks')
        report['status']='cpu_handoff_passed'
    except BaseException as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out/'report.json').write_text(encode(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
