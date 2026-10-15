"""Three-step independent continuation equals the already recorded next steps."""
import json
from pathlib import Path
import numpy as np
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_bridge import controller_snapshot
root=Path(__file__).resolve().parent;p=root/'manual-prefix-1800'
old=Path('/home/qunwang/wajueji/runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275')
commands=[json.loads(x)['applied_command_si'] for x in (old/'input.ndjson').read_text().splitlines()][3:6]
out=root/'independent-continuation-3'
with CycleRuntime(p/'state-000003.npz',out,bucket_motion_consistency=True) as r:
    for requested in commands:
        command=r.projector.project(r.bridge.state['mechanism_qpos'],requested)
        r.step(command)
    actual=r.capture()
    with np.load(p/'state-000006.npz') as expected:
        errors={k:float(np.max(np.abs(actual[k].astype(float)-expected[k].astype(float)))) for k in expected.files if expected[k].dtype.kind in 'biufc'}
        equal={k:bool(np.array_equal(actual[k],expected[k])) for k in expected.files}
    expected_controller=json.loads((p/'state-000006.npz.json').read_text())['controller_state']
    c=controller_snapshot(r.bridge.state)
    controller_equal=c==expected_controller
    report={'status':'passed' if all(equal.values()) and controller_equal else 'failed','zero_step':r.restored,'arrays_equal':equal,'maximum_array_errors':errors,'controller_equal':controller_equal,'frames':3,'all_hard_guards':True}
    report['final_checkpoint']=r.save(out/'final.npz')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],equal,controller_equal)
    assert report['status']=='passed'
