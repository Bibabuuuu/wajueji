"""Bounded, same-command failure replay; output is diagnostic, not resumable."""
import argparse,hashlib,json,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_bridge import controller_snapshot
from passive_jp_candidate import installed_passive_jp

ROOT=Path('/home/qunwang/wajueji')
ORIGINAL=ROOT/'runs/layered-47-cycle-v1/third-scoop-finish-20260924-094135-1368892/transport-after-retreat'
SOURCE=ORIGINAL/'align.npz'
EXPECTED='1f5796ab9d13c5320f2ffb19c3bd46f2d143486284354328d060486621df9a2e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--candidate',action='store_true');a=p.parse_args()
assert sha(SOURCE)==EXPECTED
rows=[json.loads(s) for s in (ORIGINAL/'frames.ndjson').read_text().splitlines()]
commands=[v['applied_command_si'] for v in rows if v['frame']>1484]
assert len(commands)==813 and rows[-1]['frame']==2297
commands=commands+[[0.,0.,0.,0.]]*180 if a.candidate else commands[:24]
r=CycleRuntime(SOURCE,a.output);out=Path(a.output);started=time.monotonic()
report=dict(schema='passive_jp_same_prefix_replay_v1',status='running',candidate=a.candidate,source_sha256=EXPECTED,maximum_frames=len(commands),training_eligible=False,resumable=False)
def snapshot(path,**_):
    arrays=r.capture();np.savez(path,**arrays)
    meta=dict(schema='passive_jp_diagnostic_not_cycle_checkpoint_v1',resumable=False,candidate=a.candidate,source=str(SOURCE),source_sha256=EXPECTED,frames=r.frames,controller_state=controller_snapshot(r.bridge.state),moving_bucket_sweep_state=r.sweep.diagnostic_state(),runtime_bindings=r.bindings,candidate_sha256=sha(Path(__file__).with_name('passive_jp_candidate.py')))
    Path(str(path)+'.json').write_text(json.dumps(meta,indent=2)+'\n')
    return dict(path=str(path),sha256=sha(path))
r.save=snapshot
try:
    with r:
        context=installed_passive_jp(r.fixture.solver,r.fixture.state) if a.candidate else nullcontext({})
        with context as counter:
            for command in commands:
                if time.monotonic()-started>420:raise RuntimeError('bounded_replay_wall_budget')
                r.step(command)
                if r.frames==24:report['prefix_snapshot']=snapshot(out/'prefix-24.npz')
                if r.frames==813:report['failure_time_snapshot']=snapshot(out/'at-old-failure.npz')
                if r.frames%120==0:print('REPLAY',r.frames,flush=True)
            report['final_snapshot']=snapshot(out/'final.npz')
            report['policy']=dict(counter)
        report.update(status='candidate_window_passed' if a.candidate else 'baseline_prefix_passed',frames=r.frames)
except BaseException as error:
    report.update(status='failed',failure=repr(error),frames=r.frames)
    raise
finally:
    report.update(source_unchanged=sha(SOURCE)==EXPECTED,wall_seconds=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
