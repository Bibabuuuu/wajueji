"""Zero-step native/MPM restore with existing truck and passive-Jp fixes."""
import copy,json,time
import numpy as np
from runtime import ROOT,W,SOURCE,SOURCE_SHA,TransportRuntime,sha

def main():
    out=ROOT/'runs/contact-carry-transport-20260924-v1/zero'
    started=time.monotonic();report=dict(status='preflight_started',physics_steps_added=0,source_sha256=SOURCE_SHA)
    try:
        with TransportRuntime(SOURCE,out,freeze_passive_jp=True) as r:
            initial=r.capture();native=copy.deepcopy(r.bridge.state);pending=copy.deepcopy(r.sweep.diagnostic_state())
            exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in initial.items()}
            assert all(exact.values()) and max(r.restored['cycle_errors'].values())<=1e-12
            assert r.truck_enabled and r.passive_jp_policy['updates']==0
            saved=r.save(out/'zero.npz',stage='hold')
            assert all(np.array_equal(v,r.capture()[k]) for k,v in initial.items())
            assert native==r.bridge.state and pending==r.sweep.diagnostic_state()
            report.update(status='transport_zero_step_passed',arrays_exact=exact,controller_errors=r.restored['cycle_errors'],
                truck_enabled=True,passive_jp_policy=r.metadata['material_history_policy'],
                source_jp_unchanged=True,pending_unchanged=True,native_unchanged=True,saved=saved)
    except BaseException as error:report.update(status='preflight_failed',failure=repr(error));raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA)
        (W/'preflight-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
