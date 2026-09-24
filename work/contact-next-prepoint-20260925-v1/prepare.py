"""Zero-step restore and existing next-cut planner; never executes a command."""
import copy,json,sys,time
import numpy as np
from runtime import ROOT,W,SOURCE,PrepointRuntime,read,sha
sys.path.append(str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2'))
from observation import stationary_terrain
import layered_47_cycle_selection as selection
from run_layered_47_sparse_manual_contact_episode import _clean

def main():
    out=ROOT/'runs/contact-next-prepoint-20260925-v1/prepare'
    report=dict(status='preparing',physics_steps_added=0,source_sha256=sha(SOURCE));started=time.monotonic()
    try:
        with PrepointRuntime(SOURCE,out/'zero',freeze_passive_jp=True) as r:
            before=r.capture();native=copy.deepcopy(r.bridge.state);pending=copy.deepcopy(r.sweep.diagnostic_state())
            exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in before.items()}
            if not all(exact.values()) or max(r.restored['cycle_errors'].values())>1e-12:
                raise ValueError('next_prepoint_zero_restore')
            saved=r.save(out/'zero.npz',stage='returned')
            if native!=r.bridge.state or pending!=r.sweep.diagnostic_state():raise ValueError('zero_save_mutated_state')
            if not all(np.array_equal(v,r.capture()[k]) for k,v in before.items()):raise ValueError('zero_save_mutated_arrays')
            report.update(zero_arrays_exact=exact,native_exact=True,pending_exact=True,zero_checkpoint=saved)
        # CPU planning uses precisely the restored source, with the already
        # checked ground-supported-rock observer. It cannot advance physics.
        r=PrepointRuntime(SOURCE,out/'planner',freeze_passive_jp=True)
        original=selection.stationary_terrain
        try:
            selection.stationary_terrain=stationary_terrain
            plan=selection.plan_next_prepoint(r)
        finally:selection.stationary_terrain=original
        path=plan.pop('path');terrain=plan.pop('terrain')
        np.savez(out/'plan.npz',qpos=path['qpos'],velocity=path['velocity'])
        if path['total_frames']+30>4830:raise ValueError('next_prepoint_frame_budget')
        report.update(status='zero_and_geometry_passed',**plan,
            planned_frames=path['total_frames'],segments=path['segments'],
            prepoint_tip_world_m=path['prepoint_tip_world_m'],plan_sha256=sha(out/'plan.npz'))
    except BaseException as error:report.update(status='preparation_failed',failure=repr(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        (W/'preparation-review.json').write_text(json.dumps(_clean(report),indent=2,allow_nan=False)+'\n')
        print(json.dumps(_clean(report),indent=2,allow_nan=False),flush=True)
if __name__=='__main__':main()
