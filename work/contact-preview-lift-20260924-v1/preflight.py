"""Bounded zero-MPM restore and kinematic checks. No live rigid step."""
import json,time,copy
from pathlib import Path
import numpy as np
from continuation_runtime import ContinuationRuntime,SOURCE,W,ROOT,PRIOR,SOURCE_SHA,sha,load_candidate
from continuous_preview import preview
from control_history import construct,restore,OUT
from lift_feedback import LiftFeedback
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector

def main():
    started=time.monotonic();fb,_,_,_,_=construct();p=ContactCommandProjector(fb.chain);actions={e['frame']:e for e in [json.loads(x) for x in (PRIOR/'actions.ndjson').read_text().splitlines()]};checks=[]
    for f in (1124,1200,1376):
        e=actions[f];q=np.array(e['before_state']['mechanism_qpos']);start=fb.body_heights(q);reference=fb.chain.residual(q);trials=[]
        for factor in (1.,.125,1e-6,1e-9,0.):
            fb.chain.q=q.copy();delta=0;err=0
            for _ in range(9):
                nq,_=preview(p,np.array(e['requested_command_si'])*factor,1/60,reference);delta=min(delta,float(min(fb.body_heights(nq)-start)));err=max(err,float(max(abs(fb.chain.residual(nq)-reference))))
            trials.append(dict(factor=factor,minimum_body_delta_m=delta,residual_change_m=err))
        if abs(trials[-2]['minimum_body_delta_m'])>1e-9 or max(x['residual_change_m'] for x in trials)>1.1e-8:raise ValueError('continuous_preview_limit_failed')
        # Original hard-limit/branch envelope is still active.
        fb.chain.q=q.copy()
        try:preview(p,np.array([.13,0,0,0]),1/60,reference)
        except ValueError as error:assert str(error)=='command_budget'
        else:raise AssertionError('speed_envelope_removed')
        checks.append(dict(frame=f,trials=trials,limit_continuous=True,original_speed_budget_enforced=True))
    with ContinuationRuntime(SOURCE,OUT/'zero',allow_training_reset=True) as r:
        live=r.capture();exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in live.items()}
        if not all(exact.values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('D153_restore_not_exact')
        feedback,phases,features,report,contract=construct(LiftFeedback,r.assembly,r.chain)
        history=restore(feedback,phases,features,OUT/'control-history.json')
        q=np.array(r.bridge.state['mechanism_qpos']);r.projector.chain.q=q.copy()
        requested,applied,detail=feedback.command(frame=2401,qpos=q,phase=phases.router.phase,bc_action=actions[2400]['action_normalized'],projector=r.projector)
        if not detail['lift_intent'] or detail['mode']!='bounded_feedback_lift' or detail['no_down_preview_minimum_delta_m']<-1e-6 or detail['preview_minimum_clearance_m']<.03-1e-6 or not any(applied):raise ValueError('D153_first_lift_command_not_admitted')
        after=r.capture()
        if not all(np.array_equal(live[k],after[k]) for k in live):raise ValueError('preflight_moved_physics')
        # Save under its own explicit candidate identity, ordinary loader must reject.
        saved=r.save(OUT/'zero.npz',stage='hold')
        from layered_47_cycle_checkpoint import load
        try:load(OUT/'zero.npz',allow_training_reset=True)
        except ValueError as error:assert str(error)=='cycle_checkpoint_schema'
        else:raise AssertionError('ordinary_loader_accepted_candidate')
        result=dict(status='D153_zero_restore_and_preview_passed',arrays_exact=exact,controller_errors=r.restored['cycle_errors'],continuous_preview_tests=checks,first_lift_request=requested.tolist(),first_lift_applied=applied.tolist(),first_lift_preview=detail,physics_steps_added=0,policy_history_path=str(OUT/'control-history.json'),policy_history_sha256=sha(OUT/'control-history.json'),source_sha256=sha(SOURCE),zero_checkpoint=saved,wall_seconds=time.monotonic()-started)
    (W/'preflight-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
