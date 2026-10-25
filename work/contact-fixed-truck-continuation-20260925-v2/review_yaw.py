"""CPU reproduction of the failed fourth command and coordinate equivariance."""
import json
import numpy as np
from runtime import ROOT,W,SOURCE,TransportRuntime,read,sha
from controller import TransportController,LIMITS,prior
from observation import stationary_terrain
from layered_47_cycle_geometry import TransportClearance
import yaw_preview

def main():
    r=TransportRuntime(SOURCE,ROOT/'runs/contact-fixed-truck-continuation-20260925-v2/yaw-review')
    terrain,_=stationary_terrain(r);checker=TransportClearance(r,terrain)
    previous=np.array(read(SOURCE.parent/'control-state.json')['previous_command'])
    old=prior.TransportController(r,checker,previous);new=TransportController(r,checker,previous)
    run=ROOT/'runs/contact-fixed-truck-continuation-20260925-v1/physical'
    actions=[json.loads(s) for s in (run/'actions.ndjson').read_text().splitlines()]
    checks={};error=0.
    for a in actions:
        args=(a['stage'],np.array(a['before_state']['mechanism_qpos']),a['controller_target'],np.array(a['desired_qpos']),np.array(a['reference_velocity']))
        before=new.previous.copy();cmd,_=new.command(*args);old.command(*args)
        error=max(error,float(np.max(abs(cmd-a['command']))))
    checks['three_issued_commands_unchanged']=error<1e-12
    with np.load(run/'recovery.npz') as f:q=f['mechanism_qpos'].copy()
    target=read(str(run/'recovery.npz')+'.json')['controller_state']['controller_target']
    path=ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/plan.npz'
    with np.load(path) as f:args=('ALIGN',q,target,f['ALIGN_qpos'][4],f['ALIGN_velocity'][3])
    try:old.command(*args);old_error=None
    except ValueError as e:old_error=str(e)
    before=new.previous.copy();cmd,_=new.command(*args)
    checks['old_fourth_failure_reproduced']=old_error=='closure_unresolved'
    checks['new_fourth_command_passed']=bool(np.isfinite(cmd).all() and np.max(abs((cmd-before)/LIMITS))<=2/60+1e-9)
    residual=r.chain.residual(q);R=yaw_preview.rotation(r.chain,q)
    maximum=0.
    for delta in (-.1,-.01,.01,.1):
        trial=q.copy();trial[7]+=delta
        expected=(residual.reshape(-1,3)@R@yaw_preview.rotation(r.chain,trial).T).reshape(-1)
        maximum=max(maximum,float(np.max(abs(expected-r.chain.residual(trial)))))
    checks['closure_offsets_rotate_with_chassis']=maximum<1e-12
    r.chain.q=q.copy();qzero,_=yaw_preview.advance(r.chain,np.zeros(4),1/60,residual,R)
    checks['zero_command_exact']=np.array_equal(qzero,q)
    # Failure cannot consume integral state for an unissued command.
    bias=new.tracking.bias.copy();prev=new.previous.copy();first=new.first
    broken=q.copy();broken[9]=-10.
    try:new.command('ALIGN',broken,target,args[3],args[4]);rejected=False
    except ValueError:rejected=True
    checks['rejected_command_transactional']=bool(rejected and np.array_equal(new.tracking.bias,bias) and np.array_equal(new.previous,prev) and new.first==first)
    report=dict(passed=all(checks.values()),checks=checks,physics_steps_added=0,
        old_failure=old_error,command_error=error,fourth_command=cmd.tolist(),rotation_identity_error=maximum,
        inputs={str(p):sha(p) for p in [SOURCE,run/'recovery.npz',run/'actions.ndjson',path,W/'controller.py',W/'yaw_preview.py',Path(__file__)]})
    (W/'yaw-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']

from pathlib import Path
if __name__=='__main__':main()
