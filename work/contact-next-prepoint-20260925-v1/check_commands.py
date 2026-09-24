"""Pure CPU reference-command regression; no physical states are advanced."""
import json,time
import numpy as np
from runtime import ROOT,W,SOURCE,PrepointRuntime,read,sha
from controller import PrepointController,PrepointClearance,LIMITS
from observation import stationary_terrain

def main():
    started=time.monotonic();review=read(W/'preparation-review.json')
    if review['status']!='zero_and_geometry_passed':raise ValueError('preparation_not_passed')
    planpath=ROOT/'runs/contact-next-prepoint-20260925-v1/prepare/plan.npz'
    if sha(planpath)!=review['plan_sha256']:raise ValueError('plan_changed')
    paths=np.load(planpath)
    r=PrepointRuntime(SOURCE,ROOT/'runs/contact-next-prepoint-20260925-v1/commands')
    terrain,_=stationary_terrain(r);checker=PrepointClearance(r,terrain)
    previous=np.array(r.metadata['transport_controller_state']['previous_command'])
    controller=PrepointController(r,checker,previous)
    target=np.array(r.metadata['controller_state']['controller_target']);q=r.arrays['mechanism_qpos'].copy()
    max_slew=0.;n=0
    for desired,velocity in [*zip(paths['qpos'][1:],paths['velocity'],strict=True),*[(paths['qpos'][-1],np.zeros(4))]*30]:
        command,_=controller.command('PREPOINT',q,target,desired,velocity)
        if n==0 and not np.array_equal(command,previous):raise ValueError('first_command_changed')
        max_slew=max(max_slew,float(np.max(abs((command-previous)/LIMITS))))
        previous=command.copy();target+=command/60;q=desired.copy();n+=1
    report=dict(status='reference_commands_passed',frames=n,physics_steps_added=0,
        maximum_normalized_command_step=max_slew,first_command_exact=True,
        source_sha256=sha(SOURCE),plan_sha256=sha(planpath),wall_seconds=time.monotonic()-started)
    (W/'control-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
