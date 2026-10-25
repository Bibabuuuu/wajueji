"""CPU command checks on every prevalidated reference pose; no physical rollout."""
import json,time
import numpy as np
from runtime import ROOT,W,SOURCE,TransportRuntime,read,sha
from controller import TransportController,LIMITS
from layered_47_cycle_geometry import TransportClearance,stationary_terrain

def main():
    started=time.monotonic();r=TransportRuntime(SOURCE,ROOT/'runs/contact-carry-transport-20260924-v1/command-check')
    terrain,_=stationary_terrain(r);checker=TransportClearance(r,terrain)
    previous=np.array(read(SOURCE.parent/'control-state.json')['controller']['previous_command']['array'])
    control=TransportController(r,checker,previous);plan=read(ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/report.json')
    paths=np.load(ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/plan.npz')
    report=dict(status='checking',physics_steps_added=0,scope='Every nominal reference pose, not closed-loop physical or kinematic rollout',count=0)
    maximum=0.;stage='INITIALIZE';index=0
    try:
        for segment in plan['stages']:
            stage=segment['name'];qs=paths[stage+'_qpos'];vs=paths[stage+'_velocity']
            for index in range(len(vs)+30+(180 if stage=='DUMP' else 0)):
                q=r.arrays['mechanism_qpos'] if report['count']==0 else qs[min(index,len(qs)-1)]
                desired=qs[min(index+1,len(qs)-1)];velocity=vs[index] if index<len(vs) else np.zeros(4)
                target=read(str(SOURCE)+'.json')['controller_state']['controller_target'] if report['count']==0 else q[r.chain.axes]
                before=control.previous.copy();command,_=control.command(stage,q,target,desired,velocity)
                maximum=max(maximum,float(np.max(abs((command-before)/LIMITS))));report['count']+=1
            print('CHECKED',stage,report['count'],flush=True)
        report.update(status='reference_commands_passed',max_normalized_command_step=maximum)
    except BaseException as error:report.update(status='reference_commands_failed',failure=repr(error),stage=stage,stage_step=index);raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,inputs={str(p):sha(p) for p in (W/'controller.py',W/'plan.py',W/'runtime.py',ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/plan.npz')})
        (W/'control-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
