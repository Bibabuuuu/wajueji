"""Zero physical steps: exact restore, static route, all nominal commands."""
import copy,json,time
import numpy as np
from runtime import ROOT,W,RUN,SOURCE,SOURCE_SHA,TransportRuntime,read,sha
from plan import build_plan
from controller import TransportController,LIMITS

def main():
    started=time.monotonic();out=RUN/'preflight';out.mkdir(parents=True,exist_ok=False)
    report=dict(status='checking',physical_steps_added=0,source_sha256=SOURCE_SHA,reference_commands=0)
    try:
        with TransportRuntime(SOURCE,out/'physics') as r:
            exact={k:bool(np.array_equal(v,r.arrays[k])) for k,v in r.capture().items()}
            report['zero_arrays_exact']=exact;report['controller_errors']=r.restored['cycle_errors']
            if not all(exact.values()) or max(r.restored['cycle_errors'].values())>1e-12:raise ValueError('inexact_restore')
            native=copy.deepcopy(r.bridge.state);pending=copy.deepcopy(r.sweep.diagnostic_state())
            report['zero_checkpoint']=r.save(out/'zero.npz')
            # Initializing and saving cannot advance simulation or alter material history.
            segments,checker,plan=build_plan(r,RUN/'plan');report['plan']=plan
            previous=np.array(r.metadata['highlevel_state']['carry']['previous_command'])
            last=json.loads((SOURCE.parent/'actions.ndjson').read_text().splitlines()[-1])
            if not np.array_equal(previous,last['command']):raise ValueError('handoff_command_history_changed')
            control=TransportController(r,checker,previous);maximum=0.
            for segment in segments:
                qs=segment['path']['qpos'];vs=segment['path']['velocity'];stage=segment['name']
                for index in range(len(vs)+30+(180 if stage=='DUMP' else 0)):
                    q=r.arrays['mechanism_qpos'] if report['reference_commands']==0 else qs[min(index,len(qs)-1)]
                    desired=qs[min(index+1,len(qs)-1)];velocity=vs[index] if index<len(vs) else np.zeros(4)
                    target=r.metadata['controller_state']['controller_target'] if report['reference_commands']==0 else q[r.chain.axes]+control.tracking.bias
                    before=control.previous.copy();command,_=control.command(stage,q,target,desired,velocity)
                    if report['reference_commands']==0:report['first_command']=command.tolist()
                    maximum=max(maximum,float(np.max(abs((command-before)/LIMITS))))
                    report['reference_commands']+=1
                print('REFERENCE_CHECK',stage,report['reference_commands'],flush=True)
            if maximum>2/60+1e-9:raise ValueError('reference_slew_changed')
            if native!=r.bridge.state or pending!=r.sweep.diagnostic_state() or not all(np.array_equal(v,r.capture()[k]) for k,v in r.arrays.items()):raise ValueError('preflight_mutated_live_state')
            report.update(status='fixed_route_preflight_passed',maximum_normalized_command_step=maximum,
                plan_sha256=sha(RUN/'plan/plan.npz'),source_unchanged=sha(SOURCE)==SOURCE_SHA,
                scope='Exact physical restore and nominal-reference CPU checks; no closed-loop physical continuation yet')
    except BaseException as error:report.update(status='failed',failure=repr(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        (W/'preflight-review.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
