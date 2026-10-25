"""Bounded recovery check in the SAME authorized validation budget."""
from pathlib import Path
import json
import sys
ROOT=Path('/home/qunwang/wajueji')
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
from manual_residual_sac_env import ManualResidualEnv,Budget,write,sha
from manual_trajectory_residual import compare_arrays


def main():
    out=ROOT/'runs/manual-progress-alignment-20261013-v2'
    prior=out/'loaded_opposed_fast/episode-0001'
    budget=Budget(ROOT/'runs/manual-progress-alignment-20261013-v1/budget.json',
                  seconds=1200,purpose='alignment_validation_only')
    budget.check()
    audit=json.loads((out/'independent-audit.json').read_text())
    assert audit['status']=='passed' and budget.remaining()>210
    output=out/'wait-recovery-review.json'
    assert not output.exists()
    report=dict(status='running',no_training=True,additional_frames=60,
                prefix_replayed_reason='bounded timeout saved as failure; loader prohibits continuation; preserve rule',
                shared_budget=str(budget.path),source_checkpoint_relabelled=False)
    write(output,report)
    env=ManualResidualEnv(out/'wait_recovery',budget,max_frames=1080)
    old_rows=[json.loads(x) for x in (prior/'physics/frames.ndjson').read_text().splitlines()]
    old_commands=[json.loads(x)['applied_si'] for x in (prior/'actions.ndjson').read_text().splitlines()]
    try:
        env.reset()
        original=env.runtime.step
        def checked_step(command):
            frame=env.runtime.frames+1
            if frame<=1020:assert np.array_equal(command,old_commands[frame-1])
            row=original(command)
            if frame<=1020:assert row['after_state']==old_rows[frame-1]['after_state']
            return row
        env.runtime.step=checked_step
        while True:
            action=np.zeros(5)
            if 900<=env.runtime.frames<1020:
                action[:4]=-np.sign(env.commands[min(1169,int(env.cursor))]);action[4]=1
            if env.runtime.frames==1020:
                with np.load(prior/'endpoint.npz',allow_pickle=False) as previous:
                    exact,errors=compare_arrays(env.runtime.capture(),previous)
                assert all(exact.values())
                report.update(prefix_full_arrays_exact=exact,initial_cursor=env.cursor,
                              initial_actual_progress=env.alignment.measured)
                write(output,report)
            _,_,done,truncated,info=env.step(action)
            if done or truncated:break
        result=info['episode_summary']
        report.update(status='completed',result=result,
            final_actual_progress=env.alignment.measured,final_cursor=env.cursor,
            progressed_after_residual_removed=env.alignment.measured>report['initial_actual_progress'],
            left_wait_state=result['last_alignment']['mode']!='wait',
            remaining_seconds=budget.remaining())
        report['training_admission']=bool(report['progressed_after_residual_removed'] and report['left_wait_state'])
        write(output,report)
        print('WAIT_RECOVERY_RESULT',json.dumps({k:report[k] for k in [
            'status','initial_cursor','initial_actual_progress','final_cursor','final_actual_progress',
            'progressed_after_residual_removed','left_wait_state','training_admission','remaining_seconds']}),flush=True)
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+':'+str(error),
                      remaining_seconds=budget.remaining())
        write(output,report);raise
    finally:env.close()


if __name__=='__main__':main()
