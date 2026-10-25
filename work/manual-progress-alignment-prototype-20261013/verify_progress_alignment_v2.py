"""User-authorized 1200-second physical admission only; never imports SAC."""
import json
from pathlib import Path
import sys
import time
ROOT=Path('/home/qunwang/wajueji')
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
from manual_residual_sac_env import ManualResidualEnv, Budget, BudgetExpired, sha, write, OLD


def main():
    old_out=ROOT/'runs/manual-progress-alignment-20261013-v1'
    out=ROOT/'runs/manual-progress-alignment-20261013-v2';out.mkdir(exist_ok=False)
    (out/'cpu-review.json').write_bytes((old_out/'cpu-review.json').read_bytes())
    review=json.loads((out/'cpu-review.json').read_text())
    assert review['status']=='passed'
    for name,expected in review['code'].items():
        assert sha(ROOT/'src/v4'/name)==expected
    assert not (out/'physical-review.json').exists()
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for name in review['code']:
        (snapshot/name).write_bytes((ROOT/'src/v4'/name).read_bytes())
    (snapshot/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    budget=Budget(old_out/'budget.json',seconds=1200,purpose='alignment_validation_only')
    manifest=dict(status='running',budget_seconds=1200,scope='alignment only, no training',
                  source_code=review['code'],episodes=[],shared_budget=str(old_out/'budget.json'),
                  previous_failed_verifier=str(old_out/'physical-review.json'),
                  correction='compare execution with already qualified projected commands; raw input remains feedforward')
    write(out/'physical-review.json',manifest)
    projected=[json.loads(line)['applied_si'] for line in (ROOT/'runs/manual-residual-sac-20261013/retiming-v2-gate/baseline/episode-0001/actions.ndjson').read_text().splitlines()]
    first=np.load(OLD/'manual-prefix-1800/display/particles.npy',mmap_mode='r')
    tail=np.load(OLD/'manual-prefix-continuation-1620/display/particles.npy',mmap_mode='r')
    env=None
    try:
        for label in ['zero_full','loaded_slow','loaded_opposed_fast']:
            if label!='zero_full' and budget.remaining()<280:
                manifest['not_run']=label;break
            env=ManualResidualEnv(out/label,budget,max_frames=1560 if label=='zero_full' else 1020)
            env.reset()
            counts=dict(particle_exact_frames=0,compact_exact_frames=0,
                        projected_command_exact_frames=0,loaded_force_frames=0,
                        maximum_loaded_force_n=0.)
            original_step=env.runtime.step
            def checked_step(command):
                frame=env.runtime.frames+1
                if label=='zero_full' and frame<=1170 or label!='zero_full' and frame<=900:
                    assert np.array_equal(command,projected[frame-1]),('projected_execution_changed',frame)
                    counts['projected_command_exact_frames']+=1
                row=original_step(command)
                if label=='zero_full' and frame<=1170 or label!='zero_full' and frame<=900:
                    expected=first[frame] if frame<=180 else tail[frame-180]
                    assert np.array_equal(env.runtime.fixture.state.particle_q.numpy(),expected),('zero_particles_changed',frame)
                    counts['particle_exact_frames']+=1
                    assert row['after_state']==env.references[frame],('zero_compact_changed',frame)
                    counts['compact_exact_frames']+=1
                if label!='zero_full' and frame>900:
                    norm=float(np.linalg.norm(env.runtime.captured['wrenches']['bucket']['force']))
                    counts['maximum_loaded_force_n']=max(counts['maximum_loaded_force_n'],norm)
                    counts['loaded_force_frames']+=norm>1e-6
                return row
            env.runtime.step=checked_step
            while True:
                action=np.zeros(5)
                if label!='zero_full' and env.runtime.frames>=900:
                    action[4]=-1 if label=='loaded_slow' else 1
                    if label=='loaded_opposed_fast':
                        action[:4]=-np.sign(env.commands[min(1169,int(env.cursor))])
                _,_,done,truncated,info=env.step(action)
                if done or truncated:
                    result=info['episode_summary'];break
            result['label']=label;result['checks']=counts
            env.close();env=None
            if label=='zero_full':
                assert result['status']=='completed' and result['success']
                assert result['retained_fill_fraction']==1.4538132234025687
                assert counts['particle_exact_frames']==1170 and counts['compact_exact_frames']==1170
            else:
                assert result['status']=='timeout' and result['physical_frames']==1020
                assert counts['loaded_force_frames']>0
            assert result['maximum_reference_lead_frames']<=6+1e-9
            manifest['episodes'].append(result)
            manifest['remaining_seconds']=budget.remaining()
            write(out/'physical-review.json',manifest)
            print('ALIGNMENT_GATE_EPISODE',label,result['status'],result['cursor'],
                  result['alignment_wait_frames'],counts,flush=True)
        manifest['status']='passed' if len(manifest['episodes'])==3 else 'incomplete_budget'
    except BaseException as error:
        manifest.update(status='budget_exhausted' if isinstance(error,BudgetExpired) else 'failed',
                        error=type(error).__name__+':'+str(error))
        raise
    finally:
        if env is not None:
            try:env.close()
            except BaseException as error:manifest['cleanup_error']=str(error)
        manifest['remaining_seconds']=budget.remaining()
        manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(out/'physical-review.json',manifest)
        print('ALIGNMENT_GATE_RESULT',manifest['status'],flush=True)


if __name__=='__main__':main()
