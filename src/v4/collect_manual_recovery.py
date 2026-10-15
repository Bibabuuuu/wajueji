"""Bounded recovery data collection. Frozen inference only; no neural training."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
from manual_residual_sac_env import ManualResidualEnv,Budget,BudgetExpired,sha,write,OBS_SIZE


def recovery_action(mode,error,holding=False):
    action=np.zeros(5,dtype=np.float32)
    if not holding and mode.startswith('wait'):
        error=np.asarray(error,dtype=float)
        if error.shape!=(4,) or not np.isfinite(error).all():raise ValueError('invalid_recovery_error')
        action[:4]=np.sign(error)
    return action


def save_transitions(env,rows,path):
    if not env.transitions:return dict(count=0,complete=False)
    before,after,actions,rewards,dones,infos=zip(*env.transitions)
    arrays=dict(observations=np.asarray(before),next_observations=np.asarray(after),
                actions=np.asarray(actions),rewards=np.asarray(rewards,dtype=np.float64),
                raw_rewards=np.asarray([x['raw_reward'] for x in infos]),dones=np.asarray(dones,dtype=bool),
                terminated=np.asarray([x['terminated'] for x in rows],dtype=bool),
                truncated=np.asarray([x['truncated'] for x in rows],dtype=bool),
                frames_before=np.asarray([x['frame_before'] for x in rows]),
                frames_after=np.asarray([x['frame_after'] for x in rows]),
                policy_source=np.asarray([x['policy_source'] for x in rows]))
    # Preserve observations even if a following integrity check rejects them.
    np.savez_compressed(path,**arrays)
    n=len(before)
    assert len(rows)==n and arrays['observations'].shape==(n,OBS_SIZE)
    assert arrays['actions'].shape==(n,5) and np.all(abs(arrays['actions'])<=1)
    assert all(np.isfinite(arrays[k]).all() for k in ['observations','next_observations','actions','rewards','raw_rewards'])
    assert np.array_equal(arrays['next_observations'][:-1],arrays['observations'][1:])
    assert not arrays['dones'][:-1].any()
    assert np.array_equal(arrays['dones'],arrays['terminated']|arrays['truncated'])
    expected=arrays['raw_rewards']+np.where(arrays['dones'],0.,arrays['next_observations'][:,-2].astype(float))-arrays['observations'][:,-2].astype(float)
    assert np.allclose(arrays['rewards'],expected,rtol=0,atol=1e-12)
    complete=bool(arrays['dones'][-1])
    total_error=float(abs(arrays['rewards'].sum()-arrays['raw_rewards'].sum()))
    if complete:assert total_error<1e-10
    return dict(count=n,complete=complete,terminal_count=int(arrays['dones'].sum()),
                total_reward_difference=total_error,sha256=sha(path),path=str(path),
                initial_potential=float(arrays['observations'][0,-2]),
                recovery_steps=int(np.sum(arrays['policy_source']=='fixed_correction')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('immutable_collection_output_exists')
    out.mkdir(parents=True)
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    previous=ROOT/'runs/manual-residual-sac-aligned-20261013/batch-v2'
    cases=[('early43',previous/'sac-round-01.zip',previous/'evaluation-01/episode-0001',120),
           ('early52',previous/'replay-update-evaluation/sac-replay-updated.zip',previous/'replay-update-evaluation/evaluation/episode-0001',120),
           ('loaded970',None,ROOT/'runs/manual-progress-alignment-20261013-v3/loaded_correction/episode-0001',1020)]
    admission=json.loads((ROOT/'runs/manual-residual-sac-aligned-20261013/reward-transfer-v2/training-admission.json').read_text())
    for name,expected in admission['code'].items():
        assert sha(ROOT/'src/v4'/name)==expected,('admission_source_changed',name)
    budget=Budget(out/'budget.json',seconds=1200,purpose='recovery_collection_validation_only')
    manifest=dict(status='prepared',budget_seconds=1200,neural_training_authorized=False,
                  neural_gradient_updates=0,episodes=[],code={name:sha(ROOT/'src/v4'/name) for name in
                  ['collect_manual_recovery.py','manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']})
    snapshot=out/'source-snapshot';snapshot.mkdir()
    for name in manifest['code']:(snapshot/name).write_bytes((ROOT/'src/v4'/name).read_bytes())
    write(out/'review.json',manifest);env=None
    try:
        for label,checkpoint,old_episode,prefix_frames in cases:
            if budget.remaining()<330:
                manifest['not_started_insufficient_budget']=label;break
            model=SAC.load(checkpoint,device='cpu') if checkpoint is not None else None
            before_updates=None if model is None else int(model._n_updates)
            expected=[]
            with (old_episode/'physics/frames.ndjson').open() as log:
                for _ in range(prefix_frames):expected.append(json.loads(next(log)))
            entry=dict(label=label,status='initializing',prefix_frames=prefix_frames,
                       prefix_commands_exact=0,prefix_compact_exact=0,
                       frozen_policy=None if checkpoint is None else dict(path=str(checkpoint),sha256=sha(checkpoint)))
            rows=[];env=ManualResidualEnv(out/label,budget,collect=True);obs,_=env.reset()
            original_step=env.runtime.step
            def checked_step(command):
                frame=env.runtime.frames+1
                if frame<=prefix_frames:
                    assert np.array_equal(command,expected[frame-1]['applied_command_si']),('prefix_command_changed',label,frame)
                    entry['prefix_commands_exact']+=1
                row=original_step(command)
                if frame<=prefix_frames:
                    assert row['after_state']==expected[frame-1]['after_state'],('prefix_compact_changed',label,frame)
                    entry['prefix_compact_exact']+=1
                return row
            env.runtime.step=checked_step
            start=None
            try:
                while True:
                    frame=env.runtime.frames;alignment=env.report['last_alignment']
                    if frame==prefix_frames:
                        assert alignment['mode'].startswith('wait'),('expected_stall_not_reproduced',label)
                        start=dict(frame=frame,cursor=env.cursor,actual_progress=env.alignment.measured,
                                   target_error_si=alignment['wait_target_error_si'])
                        entry['recovery_start']=start
                    if model is not None and frame<prefix_frames:
                        action=model.predict(obs,deterministic=True)[0];source='frozen_prefix'
                    elif label=='loaded970' and 900<=frame<1020:
                        action=np.zeros(5,dtype=np.float32);action[4]=1
                        action[:4]=-np.sign(env.commands[min(1169,int(env.cursor))]);source='bounded_perturbation'
                    else:
                        action=recovery_action(alignment['mode'],alignment['wait_target_error_si'],env.hold is not None)
                        source='fixed_correction' if np.any(action[:4]) else 'manual_follow'
                    obs,reward,done,truncated,info=env.step(action)
                    record=dict(frame_before=frame,frame_after=env.runtime.frames,policy_source=source,
                                action=np.asarray(action).tolist(),terminated=done,truncated=truncated,
                                reward=reward,raw_reward=info['raw_reward'])
                    rows.append(record)
                    with (env.directory/'policy.ndjson').open('a') as log:log.write(json.dumps(record)+'\n')
                    if done or truncated:
                        result=info['episode_summary'];entry['result']=result;break
                assert start is not None
                entry['recovered_progress']=bool(env.alignment.measured>start['actual_progress']+10)
                entry['recovery_demonstration_qualified']=bool(entry['recovered_progress'] and result['success'] and result['source_unchanged'])
                if model is not None:assert int(model._n_updates)==before_updates
                entry['status']='completed'
            finally:
                entry['dataset']=save_transitions(env,rows,out/label/'transitions.npz')
                env.close();env=None
                manifest['episodes'].append(entry);manifest['remaining_seconds']=budget.remaining()
                write(out/'review.json',manifest)
            print('RECOVERY_CASE',label,entry['result']['status'],entry['result']['cursor'],
                  entry['result'].get('retained_fill_fraction'),entry['dataset']['count'],flush=True)
        manifest['status']='completed' if len(manifest['episodes'])==3 else 'incomplete_budget'
        manifest['all_recovery_demonstrations_qualified']=len(manifest['episodes'])==3 and all(x.get('recovery_demonstration_qualified',False) for x in manifest['episodes'])
    except BaseException as error:
        manifest.update(status='budget_exhausted' if isinstance(error,BudgetExpired) else 'failed',
                        error=type(error).__name__+':'+str(error))
        raise
    finally:
        if env is not None:
            try:env.close()
            except BaseException as error:manifest['cleanup_error']=type(error).__name__+':'+str(error)
        manifest['remaining_seconds']=budget.remaining()
        manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(out/'review.json',manifest)
        print('RECOVERY_COLLECTION_RESULT',manifest['status'],flush=True)


if __name__=='__main__':main()
