"""Twenty-minute control validation only; no neural policy or replay transfer."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
from manual_residual_sac_env import ManualResidualEnv, Budget, BudgetExpired, sha, write, ACTION_HANDOFF_VERSION, environment_contract, require_environment_contract, TERMINAL_SCORE_VERSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collect',action='store_true')
    parser.add_argument('--collection-admission',type=Path)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.collect:
        if args.collection_admission is None:raise ValueError('collection_requires_versioned_admission')
        admission=json.loads(args.collection_admission.read_text())
        require_environment_contract(admission)
        if admission.get('collection_admission') is not True:raise ValueError('collection_not_qualified')
        for path,h in admission['source_evidence'].items():
            if sha(path)!=h:raise ValueError('collection_evidence_changed')
        for name,h in admission['runtime_bindings'].items():
            if sha(ROOT/'src/v4'/name)!=h:raise ValueError('collection_physics_changed')
    if args.check_only:
        print(json.dumps(dict(status='ready_for_collection' if args.collect else 'ready_for_validation',
              environment_contract=environment_contract(),physical_frames=0,neural_updates=0)));return
    if args.collect:from collect_manual_recovery import save_transitions
    out=args.output.resolve()
    if out.exists():raise ValueError('immutable_validation_output_exists')
    out.mkdir(parents=True)
    budget=Budget(out/'budget.json',seconds=1200,purpose='strength_recovery_collection_only' if args.collect else 'wait_strength_control_validation_only')
    old=json.loads((ROOT/'runs/manual-residual-sac-aligned-20261013/reward-transfer-v2/transfer.json').read_text())['baseline']
    for name,h in old['runtime_bindings'].items():assert sha(ROOT/'src/v4'/name)==h
    baseline_path=Path(old['path'])
    if not baseline_path.is_absolute():baseline_path=ROOT/baseline_path
    reference=[json.loads(s) for s in (baseline_path/'physics/frames.ndjson').read_text().splitlines()]
    assert len(reference)==1200
    review=dict(status='prepared',budget_seconds=1200,neural_updates=0,training_admission=False,
                collection_enabled=args.collect,environment_contract=environment_contract(),
                action_mode='error_strength_v1',action_handoff_version=ACTION_HANDOFF_VERSION,old_replay_reused=False,episodes=[],
                code={n:sha(ROOT/'src/v4'/n) for n in ['manual_residual_sac_env.py','manual_trajectory_alignment.py',
                      'manual_trajectory_residual.py','validate_manual_wait_strength.py']})
    if args.collect:
        review['code']['collect_manual_recovery.py']=sha(ROOT/'src/v4/collect_manual_recovery.py')
        review['collection_admission_receipt']=dict(path=str(args.collection_admission),sha256=sha(args.collection_admission))
    snap=out/'source-snapshot';snap.mkdir()
    for n in review['code']:(snap/n).write_bytes((ROOT/'src/v4'/n).read_bytes())
    write(out/'review.json',review);env=None;common_prefix=None
    try:
        for label,strength in [('manual_baseline',None),('loaded_stop',0.),('loaded_half',.5),('loaded_full',1.)]:
            if budget.remaining()<330:review['not_started_insufficient_budget']=label;break
            entry=dict(label=label,strength=strength,status='running',prefix_commands_exact=0,prefix_compact_exact=0)
            review['episodes'].append(entry);write(out/'review.json',review)
            env=ManualResidualEnv(out/label,budget,waiting_action_mode='error_strength_v1',collect=args.collect)
            policy_rows=[]
            obs,_=env.reset();original_step=env.runtime.step;prefix_rows=[]
            def checked_step(command):
                frame=env.runtime.frames+1
                expected=reference[frame-1] if strength is None and frame<=1200 else (
                    common_prefix[frame-1] if strength is not None and common_prefix is not None and frame<=1020 else None)
                if expected is not None:
                    assert np.array_equal(command,expected['applied_command_si']),('prefix_command_changed',label,frame)
                    entry['prefix_commands_exact']+=1
                row=original_step(command)
                if expected is not None:
                    assert row['after_state']==expected['after_state'],('prefix_compact_changed',label,frame)
                    entry['prefix_compact_exact']+=1
                if strength is not None and frame<=1020:prefix_rows.append(row)
                return row
            env.runtime.step=checked_step;start=None
            while True:
                frame=env.runtime.frames;mode=env.report['last_alignment']['mode']
                if strength is not None and frame==1020:
                    assert mode.startswith('wait'),('loaded_stall_not_reached',mode)
                    start=dict(frame=frame,actual_progress=env.alignment.measured,cursor=env.cursor,
                               target_error_si=env.report['last_alignment']['wait_target_error_si'])
                    entry['recovery_start']=start
                    if common_prefix is None:common_prefix=prefix_rows
                action=np.zeros(5);source='manual_follow'
                if strength is not None and 900<=frame<1020:
                    action[4]=1
                    action[:4]=-1 if mode.startswith('wait') else -np.sign(env.commands[min(1169,int(env.cursor))])
                    source='bounded_perturbation_or_wait_stop'
                elif strength is not None and frame>=1020 and mode.startswith('wait') and env.hold is None:
                    action[:4]=2*strength-1;source='fixed_wait_strength'
                obs,reward,done,truncated,info=env.step(action)
                with (env.directory/'policy.ndjson').open('a') as f:
                    record=dict(frame_before=frame,frame_after=env.runtime.frames,action=action.tolist(),
                                policy_source=source,mode_before=mode,terminated=done,truncated=truncated,
                                reward=reward,raw_reward=info['raw_reward'])
                    policy_rows.append(record);f.write(json.dumps(record)+'\n')
                if done or truncated:
                    entry['result']=info['episode_summary'];break
            if strength is None:
                assert entry['result']['success'] and env.runtime.frames==1200
                assert entry['result']['retained_fill_fraction']==old['retained_fill_fraction']
                keys=json.loads((ROOT/'runs/manual-wait-strength-validation-20261013-v1/manual_baseline/episode-0001/baseline-endpoint-array-check.json').read_text())['checks']
                with np.load(baseline_path/'endpoint.npz',allow_pickle=False) as previous, np.load(Path(entry['result']['checkpoint']['path']),allow_pickle=False) as current:
                    arrays={key:bool(np.array_equal(previous[key],current[key])) for key in keys}
                write(env.directory/'baseline-endpoint-array-check.json',dict(status='passed' if all(arrays.values()) else 'failed',checks=arrays))
                assert len(arrays)==24 and all(arrays.values()),'baseline_endpoint_arrays_changed'
                entry['manual_endpoint_arrays_exact']=arrays
                entry['manual_baseline_exact']=entry['prefix_commands_exact']==entry['prefix_compact_exact']==1200
            else:
                assert start is not None
                entry['recovered_progress']=bool(env.alignment.measured>start['actual_progress']+10)
                entry['recovery_qualified']=entry['recovered_progress'] and entry['result']['success']
            if args.collect:
                assert all(t[-1]['terminal_score_version']==TERMINAL_SCORE_VERSION for t in env.transitions)
                entry['dataset']=save_transitions(env,policy_rows,out/label/'transitions.npz')
                entry['dataset'].pop('recovery_steps',None)
                entry['dataset']['fixed_wait_action_steps']=sum(r['policy_source']=='fixed_wait_strength' for r in policy_rows)
                entry['dataset']['environment_contract']=environment_contract()
                if not entry['dataset']['complete']:raise ValueError('collection_episode_incomplete')
            entry['status']='completed';env.close();env=None
            review['remaining_seconds']=budget.remaining();write(out/'review.json',review)
            print('STRENGTH_CASE',label,entry['result']['status'],entry['result']['physical_frames'],
                  entry['result']['cursor'],entry['result'].get('retained_fill_fraction'),flush=True)
        review['status']='completed' if len(review['episodes'])==4 else 'incomplete_budget'
    except BaseException as error:
        review.update(status='budget_exhausted' if isinstance(error,BudgetExpired) else 'failed',error=type(error).__name__+':'+str(error))
        raise
    finally:
        if env is not None:
            try:
                if args.collect:
                    entry['dataset']=save_transitions(env,policy_rows,out/label/'transitions-incomplete.npz')
                    entry['dataset']['environment_contract']=environment_contract()
            except Exception as error:review['dataset_save_error']=str(error)
            finally:env.close()
        review['remaining_seconds']=budget.remaining();review['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(out/'review.json',review);print('WAIT_STRENGTH_RESULT',review['status'],flush=True)


if __name__=='__main__':main()
