"""Offline command recomputation and physical-evidence audit; never steps physics."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from manual_residual_sac_env import command_at, INPUT, SOURCE, INPUT_SHA, SOURCE_SHA, LIMITS_SI, control_mode, ACTION_HANDOFF_VERSION, TERMINAL_SCORE_VERSION, require_environment_contract

ROOT=Path('/home/qunwang/wajueji')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root):
    review=json.loads((root/'review.json').read_text());budget=json.loads((root/'budget.json').read_text())
    commands=np.asarray([json.loads(s)['applied_command_si'] for s in INPUT.read_text().splitlines()][:1170])
    checks=dict(source_unchanged=digest(INPUT)==INPUT_SHA and digest(SOURCE)==SOURCE_SHA,
                new_budget_1200=budget['seconds']==1200 and budget['purpose']==('strength_recovery_collection_only' if review.get('collection_enabled') else 'wait_strength_control_validation_only'),
                within_budget=review['remaining_seconds']>=0,
                source_snapshot=all(digest(root/'source-snapshot'/n)==h for n,h in review['code'].items()),
                source_current=all(digest(ROOT/'src/v4'/n)==h for n,h in review['code'].items()),
                neural_updates_zero=review['neural_updates']==0,
                old_replay_not_used=review['old_replay_reused'] is False)
    tree=ast.parse((root/'source-snapshot/validate_manual_wait_strength.py').read_text())
    checks['no_neural_calls']=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['learn','predict','train'] for n in ast.walk(tree))
    output=[];prefix=None
    for entry in review['episodes']:
        directory=root/entry['label']/'episode-0001';result=json.loads((directory/'report.json').read_text())
        physical=[json.loads(s) for s in (directory/'physics/frames.ndjson').read_text().splitlines()]
        actions=[json.loads(s) for s in (directory/'actions.ndjson').read_text().splitlines()]
        alignment_rows=[json.loads(s) for s in (directory/'alignment.ndjson').read_text().splitlines()]
        after={r['frame']:r for r in alignment_rows if r['stage']=='after'}
        reset=next(r for r in alignment_rows if r['stage']=='reset')
        policies=[json.loads(s) for s in (directory/'policy.ndjson').read_text().splitlines()]
        n=result['physical_frames']
        c=dict(frame_sequence=[x['frame'] for x in physical]==list(range(1,n+1)),
               action_sequence=[x['frame'] for x in actions]==list(range(1,n+1)),
               hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
               material_guards=all(x['hencky_material_guard']['safe'] for x in physical),
               sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physical),
               zero_reset=all(result['zero_step_exact'].values()),source_unchanged=result['source_unchanged'],
               mode=result['waiting_action_mode']=='error_strength_v1',lead_limit=result['maximum_reference_lead_frames']<=6+1e-9)
        exact=True;count=0;follow_after_wait=0;handoff_ok=True;invalidated_frames=0
        for row in policies:
            if row['frame_after']==row['frame_before']:continue
            original=np.asarray(row['action'])
            first=actions[row['frame_before']]
            sampled=control_mode(first['cursor_before'],after.get(row['frame_before'],reset),first['hold_frame'])
            stale=False
            for frame in range(row['frame_before']+1,row['frame_after']+1):
                actual=actions[frame-1]
                mode=control_mode(actual['cursor_before'],actual['alignment'],actual['hold_frame'])
                stale |= mode!=sampled
                action=original.copy()
                if stale:
                    action=np.zeros(5)
                    if mode=='wait':action[:4]=-1
                    invalidated_frames+=1
                handoff_ok &= actual['action_handoff_version']==ACTION_HANDOFF_VERSION
                handoff_ok &= actual['decision_mode']==sampled and actual['execution_mode']==mode
                handoff_ok &= actual['action_invalidated']==stale
                handoff_ok &= np.array_equal(actual['policy_action'],original) and np.array_equal(actual['effective_action'],action)
                if actual['hold_frame'] is None:
                    base,delta,request,pace=command_at(commands,actual['cursor_before'],action,
                        reference_limit=max(actual['cursor_before'],actual['alignment']['reference_limit']),
                        wait_target_error=actual['alignment']['wait_target_error_si'],waiting_action_mode='error_strength_v1')
                else:base=delta=request=np.zeros(4);pace=0.
                exact=exact and all(np.allclose(v,actual[k],rtol=0,atol=1e-15) for v,k in
                    [(base,'baseline_si'),(delta,'residual_si'),(request,'request_si')]) and abs(pace-actual['pace'])<1e-14
                exact=exact and np.array_equal(actual['applied_si'],physical[frame-1]['applied_command_si'])
                count+=1
                if sampled=='wait' and pace>0 and np.any(delta):follow_after_wait+=1
                next_cursor=min(1170.,actual['cursor_before']+actual['pace']) if actual['hold_frame'] is None else actual['cursor_before']
                stale |= control_mode(next_cursor,after[frame],actual['hold_frame'])!=sampled
        c['all_policy_frames_recomputed']=exact and count==n
        c['handoff_log_matches_rule']=bool(handoff_ok)
        c['no_wait_action_reinterpreted_as_follow_residual']=follow_after_wait==0
        c['handoff_version']=result['action_handoff_version']==ACTION_HANDOFF_VERSION
        waits=[x for x in actions if x['pace']==0 and x['hold_frame'] is None]
        c['wait_direction']=all(np.all(np.asarray(x['residual_si'])*np.asarray(x['alignment']['wait_target_error_si'])>=-1e-15) for x in waits)
        c['wait_cap']=all(np.all(abs(np.asarray(x['residual_si']))<=.1*LIMITS_SI+1e-15) for x in waits)
        c['no_wait_overshoot']=all(np.all(abs(np.asarray(x['residual_si']))/60<=abs(np.asarray(x['alignment']['wait_target_error_si']))+1e-15) for x in waits)
        if entry['strength'] is None:
            c['endpoint_arrays_exact']=len(entry['manual_endpoint_arrays_exact'])==24 and all(entry['manual_endpoint_arrays_exact'].values())
            c['manual_baseline_exact']=entry.get('manual_baseline_exact',False) and result['retained_fill_fraction']==1.4538132234025687
        else:
            if prefix is None:prefix=physical[:1020]
            c['loaded_prefix_exact']=len(physical)>=1020 and all(x['after_state']==y['after_state'] and x['applied_command_si']==y['applied_command_si'] for x,y in zip(prefix,physical[:1020]))
            c['stall_established']=entry['recovery_start']['frame']==1020
            if entry['strength']==0:
                c['zero_strength_stops']=all(not np.any(x['residual_si']) for x in waits if x['frame']>1020)
        if review.get('collection_enabled'):
            dataset=entry['dataset'];require_environment_contract(dataset)
            c['dataset_sha256']=digest(dataset['path'])==dataset['sha256']
            with np.load(dataset['path'],allow_pickle=False) as data:
                nrows=len(policies)
                c['dataset_shapes']=data['observations'].shape==(nrows,221) and data['actions'].shape==(nrows,5)
                c['dataset_finite']=all(np.isfinite(data[k]).all() for k in ['observations','next_observations','actions','rewards','raw_rewards'])
                c['dataset_observation_chain']=np.array_equal(data['next_observations'][:-1],data['observations'][1:])
                c['dataset_terminal']=bool(data['dones'][-1]) and not data['dones'][:-1].any() and np.array_equal(data['dones'],data['terminated']|data['truncated'])
                c['dataset_actions_rewards_frames']=all(np.array_equal(data[k],np.asarray([p[v] for p in policies])) for k,v in
                    [('actions','action'),('raw_rewards','raw_reward'),('rewards','reward'),('frames_before','frame_before'),('frames_after','frame_after')])
                expected=data['raw_rewards']+np.where(data['dones'],0.,data['next_observations'][:,-2].astype(float))-data['observations'][:,-2].astype(float)
                c['dataset_potential_rewards']=np.allclose(data['rewards'],expected,rtol=0,atol=1e-12)
                c['dataset_full_return']=abs(float(data['rewards'].sum()-data['raw_rewards'].sum()))<1e-10
        if result['success']:
            c['hold_scores_valid']=len(result['scores'])==3 and all(x['valid_estimate'] for x in result['scores'])
            if result.get('terminal_score_version')==TERMINAL_SCORE_VERSION:
                c['gap_diagnostic_only']=result['gap_role']=='diagnostic_only'
            else:c['hold_gaps_valid']=all(x['remaining_soil_gap_m'] is not None and x['remaining_soil_gap_m']>=.003 for x in result['gap_checks'])
        output.append(dict(label=entry['label'],strength=entry['strength'],status=result['status'],success=result['success'],
                frames=n,cursor=result['cursor'],actual_progress=result['last_alignment']['actual_progress'],
                wait_frames=len(waits),active_wait_frames=sum(any(abs(v)>1e-10 for v in x['residual_si']) for x in waits),
                nonzero_follow_frames_from_held_wait_action=follow_after_wait,invalidated_frames=invalidated_frames,
                recovery_start=entry.get('recovery_start'),recovered_progress=entry.get('recovered_progress'),
                qualified_fill_percent=100*result['retained_fill_fraction'] if result['success'] else None,
                raw_retained_fill_fraction=result.get('retained_fill_fraction'),checks=c,audit_passed=all(c.values())))
    checks['all_finished_episode_audits']=all(x['audit_passed'] for x in output)
    checks['all_cases_finished']=review['status']=='completed' and len(output)==4
    result=dict(status='passed' if all(checks.values()) else 'failed_or_incomplete',global_checks=checks,
                episodes=output,check_count=len(checks)+sum(len(x['checks']) for x in output),
                new_physics_frames=0,neural_updates=0,training_admission=False,
                qualified_recovery_strengths=[x['strength'] for x in output if x['strength'] is not None and x['success'] and x['recovered_progress']])
    (root/'read-only-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main(Path(sys.argv[1]))
