"""Independently recompute frozen predictions, command handoff and saved returns."""
import ast
import json
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import ROOT, INPUT, SOURCE, INPUT_SHA, SOURCE_SHA, sha, command_at, control_mode, LIMITS_SI


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


root=Path(sys.argv[1])
review=json.loads((root/'review.json').read_text())
budget=json.loads(Path(review['budget_path']).read_text())
model=SAC.load(review['policy_checkpoint']['path'],device='cpu')
torch.set_num_threads(1)
commands=np.asarray([x['applied_command_si'] for x in rows(INPUT)[:1170]])
code=ast.parse((root/'source-snapshot/evaluate_manual_residual_frozen.py').read_text())
global_checks=dict(completed=review['status']=='completed',
    budget_not_reopened=budget['seconds']==28800 and budget['epoch_deadline']==review['budget_deadline'],
    within_budget=review['remaining_seconds']>=0 and datetime.fromisoformat(review['finished_utc']).timestamp()<=budget['epoch_deadline'],
    source_input_unchanged=sha(INPUT)==INPUT_SHA and sha(SOURCE)==SOURCE_SHA,
    code_snapshots=all(sha(root/'source-snapshot'/n)==h for n,h in review['code'].items()),
    current_code=all(sha(ROOT/'src/v4'/n)==h for n,h in review['code'].items()),
    checkpoint_unchanged=sha(review['policy_checkpoint']['path'])==review['policy_checkpoint']['sha256'],
    neural_updates_zero=review['neural_updates']==0,
    learned_checkpoint=model._n_updates==review['policy_checkpoint']['updates'] and model._n_updates>0,
    no_training_calls=not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr in ['learn','train','backward','step']
                             for x in ast.walk(code)),
    receipt_unchanged=sha(review['receipt']['path'])==review['receipt']['sha256'])
episodes=[]
for entry in review['episodes']:
    result=entry['result'];directory=Path(result['path'])
    actions=rows(directory/'actions.ndjson');physics=rows(directory/'physics/frames.ndjson')
    alignment=rows(directory/'alignment.ndjson');policies=rows(directory/'policy.ndjson')
    after={x['frame']:x for x in alignment if x['stage']=='after'}
    reset=next(x for x in alignment if x['stage']=='reset')
    n=result['physical_frames']
    c=dict(fresh_exact_reset=all(result['zero_step_exact'].values()),
        source_unchanged=result['source_unchanged'],
        endpoint_checkpoint=sha(result['checkpoint']['path'])==result['checkpoint']['sha256'],
        runtime_bindings=all(sha(ROOT/'src/v4'/p)==h for p,h in result['runtime_bindings'].items()),
        sequences=[x['frame'] for x in physics]==[x['frame'] for x in actions]==list(range(1,n+1)),
        hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physics),
        material_guards=all(x['hencky_material_guard']['safe'] for x in physics),
        sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physics),
        lead_limit=result['maximum_reference_lead_frames']<=6+1e-9)
    exact=handoff=predictions=raw_rewards=True;count=0;previous=np.zeros(5)
    for row in policies:
        obs=np.asarray(row['observation'],dtype=np.float32)
        original=np.asarray(row['action'])
        predictions &= np.array_equal(model.predict(obs,deterministic=True)[0],original)
        expected_raw=-(row['frame_after']-row['frame_before'])*.000005-float(np.sum((original[:4]-previous[:4])**2))*.000001
        if row['terminated'] or row['truncated']:
            expected_raw+=result['retained_fill_fraction'] if result['success'] else -2.
        raw_rewards &= abs(expected_raw-row['raw_reward'])<1e-12
        previous=original.copy()
        if row['frame_before']==row['frame_after']:continue
        first=actions[row['frame_before']]
        sampled=control_mode(first['cursor_before'],after.get(row['frame_before'],reset),first['hold_frame'])
        stale=False
        for frame in range(row['frame_before']+1,row['frame_after']+1):
            actual=actions[frame-1];mode=control_mode(actual['cursor_before'],actual['alignment'],actual['hold_frame'])
            stale |= mode!=sampled
            effective=original.copy()
            if stale:
                effective=np.zeros(5)
                if mode=='wait':effective[:4]=-1
            handoff &= actual['decision_mode']==sampled and actual['execution_mode']==mode and actual['action_invalidated']==stale
            handoff &= np.array_equal(actual['policy_action'],original) and np.array_equal(actual['effective_action'],effective)
            if actual['hold_frame'] is None:
                base,delta,request,pace=command_at(commands,actual['cursor_before'],effective,
                    reference_limit=max(actual['cursor_before'],actual['alignment']['reference_limit']),
                    wait_target_error=actual['alignment']['wait_target_error_si'],waiting_action_mode='error_strength_v1')
            else:base=delta=request=np.zeros(4);pace=0.
            exact &= all(np.allclose(v,actual[k],rtol=0,atol=1e-15) for v,k in [(base,'baseline_si'),(delta,'residual_si'),(request,'request_si')])
            exact &= abs(pace-actual['pace'])<1e-14 and np.array_equal(actual['applied_si'],physics[frame-1]['applied_command_si'])
            next_cursor=min(1170.,actual['cursor_before']+pace) if actual['hold_frame'] is None else actual['cursor_before']
            stale |= control_mode(next_cursor,after[frame],actual['hold_frame'])!=sampled
            count+=1
    c.update(frozen_predictions_exact=bool(predictions),all_commands_recomputed=bool(exact) and count==n,handoff_recomputed=bool(handoff),raw_rewards_recomputed=bool(raw_rewards))
    waits=[x for x in actions if x['execution_mode']=='wait']
    c['wait_bounds']=all(np.all(np.asarray(x['residual_si'])*np.asarray(x['alignment']['wait_target_error_si'])>=-1e-15)
        and np.all(abs(np.asarray(x['residual_si']))<=.1*LIMITS_SI+1e-15)
        and np.all(abs(np.asarray(x['residual_si']))/60<=abs(np.asarray(x['alignment']['wait_target_error_si']))+1e-15) for x in waits)
    dataset=entry['dataset']
    c['dataset_sha']=sha(dataset['path'])==dataset['sha256']
    with np.load(dataset['path'],allow_pickle=False) as d:
        c['observation_chain']=np.array_equal(d['next_observations'][:-1],d['observations'][1:])
        c['causal_observations']=np.array_equal(d['observations'],np.asarray([x['observation'] for x in policies],dtype=np.float32))
        c['terminal_not_bootstrapped']=bool(d['dones'][-1]) and not d['dones'][:-1].any() and np.array_equal(d['dones'],d['terminated']|d['truncated'])
        c['policy_data_exact']=np.array_equal(d['actions'],np.asarray([x['action'] for x in policies],dtype=np.float32)) and np.allclose(d['rewards'],[x['reward'] for x in policies],rtol=0,atol=1e-12)
        expected=d['raw_rewards']+np.where(d['dones'],0.,d['next_observations'][:,-2].astype(float))-d['observations'][:,-2].astype(float)
        c['progress_rewards']=np.allclose(d['rewards'],expected,rtol=0,atol=1e-12) and abs(float(d['rewards'].sum()-d['raw_rewards'].sum()))<1e-10
    if result['success']:
        scores=result['scores'];start=scores[0]['frame']
        c['hold_score_valid']=len(scores)==3 and all(x['valid_estimate'] and x['occupancy_level']==.2 and x['capacity_liters']==255.164729950989 for x in scores)
        c['hold_duration_and_entry']=[x['frame'] for x in scores]==[start,start+15,start+30] and n==start+30 and after[start]['endpoint_ready']
        c['score_minimum']=result['retained_fill_fraction']==min(x['fill_percent'] for x in scores)/100
        c['gap_not_gate']=result['gap_role']=='diagnostic_only'
    else:c['no_failed_qualified_score']=result.get('retained_fill_fraction') is None
    episodes.append(dict(path=str(directory),status=result['status'],success=result['success'],physical_frames=n,
        fill_percent=100*result['retained_fill_fraction'] if result['success'] else None,checks=c,passed=all(c.values())))
global_checks['all_episodes']=all(x['passed'] for x in episodes)
goal_met=len(episodes)==3 and all(x['passed'] and x['success'] and x['fill_percent']>=145.38132234025687 for x in episodes)
report=dict(status='passed' if all(global_checks.values()) else 'failed',global_checks=global_checks,episodes=episodes,
    goal_met=goal_met,stretch_met=goal_met and min(x['fill_percent'] for x in episodes)>=147.38132234025687,
    check_count=len(global_checks)+sum(len(x['checks']) for x in episodes),new_physics_frames=0,neural_updates=0,
    scope='same initial scene reproducibility; not generalization or truck delivery',source_review_sha256=sha(root/'review.json'))
path=root/'read-only-audit.json'
if path.exists():raise ValueError('immutable_audit_exists')
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
