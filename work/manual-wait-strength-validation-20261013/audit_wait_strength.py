"""Offline command recomputation and physical-evidence audit; never steps physics."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from manual_residual_sac_env import command_at, INPUT, SOURCE, INPUT_SHA, SOURCE_SHA, LIMITS_SI

ROOT=Path('/home/qunwang/wajueji')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root):
    review=json.loads((root/'review.json').read_text());budget=json.loads((root/'budget.json').read_text())
    commands=np.asarray([json.loads(s)['applied_command_si'] for s in INPUT.read_text().splitlines()][:1170])
    checks=dict(source_unchanged=digest(INPUT)==INPUT_SHA and digest(SOURCE)==SOURCE_SHA,
                new_budget_1200=budget['seconds']==1200 and budget['purpose']=='wait_strength_control_validation_only',
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
        policies=[json.loads(s) for s in (directory/'policy.ndjson').read_text().splitlines()]
        n=result['physical_frames']
        c=dict(frame_sequence=[x['frame'] for x in physical]==list(range(1,n+1)),
               action_sequence=[x['frame'] for x in actions]==list(range(1,n+1)),
               hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
               material_guards=all(x['hencky_material_guard']['safe'] for x in physical),
               sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physical),
               zero_reset=all(result['zero_step_exact'].values()),source_unchanged=result['source_unchanged'],
               mode=result['waiting_action_mode']=='error_strength_v1',lead_limit=result['maximum_reference_lead_frames']<=6+1e-9)
        exact=True;count=0;follow_after_wait=0
        for row in policies:
            action=np.asarray(row['action'])
            for frame in range(row['frame_before']+1,row['frame_after']+1):
                actual=actions[frame-1]
                if actual['hold_frame'] is None:
                    base,delta,request,pace=command_at(commands,actual['cursor_before'],action,
                        reference_limit=max(actual['cursor_before'],actual['alignment']['reference_limit']),
                        wait_target_error=actual['alignment']['wait_target_error_si'],waiting_action_mode='error_strength_v1')
                else:base=delta=request=np.zeros(4);pace=0.
                exact=exact and all(np.allclose(v,actual[k],rtol=0,atol=1e-15) for v,k in
                    [(base,'baseline_si'),(delta,'residual_si'),(request,'request_si')]) and abs(pace-actual['pace'])<1e-14
                exact=exact and np.array_equal(actual['applied_si'],physical[frame-1]['applied_command_si'])
                count+=1
                if row['mode_before'].startswith('wait') and pace>0 and np.any(delta):follow_after_wait+=1
        c['all_policy_frames_recomputed']=exact and count==n
        waits=[x for x in actions if x['pace']==0 and x['hold_frame'] is None]
        c['wait_direction']=all(np.all(np.asarray(x['residual_si'])*np.asarray(x['alignment']['wait_target_error_si'])>=-1e-15) for x in waits)
        c['wait_cap']=all(np.all(abs(np.asarray(x['residual_si']))<=.1*LIMITS_SI+1e-15) for x in waits)
        c['no_wait_overshoot']=all(np.all(abs(np.asarray(x['residual_si']))/60<=abs(np.asarray(x['alignment']['wait_target_error_si']))+1e-15) for x in waits)
        if entry['strength'] is None:
            c['manual_baseline_exact']=entry.get('manual_baseline_exact',False) and result['retained_fill_fraction']==1.4538132234025687
        else:
            if prefix is None:prefix=physical[:1020]
            c['loaded_prefix_exact']=len(physical)>=1020 and all(x['after_state']==y['after_state'] and x['applied_command_si']==y['applied_command_si'] for x,y in zip(prefix,physical[:1020]))
            c['stall_established']=entry['recovery_start']['frame']==1020
            if entry['strength']==0:
                c['zero_strength_stops']=all(not np.any(x['residual_si']) for x in waits if x['frame']>1020)
        if result['success']:
            c['hold_scores_valid']=len(result['scores'])==3 and all(x['valid_estimate'] for x in result['scores'])
            c['hold_gaps_valid']=all(x['remaining_soil_gap_m'] is not None and x['remaining_soil_gap_m']>=.003 for x in result['gap_checks'])
        output.append(dict(label=entry['label'],strength=entry['strength'],status=result['status'],success=result['success'],
                frames=n,cursor=result['cursor'],actual_progress=result['last_alignment']['actual_progress'],
                wait_frames=len(waits),active_wait_frames=sum(any(abs(v)>1e-10 for v in x['residual_si']) for x in waits),
                nonzero_follow_frames_from_held_wait_action=follow_after_wait,
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
