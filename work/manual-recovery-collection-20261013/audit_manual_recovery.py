"""Read-only audit of completed recovery collection; never advances physics."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/home/qunwang/wajueji')
sys.path.insert(0,str(ROOT/'src/v4'))
from manual_trajectory_residual import LIMITS_SI


def audit(root):
    review=json.loads((root/'review.json').read_text())
    reports=[]
    for entry in review['episodes']:
        label=entry['label'];episode=root/label/'episode-0001'
        physical=[json.loads(x) for x in (episode/'physics/frames.ndjson').read_text().splitlines()]
        commands=[json.loads(x) for x in (episode/'actions.ndjson').read_text().splitlines()]
        policies=[json.loads(x) for x in (episode/'policy.ndjson').read_text().splitlines()]
        result=entry['result'];n=result['physical_frames']
        old_roots={
            'early43':ROOT/'runs/manual-residual-sac-aligned-20261013/batch-v2/evaluation-01/episode-0001',
            'early52':ROOT/'runs/manual-residual-sac-aligned-20261013/batch-v2/replay-update-evaluation/evaluation/episode-0001',
            'loaded970':ROOT/'runs/manual-progress-alignment-20261013-v3/loaded_correction/episode-0001'}
        with (old_roots[label]/'physics/frames.ndjson').open() as stream:
            old_prefix=[json.loads(line) for line in itertools.islice(stream,entry['prefix_frames'])]
        checks=dict(physical_frames_complete=[x['frame'] for x in physical]==list(range(1,n+1)),
                    commands_complete=[x['frame'] for x in commands]==list(range(1,n+1)),
                    all_hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
                    all_material_guards=all(x['hencky_material_guard']['safe'] for x in physical),
                    all_sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physical),
                    source_unchanged=result['source_unchanged'],
                    prefix_reproduced=entry['prefix_commands_exact']==entry['prefix_frames'] and entry['prefix_compact_exact']==entry['prefix_frames'],
                    lead_bounded=result['maximum_reference_lead_frames']<=6+1e-9)
        checks['prefix_independent_exact']=len(old_prefix)==entry['prefix_frames'] and all(
            a['after_state']==b['after_state'] and a['applied_command_si']==b['applied_command_si']
            for a,b in zip(old_prefix,physical))
        checks['dataset_sha_bound']=hashlib.sha256((root/label/'transitions.npz').read_bytes()).hexdigest()==entry['dataset']['sha256']
        waiting=[x for x in commands if x['pace']==0 and x['hold_frame'] is None]
        direction_ok=cap_ok=no_overshoot=True
        for x in waiting:
            delta=np.asarray(x['residual_si']);error=np.asarray(x['alignment']['wait_target_error_si'])
            direction_ok=direction_ok and bool(np.all(delta*error>=-1e-15))
            cap_ok=cap_ok and bool(np.all(abs(delta)<=LIMITS_SI*.1+1e-12))
            no_overshoot=no_overshoot and bool(np.all(abs(delta)/60<=abs(error)+1e-12))
        checks.update(wait_correction_direction=direction_ok,wait_correction_cap=cap_ok,wait_correction_no_target_overshoot=no_overshoot)
        with np.load(root/label/'transitions.npz',allow_pickle=False) as data:
            count=len(data['rewards'])
            checks.update(observation_shape=data['observations'].shape==(count,221),
                          action_shape=data['actions'].shape==(count,5),
                          policy_rows_match=len(policies)==count,
                          observation_chain=np.array_equal(data['next_observations'][:-1],data['observations'][1:]),
                          terminal_preserved=bool(data['dones'][-1]) and int(data['dones'].sum())==1,
                          correct_terminal_type=np.array_equal(data['dones'],data['terminated']|data['truncated']),
                          total_reward_equivalent=abs(float(data['rewards'].sum()-data['raw_rewards'].sum()))<1e-10,
                          nonempty_true_recovery=bool(np.any(data['policy_source']=='fixed_correction')))
            checks['actions_match_policy_log']=np.array_equal(data['actions'],np.asarray([x['action'] for x in policies],dtype=np.float32))
            expected=data['raw_rewards']+np.where(data['dones'],0.,data['next_observations'][:,-2].astype(float))-data['observations'][:,-2].astype(float)
            checks['per_step_reward_equivalent']=bool(np.allclose(data['rewards'],expected,rtol=0,atol=1e-12))
            if result['success']:
                checks['hold_samples_valid']=len(result['scores'])==3 and all(x['valid_estimate'] for x in result['scores'])
                checks['hold_gaps_valid']=all(x['remaining_soil_gap_m'] is not None and x['remaining_soil_gap_m']>=.003 for x in result['gap_checks'])
        reports.append(dict(label=label,status='passed' if all(checks.values()) else 'failed',checks=checks,
                            physical_frames=n,transitions=count,wait_frames=len(waiting),
                            nonzero_wait_correction_frames=sum(any(abs(v)>1e-10 for v in x['residual_si']) for x in waiting),
                            recovery_demonstration_qualified=entry['recovery_demonstration_qualified'],
                            held_fill_percent=None if result.get('retained_fill_fraction') is None else result['retained_fill_fraction']*100))
    tree=ast.parse((root/'source-snapshot/collect_manual_recovery.py').read_text())
    no_training=not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['learn','train'] for n in ast.walk(tree))
    budget=json.loads((root/'budget.json').read_text())
    global_checks=dict(no_training=no_training,collection_budget_only=budget['purpose']=='recovery_collection_validation_only' and budget['seconds']==1200,
                       finished_within_budget=review.get('remaining_seconds',-1)>=0,
                       source_snapshot_bound=all(hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==expected
                                                 for name,expected in review['code'].items()))
    output=dict(status='passed' if len(reports)==3 and all(x['status']=='passed' for x in reports) and all(global_checks.values()) else 'incomplete_or_failed',
                new_physics_frames=0,neural_training_calls_absent=no_training,global_checks=global_checks,episodes=reports,
                all_recovery_demonstrations_qualified=review.get('all_recovery_demonstrations_qualified',False))
    (root/'independent-audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output))


if __name__=='__main__':audit(Path(sys.argv[1]))
