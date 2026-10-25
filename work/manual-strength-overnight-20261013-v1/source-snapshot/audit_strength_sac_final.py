"""Audit saved evidence only: no simulator reset, step, or optimizer call."""
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import (
    ROOT, INPUT, SOURCE, INPUT_SHA, SOURCE_SHA, LIMITS_SI,
    command_at, control_mode, environment_contract,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


out = ROOT/'runs/manual-strength-sac-20261013-v1'
m = json.loads((out/'manifest.json').read_text())
b = json.loads((ROOT/'runs/manual-strength-sac-20261013-budget.json').read_text())
t = m['reward_transfer']
commands = np.asarray([s['applied_command_si'] for s in rows(INPUT)[:1170]])
checks = dict(
    completed=m['status'] == 'completed',
    budget_duration=b['seconds'] == 1800 and b['purpose'] == 'training_and_evaluation',
    within_budget=m['remaining_seconds'] >= 0 and
        datetime.fromisoformat(m['finished_utc']).timestamp() <= b['epoch_deadline'],
    environment_contract=m['environment_contract'] == environment_contract('error_strength_v1'),
    code_current=all(sha(p) == h for p, h in m['code'].items()),
    code_snapshot=all(sha(ROOT/'work/manual-strength-sac-preflight-20261013-v1/source-snapshot'/Path(p).name) == h
                      for p, h in m['code'].items()),
    inputs_unchanged=sha(INPUT) == INPUT_SHA and sha(SOURCE) == SOURCE_SHA,
    seed_replay_unchanged=sha(t['replay_path']) == t['replay_sha256'],
    fresh_policy_route='resume_from' not in m and m['replay_seed_transitions'] == 1798,
    final_model_identical=sha(out/'sac-last.zip') == sha(out/'sac-round-01.zip'),
    no_accepted_improvement=m['accepted_improvement'] is False and m['recommendation'] == 'retain_manual_baseline',
)
torch.set_num_threads(1)
model = SAC.load(out/'sac-last.zip', device='cpu')
checks['model_finite'] = all(torch.isfinite(p).all().item() for p in model.policy.parameters())
checks['model_updates'] = model._n_updates == m['final_gradient_updates'] == 2509
checks['model_timesteps'] = model.num_timesteps == m['final_timesteps'] == 2510
with Path(t['replay_path']).open('rb') as f:
    initial = pickle.load(f)
with (out/'replay-buffer.pkl').open('rb') as f:
    replay = pickle.load(f)
n = replay.size()
checks['replay_size'] = not replay.full and n == 1798 + 2509
checks['seed_prefix_exact'] = all(np.array_equal(getattr(initial, k)[:1798], getattr(replay, k)[:1798])
                                for k in ['observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts'])
checks['replay_finite'] = all(np.isfinite(getattr(replay, k)[:n]).all()
                            for k in ['observations', 'next_observations', 'actions', 'rewards'])
checks['no_timeout_bootstrap'] = not replay.handle_timeout_termination and not replay.timeouts[:n].any()
checks['terminal_count'] = int(replay.dones[:n].sum()) == 4 + 12
episode_audits = []
directories = sorted((out/'training').glob('episode-*')) + [out/'final-candidate-1/episode-0001', out/'final-candidate-2/episode-0001']
for directory in directories:
    result = json.loads((directory/'report.json').read_text())
    physical = rows(directory/'physics/frames.ndjson')
    actions = rows(directory/'actions.ndjson')
    alignment = rows(directory/'alignment.ndjson')
    after = {x['frame']: x for x in alignment if x['stage'] == 'after'}
    reset = next(x for x in alignment if x['stage'] == 'reset')
    count = result['physical_frames']
    c = dict(frame_sequence=[x['frame'] for x in physical] == list(range(1, count+1)),
             action_sequence=[x['frame'] for x in actions] == list(range(1, count+1)),
             hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
             material_guards=all(x['hencky_material_guard']['safe'] for x in physical),
             sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physical),
             exact_reset=all(result['zero_step_exact'].values()),
             lead_limit=result['maximum_reference_lead_frames'] <= 6 + 1e-9,
             runtime_binding=all(sha(ROOT/'src/v4'/p) == h for p, h in result['runtime_bindings'].items()))
    exact = handoff = True
    stale = False
    for i, action_row in enumerate(actions):
        if i % 3 == 0:
            original = np.asarray(action_row['policy_action'])
            sampled = control_mode(action_row['cursor_before'], after.get(i, reset), action_row['hold_frame'])
            stale = False
        mode = control_mode(action_row['cursor_before'], action_row['alignment'], action_row['hold_frame'])
        stale |= mode != sampled
        effective = original.copy()
        if stale:
            effective = np.zeros(5)
            if mode == 'wait':
                effective[:4] = -1
        handoff &= (action_row['decision_mode'] == sampled and action_row['execution_mode'] == mode
                    and action_row['action_invalidated'] == stale
                    and np.array_equal(action_row['policy_action'], original)
                    and np.array_equal(action_row['effective_action'], effective))
        if action_row['hold_frame'] is None:
            base, delta, request, pace = command_at(commands, action_row['cursor_before'], effective,
                reference_limit=max(action_row['cursor_before'], action_row['alignment']['reference_limit']),
                wait_target_error=action_row['alignment']['wait_target_error_si'], waiting_action_mode='error_strength_v1')
        else:
            base = delta = request = np.zeros(4)
            pace = 0.
        exact &= all(np.allclose(v, action_row[k], rtol=0, atol=1e-15)
                     for v, k in [(base, 'baseline_si'), (delta, 'residual_si'), (request, 'request_si')])
        exact &= abs(pace-action_row['pace']) < 1e-14 and np.array_equal(action_row['applied_si'], physical[i]['applied_command_si'])
        next_cursor = min(1170., action_row['cursor_before']+pace) if action_row['hold_frame'] is None else action_row['cursor_before']
        stale |= control_mode(next_cursor, after[i+1], action_row['hold_frame']) != sampled
    waits = [x for x in actions if x['execution_mode'] == 'wait']
    c['commands_recomputed'] = bool(exact)
    c['handoff_recomputed'] = bool(handoff)
    c['wait_direction_cap_no_overshoot'] = all(
        np.all(np.asarray(x['residual_si'])*np.asarray(x['alignment']['wait_target_error_si']) >= -1e-15)
        and np.all(abs(np.asarray(x['residual_si'])) <= LIMITS_SI*.1+1e-15)
        and np.all(abs(np.asarray(x['residual_si']))/60 <= abs(np.asarray(x['alignment']['wait_target_error_si']))+1e-15)
        for x in waits)
    c['failed_score_not_qualified'] = result['success'] is False and result.get('retained_fill_fraction') is None
    episode_audits.append(dict(path=str(directory), status=result['status'], physical_frames=count,
        actual_progress=result['last_alignment']['actual_progress'], cursor=result.get('cursor', actions[-1]['cursor_before']),
        wait_frames=len(waits), checks=c, passed=all(c.values())))
checks['all_episode_audits'] = all(x['passed'] for x in episode_audits)
checks['two_frozen_resets_same_outcome'] = len(m['final_evaluations']) == 2 and all(
    x['status'] == 'alignment_stop' and x['physical_frames'] == 675 and not x['success'] for x in m['final_evaluations'])
checks['evaluation_endpoints_identical'] = len({x['checkpoint']['sha256'] for x in m['final_evaluations']}) == 1 and all(
    sha(x['checkpoint']['path']) == x['checkpoint']['sha256'] for x in m['final_evaluations'])
audit = dict(status='passed' if all(checks.values()) else 'failed', checks=checks, episodes=episode_audits,
    check_count=len(checks)+sum(len(x['checks']) for x in episode_audits),
    new_physics_frames=0, neural_updates=0, replay_size=n,
    limitation='No raw evaluation observations were saved; actor predictions cannot be independently recomputed. Same scene repeats are reproducibility only.',
    partial_episode='episode-0013 closed early: final executed callback-stop step was not added; no fabricated terminal.',
    recommendation='retain_manual_baseline', source_manifest_sha256=sha(out/'manifest.json'))
path = out/'read-only-audit.json'
if path.exists():
    raise ValueError('immutable_audit_exists')
path.write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit))
