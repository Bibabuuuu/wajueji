"""Audit saved 1:1 retraining evidence without advancing physics or learning."""
import datetime
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np
from manual_trajectory_residual import LIMITS_SI

ROOT=Path('/home/qunwang/wajueji')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    receipt=json.loads((root/'initial-pool/transfer.json').read_text())
    manifest=json.loads((root/'batch-v1/manifest.json').read_text())
    budget=json.loads((root/'budget.json').read_text())
    checks=dict(source_files_unchanged=all(digest(p)==h for p,h in receipt['sources'].items()),
                initial_replay_hash=digest(receipt['replay_path'])==receipt['replay_sha256'],
                budget_1800=budget['seconds']==1800 and budget['purpose']=='training_and_evaluation',
                initial_ratio=receipt['new_transitions']==receipt['old_transitions']==1350,
                fresh_initialization=manifest.get('replay_seed_transitions')==2700 and 'resume_from' not in manifest,
                source_snapshot=all(digest(root/'source-snapshot'/Path(p).name)==h for p,h in manifest['code'].items()),
                production_code_unchanged=all(digest(p)==h for p,h in manifest['code'].items()),
                completed=manifest['status']=='completed',
                within_budget=manifest.get('remaining_seconds',-1)>=0,
                evaluation_reserved=manifest.get('evaluation_reserve_seconds')==630)
    with Path(receipt['replay_path']).open('rb') as f:initial=pickle.load(f)
    with (root/'batch-v1/replay-buffer.pkl').open('rb') as f:final=pickle.load(f)
    for field in ['observations','next_observations','actions','rewards','dones','timeouts']:
        checks['initial_'+field+'_preserved']=np.array_equal(getattr(initial,field)[:2700],getattr(final,field)[:2700])
    checks['capacity_preserved']=final.buffer_size==30000 and not final.full
    checks['online_transitions_retained']=final.size()==2700+manifest['final_gradient_updates']
    checks['no_timeout_bootstrap']=not final.handle_timeout_termination and not final.timeouts[:final.size()].any()
    selected=manifest['training_chunks'][-1]['checkpoint']
    checks['same_frozen_checkpoint']=all(x['policy_checkpoint']==selected for x in manifest['final_evaluations'])
    episodes=[]
    for report_path in sorted((root/'batch-v1').glob('*/episode-*/report.json')):
        result=json.loads(report_path.read_text());directory=report_path.parent
        n=result['physical_frames'];physical=[]
        with (directory/'physics/frames.ndjson').open() as f:
            for line in f:physical.append(json.loads(line))
        commands=[json.loads(x) for x in (directory/'actions.ndjson').read_text().splitlines()]
        c=dict(physical_frames_complete=[x['frame'] for x in physical]==list(range(1,n+1)),
               commands_complete=[x['frame'] for x in commands]==list(range(1,n+1)),
               hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in physical),
               material_guards=all(x['hencky_material_guard']['safe'] for x in physical),
               sweep_guards=all(x['moving_bucket_sweep']['safe'] for x in physical),
               lead_bounded=result['maximum_reference_lead_frames']<=6+1e-9,
               source_identity=digest(ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz')==result['source_sha256'],
               exact_reset=all(result['zero_step_exact'].values()))
        waiting=[x for x in commands if x['pace']==0 and x['hold_frame'] is None]
        c['directed_wait']=all(np.all(np.asarray(x['residual_si'])*np.asarray(x['alignment']['wait_target_error_si'])>=-1e-15) for x in waiting)
        c['wait_cap']=all(np.all(abs(np.asarray(x['residual_si']))<=LIMITS_SI*.1+1e-12) for x in waiting)
        c['no_target_overshoot']=all(np.all(abs(np.asarray(x['residual_si']))/60<=abs(np.asarray(x['alignment']['wait_target_error_si']))+1e-12) for x in waiting)
        if result['success']:
            c['valid_hold']=len(result['scores'])==3 and all(x['valid_estimate'] for x in result['scores'])
            c['off_soil']=all(x['remaining_soil_gap_m'] is not None and x['remaining_soil_gap_m']>=.003 for x in result['gap_checks'])
        recovery_bursts=0;waiting_progress=0;maximum_wait_progress=0.;anchor=None
        for x in commands:
            progress=x['alignment']['actual_progress']
            if x['pace']==0 and x['hold_frame'] is None:
                if anchor is None:anchor=progress
                maximum_wait_progress=max(maximum_wait_progress,progress-anchor)
            elif anchor is not None:
                if progress-anchor>10:recovery_bursts+=1
                waiting_progress+=max(0,progress-anchor);anchor=None
        episodes.append(dict(path=str(directory),status=result['status'],success=result['success'],
                physical_frames=n,cursor=result.get('cursor'),actual_progress=result['last_alignment']['actual_progress'],
                wait_frames=len(waiting),nonzero_wait_correction_frames=sum(any(abs(v)>1e-10 for v in x['residual_si']) for x in waiting),
                completed_wait_bursts_progress_gt10=recovery_bursts,max_progress_within_wait_burst=maximum_wait_progress,
                retained_fill_percent=None if not result['success'] else 100*result['retained_fill_fraction'],
                checks=c,audit_passed=all(c.values())))
    checks['all_episode_audits']=all(x['audit_passed'] for x in episodes)
    ev=manifest['final_evaluations'];baseline=manifest['baseline']['retained_fill_fraction']
    improved=len(ev)==2 and all(x['success'] and x['source_unchanged'] and x['retained_fill_fraction']>baseline+1e-6 for x in ev)
    checks['honest_improvement']=improved==manifest['accepted_improvement']
    output=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
                episodes=episodes,check_count=len(checks)+sum(len(x['checks']) for x in episodes),
                final_evaluations=len(ev),accepted_improvement=improved,baseline_percent=baseline*100,
                training_steps=manifest['final_timesteps'],gradient_updates=manifest['final_gradient_updates'],
                final_replay_size=final.size(),new_physics_frames=0,neural_updates=0,
                frozen_checkpoint=selected,frozen_checkpoint_sha256=digest(selected))
    (root/'read-only-audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output))


if __name__=='__main__':main(Path(sys.argv[1]))
