"""Read-only launch readiness. No simulator, CUDA initialization or subprocess trial."""
import json,shutil,sys
from pathlib import Path
from batch import ROOT,W,RUN,BASE_CONTACT,BASE_TAIL,read,sha,verify,baseline_readiness

def inspect():
    manifest=read(W/'manifest.json');blockers=[]
    try:verify(manifest);bindings=True
    except (ValueError,OSError) as error:bindings=False;blockers.append(str(error))
    baseline=read(BASE_TAIL/'report.json')
    blockers.extend(baseline_readiness())
    if manifest['status']!='prepared_not_launched':blockers.append('manifest_pending_final_baseline_binding')
    if (RUN/'batch.json').exists():blockers.append('immutable_batch_already_started')
    free=shutil.disk_usage(ROOT).free
    if free<55*1024**3:blockers.append('disk_reserve_55GiB')
    return dict(status='launch_blocked' if blockers else 'ready_for_smoke_and_bounded_search',
                blockers=blockers,bindings_match=bindings,free_bytes=free,
                source_sha256=manifest['source_sha256'],policy_sha256=manifest['policy_sha256'],
                maximum_wall_seconds=manifest['budget_seconds'],maximum_candidates=len(manifest['candidates']),
                physical_steps_added=0,neural_training_started=False,batch_started=(RUN/'batch.json').exists(),
                baseline_status=baseline['status'],baseline_frames=baseline['frames'],
                baseline_handoff=baseline.get('handoff'),baseline_handoff_checkpoint=baseline.get('handoff_checkpoint'),
                pending_checks=['completed_baseline_reviews_and_final_binding','physical120_frame_identity_smoke','candidate_and_tail_physical_validation',
                                'valid_retained_volume_scoring','selected_candidate_fixed_unload_confirmation'],
                scope='Only curl_load below95pct adjusted. Same saved prepoint per trial; not continuous scoops.')

if __name__=='__main__':
    result=inspect();print(json.dumps(result,indent=2,ensure_ascii=False))
    if len(sys.argv)>1:
        target=Path(sys.argv[1]);target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    sys.exit(2 if result['blockers'] else 0)
