"""Read-only preparation inventory; no CUDA context, training or physics."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
TRAIN=Path('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
policy=ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
source=ROOT/'runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz'
rpath=ROOT/'runs/contact-next-scoop-20260925-v1/candidate/report.json';r=json.loads(rpath.read_text())
rate=r['frames_completed']/r['wall_seconds']
code="from importlib import metadata; import json,sys; r={'python':sys.version,'executable':sys.executable};\nfor p in ('torch','numpy','gymnasium','stable-baselines3','tensorboard'):\n try:r[p]=metadata.version(p)\n except metadata.PackageNotFoundError:r[p]=None\nprint(json.dumps(r))"
deps=json.loads(subprocess.check_output([str(TRAIN),'-c',code],text=True))
result=dict(status='prepared_route_pending_no_training_started',physics_steps_added=0,training_steps_added=0,
 budget_seconds=28800,budget_includes=['startup_and_short_smoke','collection_and_learning','baseline_comparison','final_save'],
 measured_timing=dict(source=str(rpath),sha256=sha(rpath),frames=r['frames_completed'],wall_seconds=r['wall_seconds'],frames_per_wall_second=rate,
 theoretical_8h_frames=28800*rate,theoretical_6h_contact_rollouts=21600/r['wall_seconds'],
 caveat='extrapolation of current contact run, not RL benchmark; reset, gradients and evaluation cost additional time'),
 dependencies=deps,free_disk_bytes=shutil.disk_usage(ROOT).free,
 immutable_inputs={str(p):sha(p) for p in (policy,source,rpath)},
 environment_gaps=['current-scene training reset loop and causal controller-history reset','current-scene critic transitions/reward contract','residual or search action contract and bounds','budgeted checkpoint/resume and comparison runner','short update/save/reload/rollout check'],
 qualified=False,formal_volume_reward_ready=False,
 baseline_scope='current geometric feedback plus causal phase BC; source strict-zero prepoint qualification remains false',
 artifact_policy=dict(directory='runs/night-learning-20260925-v1',overwrite_baseline=False,
 save_every_wall_minutes=15,save_every_episode=True,save_fields=['actor_or_search_state','critics_if_used','optimizer_and_normalization','rng','replay_or_candidate_history','source_and_code_hashes','elapsed_budget','configuration'],
 best_and_latest_separate=True,full_replay='baseline and selected candidate, not every exploration frame',
 physical_reset='full qualified source state per episode, never reset between scoops inside continuous evaluation'),
 launch=False)
(W/'readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':pass
