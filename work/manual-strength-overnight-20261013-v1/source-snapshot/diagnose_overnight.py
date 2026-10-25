import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import ROOT, environment_contract

out = ROOT/'work/manual-strength-overnight-20261013-v1'
out.mkdir(exist_ok=False)
budget_path = ROOT/'runs/manual-strength-overnight-20261013-budget.json'
start = datetime.fromisoformat('2026-10-13T15:47:35+00:00').timestamp()
deadline = start+28800
budget = dict(seconds=28800., purpose='training_and_evaluation', epoch_start=start,
              epoch_deadline=deadline, monotonic_deadline=time.monotonic()+deadline-time.time(),
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              started_utc='2026-10-13T15:47:35+00:00')
assert not budget_path.exists()
budget_path.write_text(json.dumps(budget, indent=2)+'\n')
torch.set_num_threads(1)
model = SAC.load(ROOT/'runs/manual-strength-sac-20261013-v1/sac-last.zip', device='cpu')
review = json.loads((ROOT/'runs/manual-strength-recovery-collection-20261013-v1/review.json').read_text())
diagnostics = []
for e in review['episodes']:
    with np.load(e['dataset']['path'], allow_pickle=False) as data:
        obs, actions = data['observations'], data['actions']
        predicted = model.predict(obs, deterministic=True)[0]
        sources = {str(k): int(v) for k, v in zip(*np.unique(data['policy_source'], return_counts=True))}
        parts = {}
        for name, mask in [('follow', obs[:, -1] == 0), ('wait', obs[:, -1] == 1)]:
            if mask.any():
                parts[name] = dict(count=int(mask.sum()),
                    demonstration_mean=actions[mask].mean(0).tolist(),
                    prediction_mean=predicted[mask].mean(0).tolist(),
                    prediction_abs_max=abs(predicted[mask]).max(0).tolist(),
                    mse=float(np.mean((predicted[mask]-actions[mask])**2)))
        diagnostics.append(dict(label=e['label'], policy_sources=sources, parts=parts))
report = dict(status='completed', environment_contract=environment_contract('error_strength_v1'),
    budget=budget, user_authorization='User granted eight hours and autonomous decisions while asleep; no further discussion required.',
    acceptance=dict(independent_same_scene_resets=3, manual_reference_frames=1170, hold_frames=30,
                    minimum_fill_percent=145.38132234025687, stretch_fill_percent=147.38132234025687,
                    evaluation_reserve_seconds=3600),
    prior_checkpoint_sha256=hashlib.sha256((ROOT/'runs/manual-strength-sac-20261013-v1/sac-last.zip').read_bytes()).hexdigest(),
    diagnostics=diagnostics, new_physics_frames=0, neural_updates=0)
(out/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
(out/'diagnosis-source.py').write_text(Path(__file__).read_text() if '__file__' in globals() and Path(__file__).is_file() else 'Executed via base64; source preserved in local scratch.\n')
print(json.dumps(report))
