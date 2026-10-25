#!/usr/bin/env bash
set -euo pipefail

repo=/home/qunwang/wajueji
bundle=/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1
python=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
stamp=$(date +%Y%m%d-%H%M%S)-$$
output="$repo/runs/layered-47-sparse-manual-contact-v1/manual-contact-$stamp"
printf '%s\n' "$output" > "$repo/work/manual-three-scoop-showcase-20260925-v1/attempt-04-run.txt"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle/excavator-resources/texture"

exec "$python" "$repo/src/v4/run_layered_47_sparse_manual_contact_episode.py" \
  --config "$bundle/repo/configs/v4/layered-47-excavation-preflight-v1.json" \
  --repository-root "$bundle/repo" \
  --output "$output" \
  --candidate-report "$bundle/repo/reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json" \
  --resume-report "$repo/runs/layered-47-frequency-probe-v1/bundle-local-60hz-v3-resume.json" \
  --local-source-config "$repo/configs/v4/layered-47-interactive-proxy-benchmark-v1.json" \
  --checkpoint "$repo/runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz" \
  --metadata "$repo/runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz.json" \
  --approach-execution "$repo/runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-execution.json" \
  --approach-report "$repo/runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json" \
  --restore-audit "$repo/work/manual-three-scoop-showcase-20260925-v1/first-audit-02/audit.json" \
  --manual-config "$repo/configs/v4/manual-input-v1.json" \
  --display-config "$repo/configs/v4/layered-47-interactive-proxy-practice-v1.json" \
  --device cuda:0 \
  --mujoco-include "$bundle/mujoco/include" \
  --mujoco-library-dir "$bundle/mujoco/lib" \
  --mechanism-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  --kinematics-python-dir /home/qunwang/codex-work/small-soil-20260910-v17-performance \
  --kinematics-model /home/qunwang/codex-work/small-soil-20260910-v17-performance/scene/input/model/A.xml \
  --kinematics-mesh-dir /home/qunwang/codex-work/small-soil-20260910-v17-performance/scene/input/mesh \
  --bucket-integral-candidate \
  --active-curl-candidate \
  --hencky-update-candidate \
  --freeze-passive-jp \
  --moving-bucket-sweep-candidate \
  --cycle-checkpoint \
  --cycle-save \
  --showcase-continuous \
  --truck-config "$repo/configs/v4/layered-47-truck-physics-v1.json" \
  --truck-repository-root "$repo" \
  --live-fill-config "$repo/configs/v4/live-bucket-fill-local-v1.json" \
  --display-cache-frames 12000
