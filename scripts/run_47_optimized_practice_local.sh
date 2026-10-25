#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bundle="/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
seed="$repo/runs/layered-47-frequency-probe-v1"
exec 9>/tmp/wajueji-local-47-optimized-practice.lock
flock -n 9 || { echo 'Another optimized practice task is running' >&2; exit 2; }
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || { echo "GPU0 compute process exists: $busy" >&2; exit 2; }
[[ -x "$python_bin" && -f "$seed/bundle-local-60hz-v3-resume.json" ]]
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle/excavator-resources/texture"
run_id="practice-$(date +%Y%m%d-%H%M%S)-$$"
out="$repo/runs/layered-47-optimized-practice-v1/$run_id"
mkdir -p "$(dirname -- "$out")"
echo "47 rocks / 10cm / 60Hz / optimized / starts paused; output=$out"
"$python_bin" -u "$repo/src/v4/run_layered_47_optimized_practice.py" \
  --repository-root "$bundle/repo" --config "$bundle/repo/configs/v4/layered-47-excavation-preflight-v1.json" \
  --output "$out" --candidate-report "$seed/bundle-local-60hz-v3.json" --resume-report "$seed/bundle-local-60hz-v3-resume.json" \
  --checkpoint-sha256 a3a78553efcf0cbb0018951b93640a95c2e7056558311b51472e255affa8f4f7 \
  --local-source-config "$repo/configs/v4/layered-47-interactive-proxy-benchmark-v1.json" \
  --manual-config "$repo/configs/v4/manual-input-v1.json" --display-config "$repo/configs/v4/layered-47-interactive-proxy-practice-v1.json" \
  --device cuda:0 --mujoco-include "$bundle/mujoco/include" --mujoco-library-dir "$bundle/mujoco/lib" \
  --mechanism-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  "$@" 2>&1 | tee "$out.log"
