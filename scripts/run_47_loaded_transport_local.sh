#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bundle="/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
kinematics="/home/qunwang/codex-work/small-soil-20260910-v17-performance"
seed="$repo/runs/layered-47-frequency-probe-v1"
approach="$repo/runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089"
loaded="$repo/runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2"
truck_gate="$repo/runs/layered-47-truck-physics-gate-v1/truck-static-gate-20260923-v6/truck-physics-gate.json"

exec 9>/tmp/wajueji-local-47-loaded-transport.lock
flock -n 9 || {
  echo "Another loaded transport task is running" >&2
  exit 2
}
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || {
  echo "GPU0 compute process exists: $busy" >&2
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle/excavator-resources/texture"

run_id="loaded-transport-$(date +%Y%m%d-%H%M%S)-$$"
out="$repo/runs/layered-47-loaded-transport-v1/$run_id"
mkdir -p "$(dirname -- "$out")"

echo "S07 loaded transport/unload development pilot; NOT complete-chain evidence"
echo "Direct restore: $loaded/loaded-handoff.npz"
echo "Output: $out"
echo "Starts paused; Space pause/run; X/Esc/Back saves and finishes"
echo "Phases: 1 lift/transport, 2 align receiver, 3 unload/open, 4 settle/return"

"$python_bin" -u \
  "$repo/src/v4/run_layered_47_sparse_manual_contact_episode.py" \
  --loaded-transport \
  --repository-root "$bundle/repo" \
  --truck-repository-root "$repo" \
  --config "$bundle/repo/configs/v4/layered-47-excavation-preflight-v1.json" \
  --output "$out" \
  --candidate-report "$seed/bundle-local-60hz-v3.json" \
  --resume-report "$seed/bundle-local-60hz-v3-resume.json" \
  --local-source-config "$repo/configs/v4/layered-47-interactive-proxy-benchmark-v1.json" \
  --checkpoint "$loaded/loaded-handoff.npz" \
  --metadata "$loaded/loaded-handoff.npz.json" \
  --approach-execution "$approach/approach-execution.json" \
  --approach-report "$approach/report.json" \
  --restore-audit "$truck_gate" \
  --truck-config "$repo/configs/v4/layered-47-truck-physics-v1.json" \
  --manual-config "$repo/configs/v4/manual-input-v1.json" \
  --display-config "$repo/configs/v4/layered-47-interactive-proxy-practice-v1.json" \
  --device cuda:0 \
  --mujoco-include "$bundle/mujoco/include" \
  --mujoco-library-dir "$bundle/mujoco/lib" \
  --mechanism-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  --kinematics-python-dir "$kinematics" \
  --kinematics-model "$kinematics/scene/input/model/A.xml" \
  --kinematics-mesh-dir "$kinematics/scene/input/mesh" \
  --rigid-internal-hz 4800 \
  --bucket-integral-candidate \
  --hencky-update-candidate \
  "$@" \
  2>&1 | tee "$out.log"
