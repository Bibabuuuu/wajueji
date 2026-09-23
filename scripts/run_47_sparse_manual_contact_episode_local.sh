#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 APPROACH_OUTPUT_DIRECTORY [--manual-input gamepad|keyboard] [--ui-smoke|--zero-probe-frames 3]" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bundle="/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
kinematics="/home/qunwang/codex-work/small-soil-20260910-v17-performance"
seed="$repo/runs/layered-47-frequency-probe-v1"
handoff_dir="$(realpath -- "$1")"
shift

exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9 || {
  echo "Another sparse manual-contact task is running" >&2
  exit 2
}
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || {
  echo "GPU0 compute process exists: $busy" >&2
  exit 2
}
[[ -x "$python_bin" ]]
[[ -f "$handoff_dir/approach-handoff.npz" ]]
[[ -f "$handoff_dir/approach-handoff.npz.json" ]]
[[ -f "$handoff_dir/approach-execution.json" ]]
[[ -f "$handoff_dir/report.json" ]]
[[ -f "$handoff_dir/independent-restore-audit.json" ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle/excavator-resources/texture"

run_id="manual-contact-$(date +%Y%m%d-%H%M%S)-$$"
out="$repo/runs/layered-47-sparse-manual-contact-v1/$run_id"
mkdir -p "$(dirname -- "$out")"

echo "Protected MANUAL single-scoop pilot; no scripted excavation; NOT training data"
display_source="$handoff_dir/approach-handoff.npz"
expect_source=false
for argument in "$@"; do
  if $expect_source; then display_source="$argument"; expect_source=false; fi
  if [[ "$argument" == --checkpoint ]]; then expect_source=true; fi
done
echo "Source: $display_source"
echo "Output: $out"
echo "Start paused; Space pause/run; neutral sticks to arm; X/Esc/Back finish"
echo "Phases: 1 penetrate, 2 curl/load, 3 lift/clear, 4 off-soil, R recovery"
echo "Events: 5 jam, 6 spill, 7 load detached, 8 clear"
echo "Budget: 30 simulation seconds / 900 wall seconds; final normal stop holds zero for 30 frames"
echo "Velocity references are diagnostic warnings, not contact hard stops"
echo "A true hard fault freezes physics and keeps the viewer open until X/Esc/Back"

"$python_bin" -u \
  "$repo/src/v4/run_layered_47_sparse_manual_contact_episode.py" \
  --repository-root "$bundle/repo" \
  --config "$bundle/repo/configs/v4/layered-47-excavation-preflight-v1.json" \
  --output "$out" \
  --candidate-report "$seed/bundle-local-60hz-v3.json" \
  --resume-report "$seed/bundle-local-60hz-v3-resume.json" \
  --local-source-config "$repo/configs/v4/layered-47-interactive-proxy-benchmark-v1.json" \
  --checkpoint "$handoff_dir/approach-handoff.npz" \
  --metadata "$handoff_dir/approach-handoff.npz.json" \
  --approach-execution "$handoff_dir/approach-execution.json" \
  --approach-report "$handoff_dir/report.json" \
  --restore-audit "$handoff_dir/independent-restore-audit.json" \
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
  "$@" \
  2>&1 | tee "$out.log"
