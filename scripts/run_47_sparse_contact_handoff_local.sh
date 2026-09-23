#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 APPROACH_OUTPUT_DIRECTORY" >&2
  exit 2
fi

repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
bundle="/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
kinematics="/home/qunwang/codex-work/small-soil-20260910-v17-performance"
seed="$repo/runs/layered-47-frequency-probe-v1"
handoff_dir="$(realpath -- "$1")"

exec 9>/tmp/wajueji-local-47-sparse-contact.lock
flock -n 9 || {
  echo "Another sparse contact task is running" >&2
  exit 2
}
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || {
  echo "GPU0 compute process exists: $busy" >&2
  exit 2
}
[[ -x "$python_bin" ]]
[[ -f "$handoff_dir/approach-handoff.npz" ]]
[[ -f "$handoff_dir/independent-restore-audit.json" ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle/excavator-resources/texture"

run_id="contact-$(date +%Y%m%d-%H%M%S)-$$"
out="$repo/runs/layered-47-sparse-contact-handoff-v1/$run_id"
mkdir -p "$(dirname -- "$out")"
echo "47 rocks / sparse 10cm / MPM 60Hz / rigid 4800Hz / first-contact-only; output=$out"

"$python_bin" -u "$repo/src/v4/run_layered_47_sparse_contact_handoff.py" \
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
  --device cuda:0 \
  --mujoco-include "$bundle/mujoco/include" \
  --mujoco-library-dir "$bundle/mujoco/lib" \
  --mechanism-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$bundle/repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  --kinematics-python-dir "$kinematics" \
  --kinematics-model "$kinematics/scene/input/model/A.xml" \
  --kinematics-mesh-dir "$kinematics/scene/input/mesh" \
  --rigid-internal-hz 4800 \
  2>&1 | tee "$out.log"
