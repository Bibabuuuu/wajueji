#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 DETACHED_CYCLE_CHECKPOINT" >&2; exit 2; }
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
source_point="$(realpath -- "$1")"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9 || { echo "Another local collection/preparation is running" >&2; exit 2; }
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || { echo "GPU0 busy: $busy" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src/v4:$repo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
output="$repo/runs/fast-contact-collection/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$output"
echo "Fast collection branch: source must have observably detached load. Output: $output"
"$python_bin" -u "$repo/src/v4/prepare_47_fast_collection.py" --source "$source_point" \
  --output "$output/prepare" --execute > "$output/prepare.log" 2>&1
"$python_bin" -u "$repo/src/v4/audit_layered_47_cycle.py" --source "$output/prepare/prepoint.npz" \
  --output "$output/restore-audit" --frames 3 --require-no-bucket-contact \
  --training-reset-checkpoint > "$output/restore-audit.log" 2>&1
echo "READY: bash $repo/scripts/run_47_cycle_manual_local.sh $output/prepare/prepoint.npz $output/restore-audit/audit.json --training-reset-checkpoint"
