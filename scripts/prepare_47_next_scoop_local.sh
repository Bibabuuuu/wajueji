#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NORMAL_FINAL_CYCLE_NPZ" >&2
  exit 2
fi
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
source_point="$(realpath -- "$1")"
python_bin="/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python"
[[ -f "$source_point" && -f "$source_point.json" ]]
exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9 || { echo "Another local collection/preparation is running" >&2; exit 2; }
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || { echo "GPU0 busy: $busy" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
run_id="next-scoop-$(date +%Y%m%d-%H%M%S)-$$"
output="$repo/runs/layered-47-cycle-v1/$run_id"
mkdir -p "$output"
echo "Preparing the actual excavated scene; output: $output"
"$python_bin" -u "$repo/src/v4/audit_layered_47_cycle.py" --source "$source_point" --output "$output/source-audit" --frames 30 --truck 2>&1 | tee "$output/source-audit.log"
"$python_bin" -u "$repo/src/v4/run_layered_47_cycle_transport.py" --source "$output/source-audit/final.npz" --output "$output/transport" 2>&1 | tee "$output/transport.log"
"$python_bin" -u "$repo/src/v4/run_layered_47_cycle_prepoint.py" --source "$output/transport/return_high.npz" --output "$output/prepoint" 2>&1 | tee "$output/prepoint.log"
"$python_bin" -u "$repo/src/v4/audit_layered_47_cycle.py" --source "$output/prepoint/prepoint.npz" --output "$output/restore-audit" --frames 3 2>&1 | tee "$output/restore-audit.log"
echo "READY: bash $repo/scripts/run_47_cycle_manual_local.sh $output/prepoint/prepoint.npz $output/restore-audit/audit.json"
