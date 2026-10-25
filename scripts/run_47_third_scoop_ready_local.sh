#!/usr/bin/env bash
# Actual second scoop terrain after unload and return; restore directly paused.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
prepared="$repo/runs/layered-47-cycle-v1/next-scoop-20260924-090702-1347301"
exec bash "$repo/scripts/run_47_cycle_manual_local.sh" \
  "$prepared/qualified-prepoint/prepoint.npz" \
  "$prepared/restore-audit/audit.json" "$@"
