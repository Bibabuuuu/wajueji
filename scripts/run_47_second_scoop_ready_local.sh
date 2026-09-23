#!/usr/bin/env bash
# D-139: actual terrain after the 85.19% first scoop, unload and safe return.
# This restores the audited prepoint directly; it does not replay transport.
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec bash "$repo/scripts/run_47_cycle_manual_local.sh" \
  "$repo/runs/layered-47-cycle-v1/next-prepoint-qualified-v2/prepoint.npz" \
  "$repo/runs/layered-47-cycle-v1/next-prepoint-qualified-audit-v1/audit.json" "$@"
