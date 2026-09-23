#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 PREPOINT_NPZ RESTORE_AUDIT_JSON [manual options]" >&2
  exit 2
fi
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cycle_point="$(realpath -- "$1")"
cycle_audit="$(realpath -- "$2")"
shift 2
[[ -f "$cycle_point" && -f "$cycle_point.json" && -f "$cycle_audit" ]]
echo "Continuous manual scoop: exact saved prepoint, initially paused"
echo "Source: $cycle_point"
exec bash "$repo/scripts/run_47_contact_bc_active_curl_local.sh" \
  --checkpoint "$cycle_point" --metadata "$cycle_point.json" \
  --restore-audit "$cycle_audit" --cycle-checkpoint --cycle-save \
  --truck-config "$repo/configs/v4/layered-47-truck-physics-v1.json" \
  --truck-repository-root "$repo" "$@"
