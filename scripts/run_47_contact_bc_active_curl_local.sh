#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
echo "D-136 ACTIVE CURL PILOT: error-dependent assist, original support integral"
echo "Small stop oscillation accepted for pilot; full-bucket cutting remains unverified"
exec bash "$repo/scripts/run_47_contact_bc_collection_local.sh" \
  "$@" --active-curl-candidate --cycle-save \
  --live-fill-config "$repo/configs/v4/live-bucket-fill-local-v1.json"
