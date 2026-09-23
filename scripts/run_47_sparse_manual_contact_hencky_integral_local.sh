#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
echo "D-128 Hencky soil + D-120 bucket integral GUARDED PILOT"
echo "Recorded trajectory passed; new manual single scoop is not yet qualified"
exec bash "$repo/scripts/run_47_sparse_manual_contact_episode_local.sh" \
  "$@" --bucket-integral-candidate --hencky-update-candidate
