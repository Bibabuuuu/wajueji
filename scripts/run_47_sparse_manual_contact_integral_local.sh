#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
echo "D-120 bucket integral PILOT: improved curl; reverse/stopped qualification pending"
exec bash "$repo/scripts/run_47_sparse_manual_contact_episode_local.sh" \
  "$@" --bucket-integral-candidate
