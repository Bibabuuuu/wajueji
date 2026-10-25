#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "$0")/.." && pwd)"
handoff="$repo/runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089"
archive="/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo/src/v4/run_layered_sparse_resume_audit.py"
echo "Contact BC candidate collection: saved rear-lower prepoint, start paused"
echo "Hencky + bucket integral + moving bucket path guard; raw data reviewed separately"
exec bash "$repo/scripts/run_47_sparse_manual_contact_hencky_integral_local.sh" \
  "$handoff" "$@" --moving-bucket-sweep-candidate \
  --seed-auditor-source "$archive"
