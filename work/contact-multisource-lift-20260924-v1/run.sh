#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
case "${1:-}" in
 first) task_source=runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz; reset_args=();;
 second) task_source=runs/fast-contact-collection/20260924-114750-1451324/prepare/prepoint.npz; reset_args=(--training-reset-checkpoint);;
 *) echo 'Usage: run.sh first|second (physical budget must be authorized first)' >&2; exit 2;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
physics_python=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
"$physics_python" work/contact-multisource-lift-20260924-v1/run_candidate.py \
 --source "$task_source" --policy runs/contact-phase-bc-pilot-v1/pilot.pt \
 --policy-sha256 b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095 \
 --output "runs/contact-multisource-lift-20260924-v1/$1-candidate" --frames 3000 \
 --training-python /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \
 --coupling-hz 60 --capture-display "${reset_args[@]}" \
 --phase-contract "work/contact-multisource-lift-20260924-v1/$1-protocol.json"
"$physics_python" work/contact-multisource-lift-20260924-v1/audit_candidate.py "$1"
