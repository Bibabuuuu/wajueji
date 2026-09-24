#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python src/v4/run_contact_geometric_probe.py \
 --source runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz \
 --training-reset-checkpoint --policy runs/contact-phase-bc-pilot-v1/pilot.pt \
 --policy-sha256 b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095 \
 --training-python /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \
 --phase-contract work/contact-geometric-feedback-20260924-v8/protocol.json "$@"
