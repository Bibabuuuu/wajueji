#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
case "${1:-}" in
 zero) frames=0; destination=zero;;
 candidate) frames=3000; destination=candidate;;
 *) echo 'Usage: run.sh zero|candidate'; exit 2;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
physics_python=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
"$physics_python" work/contact-next-scoop-20260925-v1/run_candidate.py \
 --source runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz \
 --policy runs/contact-phase-bc-pilot-v1/pilot.pt \
 --policy-sha256 b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095 \
 --output "runs/contact-next-scoop-20260925-v1/$destination" --frames "$frames" \
 --training-python /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \
 --coupling-hz 60 --capture-display \
 --phase-contract work/contact-next-scoop-20260925-v1/protocol.json
