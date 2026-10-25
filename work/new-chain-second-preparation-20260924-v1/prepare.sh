#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
out=/home/qunwang/wajueji/runs/layered-47-cycle-v1/new-chain-second-20260924-v1
"$py" -u src/v4/run_layered_47_cycle_transport.py --source "$out/contact-exit-physical/cleared.npz" --output "$out/transport" 2>&1 | tee "$out/transport.log"
"$py" -u src/v4/run_layered_47_cycle_prepoint.py --source "$out/transport/return_high.npz" --output "$out/prepoint" 2>&1 | tee "$out/prepoint.log"
"$py" -u src/v4/qualify_layered_47_cycle_prepoint.py --source "$out/prepoint/prepoint.npz" --output "$out/qualified-prepoint" 2>&1 | tee "$out/qualified-prepoint.log"
"$py" -u src/v4/audit_layered_47_cycle.py --source "$out/qualified-prepoint/prepoint.npz" --output "$out/restore-audit" --frames 3 --require-no-bucket-contact 2>&1 | tee "$out/restore-audit.log"
echo NEW_CHAIN_SECOND_SCOOP_READY
