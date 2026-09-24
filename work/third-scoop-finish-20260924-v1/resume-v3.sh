#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
out=/home/qunwang/wajueji/runs/layered-47-cycle-v1/third-scoop-finish-20260924-094135-1368892
"$py" -u work/third-scoop-finish-20260924-v1/clearance_retreat.py --source "$out/source-audit/final.npz" --output "$out/retreat-boom-physical-v3" --execute 2>&1 | tee "$out/retreat-boom-physical-v3.log"
"$py" -u src/v4/run_layered_47_cycle_transport.py --source "$out/retreat-boom-physical-v3/cleared.npz" --output "$out/transport-after-retreat" 2>&1 | tee "$out/transport-after-retreat.log"
"$py" -u src/v4/audit_layered_47_cycle.py --source "$out/transport-after-retreat/return_high.npz" --output "$out/return-hold" --frames 30 --truck 2>&1 | tee "$out/return-hold.log"
echo 'THIRD_SCOOP_UNLOAD_RETURN_FINISHED'
