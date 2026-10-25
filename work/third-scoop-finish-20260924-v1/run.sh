#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/wajueji
exec 9>/tmp/wajueji-local-47-sparse-manual-contact.lock
flock -n 9
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || { echo "GPU0 busy: $busy"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
source_point=/home/qunwang/wajueji/runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-093202-1362929/final-001264.cycle.npz
"$py" - "$source_point" <<'PY'
import sys
from layered_47_cycle_checkpoint import load
_,m=load(sys.argv[1])
assert m['stage']=='loaded_manual_endpoint' and m['cycle_index']==3
assert m['source_identity']['sha256']=='5b9c4e7a5520a0c4beab9cc124e7f4a54e24af4b8953fea6f6bdfb0b1e3077b0'
PY
out="/home/qunwang/wajueji/runs/layered-47-cycle-v1/third-scoop-finish-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$out"
printf '%s\n' "$out" > work/third-scoop-finish-20260924-v1/run-directory.txt
echo "RUN: $out"
"$py" -u src/v4/audit_layered_47_cycle.py --source "$source_point" --output "$out/source-audit" --frames 30 --truck 2>&1 | tee "$out/source-audit.log"
"$py" -u src/v4/run_layered_47_cycle_transport.py --source "$out/source-audit/final.npz" --output "$out/transport" 2>&1 | tee "$out/transport.log"
"$py" -u src/v4/audit_layered_47_cycle.py --source "$out/transport/return_high.npz" --output "$out/return-hold" --frames 30 --truck 2>&1 | tee "$out/return-hold.log"
echo "THIRD_SCOOP_UNLOAD_RETURN_FINISHED: $out"
