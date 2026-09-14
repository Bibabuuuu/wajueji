#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/codex-work/autonomous-multiscoop-2026-09-14-v1
test ! -e empty-3cycle-v1
test ! -e empty-3cycle-v1-console.log
nohup /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python -u run.py \
  --run empty-3cycle-v1 --geometry geometry-v5 --mode empty --cycles 3 \
  --max-wall 2400 --max-sim 540 >empty-3cycle-v1-console.log 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >empty-3cycle-v1.pid
printf 'started_pid=%s\n' "$run_pid"
