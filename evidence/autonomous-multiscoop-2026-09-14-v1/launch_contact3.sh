#!/usr/bin/env bash
set -euo pipefail
cd /home/qunwang/codex-work/autonomous-multiscoop-2026-09-14-v1
test ! -e contact-3cycle-v1
test ! -e contact-3cycle-v1-console.log
nohup /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python -u run.py \
  --run contact-3cycle-v1 --geometry geometry-v5 --mode contact \
  --empty-report empty-3cycle-v1/report.json --cycles 3 \
  --max-wall 12000 --max-sim 540 >contact-3cycle-v1-console.log 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >contact-3cycle-v1.pid
printf 'started_pid=%s\n' "$run_pid"
