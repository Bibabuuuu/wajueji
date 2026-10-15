#!/usr/bin/env bash
set -euo pipefail

repo=/home/qunwang/wajueji
python_bin=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
run_id="step-keyboard-d152-$(date +%Y%m%d-%H%M%S)-$$"
output="$repo/runs/roadshow-measurements-20261011/$run_id"

cd "$repo"
printf 'D152 step keyboard output: %s\n' "$output"
exec "$python_bin" src/v4/roadshow_d152_manual_collection.py \
  --frames 1800 --manual-input step_keyboard --step-frames 30 --output "$output"
