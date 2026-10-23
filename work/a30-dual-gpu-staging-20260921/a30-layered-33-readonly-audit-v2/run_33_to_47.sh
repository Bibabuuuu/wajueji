#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
runtime_dir="${HOME}/newton-warp-offline-py312-a30-v2"
repo="$bundle_dir/repo"
device="${A30_DEVICE:-cuda:0}"
device_tag="${device//:/-}"
source_checkpoint="${LAYERED_INPUT_CHECKPOINT:-/home/user01/a30-layered-checkpoint-v2/checkpoints/three-layer-checkpoint-cuda-0-20260921-170946-35334.npz}"
source_metadata="${LAYERED_INPUT_CHECKPOINT_METADATA:-${source_checkpoint}.json}"
expected_source_sha="d30db1bd42c064934badc21716d0ac6f1db80e24d009c71d27a4751b77f3f545"
run_id="four-layer-47-checkpoint-${device_tag}-$(date +%Y%m%d-%H%M%S)-$$"
report="$bundle_dir/outputs/$run_id.json"
checkpoint="$bundle_dir/checkpoints/$run_id.npz"
log="$bundle_dir/outputs/$run_id.log"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle_dir/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle_dir/excavator-resources/texture"

if [[ ! -x "$runtime_dir/run_python.sh" ]]; then
  echo "ERROR: missing $runtime_dir/run_python.sh" >&2
  exit 1
fi
if [[ ! -f "$source_checkpoint" || ! -f "$source_metadata" ]]; then
  echo "ERROR: source checkpoint or metadata missing" >&2
  exit 1
fi
printf '%s  %s\n' "$expected_source_sha" "$source_checkpoint" | sha256sum -c -

cd "$bundle_dir"
sha256sum -c MANIFEST.sha256
echo '=== GPU0 33 -> 47 FOUR-LAYER MAINLINE ==='
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader

PYTHONPATH="$repo/src/v4" "$runtime_dir/run_python.sh" -u \
  "$bundle_dir/preflight_checkpoint_bridge.py" \
  --config "$repo/configs/v4/layered-mound-fortyseven-rock-four-layer-v1.json" \
  --repository-root "$repo" --checkpoint "$source_checkpoint" \
  --checkpoint-metadata "$source_metadata" --output-rock-count 47 --cxx g++ \
  --mujoco-include "$bundle_dir/mujoco/include" \
  --mujoco-library-dir "$bundle_dir/mujoco/lib" \
  --mechanism-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator"

set +e
PYTHONPATH="$repo/src/v4" "$runtime_dir/run_python.sh" -u \
  "$repo/src/v4/run_layered_mound_fortyseven_rock_four_layer.py" \
  --config "$repo/configs/v4/layered-mound-fortyseven-rock-four-layer-v1.json" \
  --repository-root "$repo" --output "$report" \
  --output-checkpoint "$checkpoint" --checkpoint "$source_checkpoint" \
  --checkpoint-metadata "$source_metadata" --device "$device" --cxx g++ \
  --mujoco-include "$bundle_dir/mujoco/include" \
  --mujoco-library-dir "$bundle_dir/mujoco/lib" \
  --mechanism-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  2>&1 | tee "$log"
runner_code=${PIPESTATUS[0]}
set -e

"$runtime_dir/run_python.sh" "$bundle_dir/summarize_47.py" \
  "$report" "$checkpoint" "$source_checkpoint" "$runner_code"
exit "$runner_code"
