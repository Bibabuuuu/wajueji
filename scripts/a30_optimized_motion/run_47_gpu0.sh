#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repo="$bundle_dir/repo"
runtime_dir="/home/user01/newton-warp-offline-py312-a30-v2"
candidate="/home/user01/a30-layered-sparse-frequency-v3/outputs/sparse-47-cuda-0-20260922-174044-183267-60hz.json"
audit="/home/user01/a30-layered-sparse-frequency-v3/outputs/sparse-47-cuda-0-20260922-174044-183267-60hz-resume.json"
checkpoint_sha="003dafa7557cda74d22f53c5b0282079ae162e2b4cc889c3535662d388152589"
report=""
on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    python3 - "$code" "$report" <<'PY'
import json,sys
from pathlib import Path
print('COPY_RESULT_BEGIN')
print(json.dumps(dict(status='wrapper_or_optimized_motion_failed',runner_exit_code=int(sys.argv[1]),report=sys.argv[2],report_exists=bool(sys.argv[2]) and Path(sys.argv[2]).is_file()),indent=2))
print('COPY_RESULT_END')
PY
  fi
}
trap on_exit EXIT
exec 9>/tmp/wajueji-a30-sparse-cuda-0.lock
flock -n 9 || { echo 'GPU0 task lock held' >&2; exit 2; }
cd "$bundle_dir"
sha256sum --quiet -c MANIFEST.sha256
[[ -x "$runtime_dir/run_python.sh" && -f "$candidate" && -f "$audit" ]]
busy="$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$busy" ]] || { echo "GPU0 busy: $busy" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle_dir/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle_dir/excavator-resources/texture"
mkdir -p "$bundle_dir/outputs"
run_id="optimized-47-motion-cuda-0-$(date +%Y%m%d-%H%M%S)-$$"
report="$bundle_dir/outputs/$run_id.json"
echo "GPU0: 47 rocks, 10cm/60Hz, optimized feedforward, 270 frames; log=$bundle_dir/outputs/$run_id.log"
"$runtime_dir/run_python.sh" -u "$repo/src/v4/run_layered_47_optimized_motion.py" \
  --config "$repo/configs/v4/layered-47-excavation-preflight-v1.json" --repository-root "$repo" \
  --candidate-report "$candidate" --resume-report "$audit" --checkpoint-sha256 "$checkpoint_sha" \
  --device cuda:0 --cxx g++ --output "$report" \
  --mujoco-include "$bundle_dir/mujoco/include" --mujoco-library-dir "$bundle_dir/mujoco/lib" \
  --mechanism-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8" \
  --controller-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator" \
  2>&1 | tee "$bundle_dir/outputs/$run_id.log"
