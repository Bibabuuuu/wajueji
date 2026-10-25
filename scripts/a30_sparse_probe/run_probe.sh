#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
rocks="${1:?Use 47 or 65}"
case "$rocks" in
  47) device=cuda:0; source_checkpoint="${LAYERED_47_INPUT_CHECKPOINT:-/home/user01/a30-layered-33-to-47-v5/checkpoints/four-layer-47-checkpoint-cuda-0-20260921-200447-53738.npz}" ;;
  65) device=cuda:1; source_checkpoint="${LAYERED_65_INPUT_CHECKPOINT:-/home/user01/a30-layered-47-to-65-v5/checkpoints/five-layer-65-checkpoint-cuda-0-20260921-232339-74114.npz}" ;;
  *) echo 'Expected 47 or 65' >&2; exit 2 ;;
esac
runtime_dir="${HOME}/newton-warp-offline-py312-a30-v2"
repo="$bundle_dir/repo"
report=""
on_exit() {
  exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    python3 - "$exit_code" "$report" "$source_checkpoint" <<'PY'
import json, sys
print('COPY_RESULT_BEGIN')
print(json.dumps(dict(status='wrapper_or_probe_failed', runner_exit_code=int(sys.argv[1]), report=sys.argv[2], source_checkpoint=sys.argv[3]), indent=2))
print('COPY_RESULT_END')
PY
  fi
}
trap on_exit EXIT
mkdir -p "$bundle_dir/outputs"
exec 9>"/tmp/wajueji-a30-sparse-${device//:/-}.lock"
flock -n 9 || { echo 'Another probe owns this GPU lock' >&2; exit 2; }
[[ -x "$runtime_dir/run_python.sh" && -f "$source_checkpoint" && -f "${source_checkpoint}.json" ]]
cd "$bundle_dir"
sha256sum -c MANIFEST.sha256
busy="$(nvidia-smi -i "${device#cuda:}" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$busy" ]]; then
  echo "GPU ${device} has compute processes; do not interrupt them: $busy" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src/v4:$repo"
export EXCAVATOR_MUJOCO_MESH_DIR="$bundle_dir/excavator-resources/mesh"
export EXCAVATOR_MUJOCO_TEXTURE_DIR="$bundle_dir/excavator-resources/texture"
config="$repo/configs/v4/layered-${rocks}-excavation-preflight-v1.json"
common=(--config "$config" --repository-root "$repo" --cxx g++
  --mujoco-include "$bundle_dir/mujoco/include" --mujoco-library-dir "$bundle_dir/mujoco/lib"
  --mechanism-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8"
  --controller-header-dir "$repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator")
"$runtime_dir/run_python.sh" "$bundle_dir/preflight_checkpoint_bridge.py" "${common[@]}" \
  --checkpoint "$source_checkpoint" --checkpoint-metadata "${source_checkpoint}.json" --output-rock-count "$rocks"
run_id="sparse-${rocks}-${device//:/-}-$(date +%Y%m%d-%H%M%S)-$$"
seed_report="${LAYERED_SPARSE_120_REPORT:-}"
seed_audit="${LAYERED_SPARSE_120_AUDIT:-}"
if [[ -n "$seed_report" || -n "$seed_audit" ]]; then
  [[ -f "$seed_report" && -f "$seed_audit" ]] || { echo 'Both explicit 120Hz candidate and audit reports are required' >&2; exit 2; }
fi
for hz in 120 60; do
  if [[ "$hz" == 120 && -n "$seed_report" ]]; then
    echo "Reusing explicit 120Hz checkpoint; binding will be verified before any 60Hz step: $seed_report"
    continue
  fi
  seed_args=()
  if [[ "$hz" == 60 ]]; then
    seed_args=(--seed-candidate-report "$seed_report" --seed-resume-report "$seed_audit")
  fi
  report="$bundle_dir/outputs/${run_id}-${hz}hz.json"
  "$runtime_dir/run_python.sh" -u "$repo/src/v4/run_layered_sparse_a30_probe.py" \
    "${common[@]}" --checkpoint "$source_checkpoint" --checkpoint-metadata "${source_checkpoint}.json" \
    --rocks "$rocks" --solver sparse10 --hz "$hz" --device "$device" --output "$report" "${seed_args[@]}" \
    2>&1 | tee "$bundle_dir/outputs/${run_id}-${hz}hz.log"
  candidate_report="$report"
  report="$bundle_dir/outputs/${run_id}-${hz}hz-resume.json"
  "$runtime_dir/run_python.sh" -u "$repo/src/v4/run_layered_sparse_resume_audit.py" \
    "${common[@]}" --candidate-report "$candidate_report" --rocks "$rocks" --hz "$hz" \
    --device "$device" --output "$report" 2>&1 | tee "$bundle_dir/outputs/${run_id}-${hz}hz-resume.log"
  if [[ "$hz" == 120 ]]; then
    seed_report="$candidate_report"
    seed_audit="$report"
  fi
done
"$runtime_dir/run_python.sh" "$bundle_dir/summarize_pair.py" \
  "$seed_report" "$bundle_dir/outputs/${run_id}-60hz.json" \
  "$seed_audit" "$bundle_dir/outputs/${run_id}-60hz-resume.json"
