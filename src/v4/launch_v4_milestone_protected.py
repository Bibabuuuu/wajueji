"""Launch one initially paused D-061 milestone protected evaluation."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from launch_dagger_takeover import (
    BACKEND, EXTRA_SITE, FILL_ROOT, POLICY_SITE, PYTHON,
    _running_acquisition, _session_environment,
)


SOURCE = Path(__file__).resolve().parent
POLICY = SOURCE / "results/v4-actor-full-v1/actor-bc-checkpoint.pt"
PREFLIGHT = SOURCE / "results/v4-milestone-preflight-v1/report.json"
MILESTONE_AUDIT = SOURCE / "results/v4-milestone-guard-audit-v2/report.json"
BLIND_REPORT = SOURCE / "results/v4-blind-evaluation-v1/report.json"
RUN_ROOT = Path(
    "/home/qunwang/codex-work/rl-four-axis-v4-milestone-runs-2026-09-18-v1")
EXPECTED_CHECKPOINT_SHA256 = (
    "b32a460e4c849e6a2d0986405d39e3397c9eccad309946cfeb6d803a4bd238d9")


def main():
    required = (
        PYTHON, SOURCE / "mpm_manual_demo.py", POLICY, PREFLIGHT,
        MILESTONE_AUDIT, BLIND_REPORT,
        BACKEND / "rl-fast-qualification-geometry-v2/report.json",
        BACKEND / "rl-fast-qualification-empty-v2/report.json",
        BACKEND / "rl-training-smoke-v1/acquisition-baseline-001/run/report.json",
        FILL_ROOT / "calibration-known-geometry-v1/report.json",
        FILL_ROOT / "held-out-phases-v1/report.json", EXTRA_SITE, POLICY_SITE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing_v4_milestone_inputs:" + ",".join(missing))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    milestone = json.loads(MILESTONE_AUDIT.read_text(encoding="utf-8"))
    blind = json.loads(BLIND_REPORT.read_text(encoding="utf-8"))
    if (preflight.get("status") != "passed"
            or preflight.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("v4_milestone_preflight_not_passed")
    if milestone.get("status") != "passed":
        raise RuntimeError("v4_milestone_fixed_log_audit_not_passed")
    if (blind.get("status") != "failed"
            or blind.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("v4_blind_failure_acknowledgement_missing")
    running = _running_acquisition()
    if running is not None:
        raise RuntimeError(f"acquisition_already_running:{running}")
    env = _session_environment()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    episode_id = "v4-milestone-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    output = RUN_ROOT / episode_id
    stdout_path = RUN_ROOT / f"{episode_id}.launch.log"
    command = [
        str(PYTHON), "-B", str(SOURCE / "mpm_manual_demo.py"),
        "--backend-root", str(BACKEND), "--output", str(output),
        "--geometry-report", str(
            BACKEND / "rl-fast-qualification-geometry-v2/report.json"),
        "--empty-report", str(
            BACKEND / "rl-fast-qualification-empty-v2/report.json"),
        "--fast-reference-report", str(
            BACKEND / "rl-training-smoke-v1/acquisition-baseline-001/run/report.json"),
        "--fill-root", str(FILL_ROOT),
        "--fill-calibration-report", str(
            FILL_ROOT / "calibration-known-geometry-v1/report.json"),
        "--fill-held-out-report", str(FILL_ROOT / "held-out-phases-v1/report.json"),
        "--extra-site-packages", str(EXTRA_SITE),
        "--policy-site-packages", str(POLICY_SITE),
        "--policy-checkpoint", str(POLICY),
        "--policy-observation-version", "v4",
        "--takeover-ttc-s", "2.0", "--takeover-clearance-m", "0.10",
        "--session-label", "V4 MILESTONE PROTECTED - D061 - NO TRAINING",
        "--episode-id", episode_id, "--max-wall", "3600", "--max-sim", "60",
    ]
    with stdout_path.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    manifest = {
        "schema": "excavation_v4_milestone_launch_v1",
        "decision": "D-061", "episode_id": episode_id, "pid": child.pid,
        "output": str(output), "stdout": str(stdout_path),
        "initially_paused": True,
        "runtime_phase_route": "causal_milestone_two_tick_episode_reset",
        "curl_reverse_full_action_pause": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "preflight_status": "passed", "training_started": False,
        "persistent_replay_write_performed": False, "command": command,
    }
    (SOURCE / "latest-v4-milestone-launch.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status_path = output / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") in ("paused_ready_for_user", "paused"):
                manifest["ready"] = True
                print(json.dumps(manifest, indent=2), flush=True)
                return 0
            if status.get("status") == "closed":
                raise RuntimeError("v4_milestone_closed_during_start:" + str(status))
        if child.poll() is not None:
            raise RuntimeError(f"v4_milestone_exited_during_start:{child.returncode}")
        time.sleep(0.5)
    manifest["ready"] = False
    manifest["reason"] = "still_initializing_after_30s"
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
