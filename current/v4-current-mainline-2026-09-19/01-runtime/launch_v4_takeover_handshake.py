"""Launch one initially paused D-063 exact-state takeover validation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
MILESTONE_PREFLIGHT = SOURCE / "results/v4-milestone-preflight-v1/report.json"
MILESTONE_AUDIT = SOURCE / "results/v4-milestone-guard-audit-v2/report.json"
BLIND_REPORT = SOURCE / "results/v4-blind-evaluation-v1/report.json"
HANDSHAKE_PREFLIGHT = Path(
    "/home/qunwang/codex-work/v4-takeover-handshake-preflight-2026-09-18-v3/report.json")
RUN_ROOT = Path(
    "/home/qunwang/codex-work/rl-four-axis-v4-takeover-handshake-runs-2026-09-18-v1")
EXPECTED_CHECKPOINT_SHA256 = (
    "b32a460e4c849e6a2d0986405d39e3397c9eccad309946cfeb6d803a4bd238d9")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--observation-audit", type=Path)
    args = parser.parse_args()
    required = (
        PYTHON, SOURCE / "mpm_manual_demo.py", SOURCE / "dagger_supervisor.py",
        POLICY, MILESTONE_PREFLIGHT, MILESTONE_AUDIT, BLIND_REPORT,
        HANDSHAKE_PREFLIGHT,
        BACKEND / "rl-fast-qualification-geometry-v2/report.json",
        BACKEND / "rl-fast-qualification-empty-v2/report.json",
        BACKEND / "rl-training-smoke-v1/acquisition-baseline-001/run/report.json",
        FILL_ROOT / "calibration-known-geometry-v1/report.json",
        FILL_ROOT / "held-out-phases-v1/report.json", EXTRA_SITE, POLICY_SITE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing_v4_takeover_handshake_inputs:" + ",".join(missing))

    milestone_preflight = json.loads(
        MILESTONE_PREFLIGHT.read_text(encoding="utf-8"))
    milestone = json.loads(MILESTONE_AUDIT.read_text(encoding="utf-8"))
    blind = json.loads(BLIND_REPORT.read_text(encoding="utf-8"))
    handshake = json.loads(HANDSHAKE_PREFLIGHT.read_text(encoding="utf-8"))
    if (milestone_preflight.get("status") != "passed"
            or milestone_preflight.get("checkpoint_sha256")
            != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("v4_milestone_preflight_not_passed")
    if milestone.get("status") != "passed":
        raise RuntimeError("v4_milestone_fixed_log_audit_not_passed")
    if (blind.get("status") != "failed"
            or blind.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256):
        raise RuntimeError("v4_blind_failure_acknowledgement_missing")
    expected_sources = handshake.get("source_sha256", {})
    if (handshake.get("status") != "passed"
            or handshake.get("scope", {}).get("physics_steps") != 0
            or handshake.get("scope", {}).get("gui_started") is not False
            or expected_sources.get("mpm_manual_demo.py")
            != sha256(SOURCE / "mpm_manual_demo.py")
            or expected_sources.get("dagger_supervisor.py")
            != sha256(SOURCE / "dagger_supervisor.py")):
        raise RuntimeError("v4_takeover_handshake_preflight_not_passed")
    if sha256(POLICY) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("v4_actor_checkpoint_identity_mismatch")

    policy_path, policy_hash = POLICY, EXPECTED_CHECKPOINT_SHA256
    run_root = RUN_ROOT
    if args.observation_audit:
        audit = json.loads(args.observation_audit.read_text(encoding="utf-8"))
        if (audit.get("status") != "passed_observation_only"
                or audit.get("decision") not in ("D-066", "D-071")
                or not audit.get("checks") or not all(audit["checks"].values())
                or audit.get("offline_training_status") != ("offline_checks_failed" if audit.get("decision") == "D-071" else "failed")):
            raise RuntimeError("observation_audit_not_passed")
        for path, expected in audit["source_sha256"].items():
            if sha256(Path(path)) != expected:
                raise RuntimeError("observation_audited_source_changed:" + path)
        for path in SOURCE.glob("*.py"):
            if audit["source_sha256"].get(str(path)) != sha256(path):
                raise RuntimeError("observation_runtime_source_not_audited:" + str(path))
        policy_path = Path(audit["checkpoint"])
        policy_hash = audit["checkpoint_sha256"]
        if sha256(policy_path) != policy_hash:
            raise RuntimeError("observation_checkpoint_changed")
        run_root = Path("/home/qunwang/codex-work/v4-observation-runs-2026-09-18-v1")

    if args.preflight_only:
        print(json.dumps({
            "schema": "excavation_v4_takeover_handshake_launch_preflight_v1",
            "status": "passed",
            "decision": "D-063",
            "source": str(SOURCE),
            "checkpoint_sha256": policy_hash,
            "observation_only": bool(args.observation_audit),
            "handshake_preflight_sha256": sha256(HANDSHAKE_PREFLIGHT),
            "physics_steps": 0,
            "gui_started": False,
            "training_started": False,
        }, indent=2), flush=True)
        return 0

    running = _running_acquisition()
    # The helper checks its own directory; also reject other isolated instances.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if any(Path(arg.decode()).name == 'mpm_manual_demo.py' for arg in
                   (proc / 'cmdline').read_bytes().split(b'\0') if arg):
                running = int(proc.name)
                break
        except (OSError, UnicodeDecodeError):
            continue
    if running is not None:
        raise RuntimeError(f"acquisition_already_running:{running}")
    env = _session_environment()
    run_root.mkdir(parents=True, exist_ok=True)
    episode_id = ("v4-observation-" if args.observation_audit else "v4-handshake-") + datetime.now().strftime("%Y%m%d-%H%M%S")
    output = run_root / episode_id
    stdout_path = run_root / f"{episode_id}.launch.log"
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
        "--fill-held-out-report", str(
            FILL_ROOT / "held-out-phases-v1/report.json"),
        "--extra-site-packages", str(EXTRA_SITE),
        "--policy-site-packages", str(POLICY_SITE),
        "--policy-checkpoint", str(policy_path),
        "--policy-observation-version", "v4",
        "--takeover-ttc-s", "2.0", "--takeover-clearance-m", "0.10",
        "--session-label", (("V4 D070 - OBSERVATION ONLY D071 - NO TRAINING" if audit['decision'] == 'D-071' else "V4 D065 STEP500 - OBSERVATION ONLY D066 - NO TRAINING") if args.observation_audit else "V4 EXACT TAKEOVER HANDSHAKE - D063 - NO TRAINING"),
        "--episode-id", episode_id, "--max-wall", "3600", "--max-sim", "60",
    ]
    with stdout_path.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    manifest = {
        "schema": "excavation_v4_takeover_handshake_launch_v1",
        "decision": audit['decision'] if args.observation_audit else "D-063", "episode_id": episode_id, "pid": child.pid,
        "output": str(output), "stdout": str(stdout_path),
        "initially_paused": True,
        "operator_enter_only_arms_takeover": True,
        "physics_frozen_until_first_nonzero_human_action": True,
        "first_nonzero_action_recorded_as_direct_expert_label": True,
        "runtime_phase_route": audit.get('runtime_phase_route', 'causal_milestone_two_tick_episode_reset') if args.observation_audit else 'causal_milestone_two_tick_episode_reset',
        "curl_reverse_full_action_pause": True,
        "checkpoint_sha256": policy_hash,
        "observation_only": bool(args.observation_audit),
        "observation_audit": str(args.observation_audit) if args.observation_audit else None,
        "handshake_preflight_sha256": sha256(HANDSHAKE_PREFLIGHT),
        "training_started": False,
        "persistent_replay_write_performed": False,
        "command": command,
    }
    for path in (SOURCE / "latest-v4-takeover-handshake-launch.json",
                 run_root / "latest-launch.json"):
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
                raise RuntimeError(
                    "v4_takeover_handshake_closed_during_start:" + str(status))
        if child.poll() is not None:
            raise RuntimeError(
                f"v4_takeover_handshake_exited_during_start:{child.returncode}")
        time.sleep(0.5)
    manifest["ready"] = False
    manifest["reason"] = "still_initializing_after_30s"
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
