"""Launch one paused actor-to-human DAgger takeover episode."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


SOURCE = Path(__file__).resolve().parent
PYTHON = Path("/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python")
BACKEND = Path("/home/qunwang/codex-work/autonomous-multiscoop-speed-opt-2026-09-14-v1")
FILL_ROOT = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1")
RUN_ROOT = Path("/home/qunwang/codex-work/rl-four-axis-dagger-runs-2026-09-17-v1")
EXTRA_SITE = Path("/home/qunwang/codex-work/soil-separation-2026-09-11/.venv/lib/python3.12/site-packages")
POLICY_SITE = Path("/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/lib/python3.12/site-packages")
POLICY = SOURCE / "results/independent-manual-bc-ft-500-lr0.1/actor-bc-checkpoint.pt"


def _session_environment() -> dict[str, str]:
    env = os.environ.copy()
    required = ("DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR")
    if all(env.get(name) for name in required):
        return env
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() != "gnome-shell":
                continue
            if proc.stat().st_uid != os.getuid():
                continue
            values = {}
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    name, value = item.split(b"=", 1)
                    values[name.decode()] = value.decode()
            for name in required:
                if values.get(name):
                    env[name] = values[name]
            if all(env.get(name) for name in required):
                return env
        except (OSError, UnicodeDecodeError):
            continue
    raise RuntimeError("logged_in_gnome_session_environment_not_found")


def _running_acquisition() -> int | None:
    marker = str(SOURCE / "mpm_manual_demo.py").encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if marker in (proc / "cmdline").read_bytes():
                return int(proc.name)
        except (OSError, PermissionError):
            continue
    return None


def main() -> int:
    required_paths = (
        PYTHON, SOURCE / "mpm_manual_demo.py", POLICY,
        BACKEND / "rl-fast-qualification-geometry-v2/report.json",
        BACKEND / "rl-fast-qualification-empty-v2/report.json",
        BACKEND / "rl-training-smoke-v1/acquisition-baseline-001/run/report.json",
        FILL_ROOT / "calibration-known-geometry-v1/report.json",
        FILL_ROOT / "held-out-phases-v1/report.json", EXTRA_SITE, POLICY_SITE,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing_dagger_inputs:" + ",".join(missing))
    running = _running_acquisition()
    if running is not None:
        raise RuntimeError(f"acquisition_already_running:{running}")

    env = _session_environment()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    episode_id = "dagger-takeover-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    output = RUN_ROOT / episode_id
    stdout_path = RUN_ROOT / f"{episode_id}.launch.log"
    command = [
        str(PYTHON), "-B", str(SOURCE / "mpm_manual_demo.py"),
        "--backend-root", str(BACKEND),
        "--output", str(output),
        "--geometry-report", str(BACKEND / "rl-fast-qualification-geometry-v2/report.json"),
        "--empty-report", str(BACKEND / "rl-fast-qualification-empty-v2/report.json"),
        "--fast-reference-report", str(
            BACKEND / "rl-training-smoke-v1/acquisition-baseline-001/run/report.json"),
        "--fill-root", str(FILL_ROOT),
        "--fill-calibration-report", str(
            FILL_ROOT / "calibration-known-geometry-v1/report.json"),
        "--fill-held-out-report", str(FILL_ROOT / "held-out-phases-v1/report.json"),
        "--extra-site-packages", str(EXTRA_SITE),
        "--policy-site-packages", str(POLICY_SITE),
        "--policy-checkpoint", str(POLICY),
        "--policy-seed", "20260917",
        "--takeover-ttc-s", "2.0",
        "--takeover-clearance-m", "0.10",
        "--episode-id", episode_id,
        "--max-wall", "3600",
        "--max-sim", "60",
    ]
    with stdout_path.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    manifest = {
        "schema": "excavation_dagger_takeover_launch_v1",
        "episode_id": episode_id,
        "pid": child.pid,
        "output": str(output),
        "stdout": str(stdout_path),
        "initially_paused": True,
        "actor_runs_until_closed_bucket_or_predictive_safety_pause": True,
        "operator_accept_key": "ENTER",
        "training_started": False,
        "persistent_replay_write_performed": False,
        "command": command,
    }
    for path in (SOURCE / "latest-dagger-launch.json",
                 RUN_ROOT / "latest-launch.json"):
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
                raise RuntimeError("dagger_closed_during_start:" + str(status))
        if child.poll() is not None:
            raise RuntimeError(f"dagger_exited_during_start:{child.returncode}")
        time.sleep(0.5)
    manifest["ready"] = False
    manifest["reason"] = "still_initializing_after_30s"
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
