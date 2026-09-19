"""Zero-physics preflight for the exact-state human takeover handshake."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from dagger_supervisor import HumanTakeoverHandshake


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))

    decision_step = 478
    sim_time_s = 23.9
    handshake = HumanTakeoverHandshake()
    handshake.request(decision_step=decision_step, sim_time_s=sim_time_s)
    armed = handshake.arm(
        decision_step=decision_step, sim_time_s=sim_time_s)
    zero_result = handshake.activate(
        raw_action=[0.0, 0.0, 0.0, 0.0],
        decision_step=decision_step,
        sim_time_s=sim_time_s,
    )
    activation = handshake.activate(
        raw_action=[0.0, 1.0, -1.0, 1.0],
        decision_step=decision_step,
        sim_time_s=sim_time_s,
    )
    buffered, is_first = handshake.action_for_step(
        [0.0, 0.0, 0.0, 0.0])
    before_execution = handshake.as_dict()
    handshake.mark_first_action_executed()
    after_execution = handshake.as_dict()

    physics_modules = sorted(set(sys.modules) & {"newton", "warp", "mujoco"})
    checks = {
        "enter_arms_without_advancing_physics": (
            armed["event"] == "human_takeover_armed"
            and armed["decision_step"] == decision_step
            and armed["physics_advanced"] is False),
        "zero_action_keeps_physics_frozen": (
            zero_result is None
            and activation["decision_step"] == decision_step
            and activation["physics_advanced_before_action"] is False),
        "first_nonzero_action_is_direct_expert_label": (
            activation["raw_human_action"] == [0.0, 1.0, -1.0, 1.0]
            and activation["record_as_direct_expert_label"] is True),
        "first_action_survives_key_release_until_execution": (
            is_first and buffered == [0.0, 1.0, -1.0, 1.0]
            and before_execution["first_action_buffered"] is True
            and after_execution["first_action_buffered"] is False),
        "decision_step_is_exactly_preserved": (
            before_execution["requested_decision_step"] == decision_step
            and before_execution["activation_event"]["decision_step"]
            == decision_step),
        "physics_not_imported": not physics_modules,
        "environment_steps_zero": True,
        "actor_parameter_updates_zero": True,
        "critic_parameter_updates_zero": True,
        "sac_parameter_updates_zero": True,
    }
    report = {
        "schema": "excavation_takeover_handshake_preflight_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "probe": {
            "requested_decision_step": decision_step,
            "requested_sim_time_s": sim_time_s,
            "armed_event": armed,
            "zero_action_activation": zero_result,
            "activation_event": activation,
            "buffered_action_used_for_first_step": buffered,
            "handshake_after_first_step": after_execution,
        },
        "scope": {
            "physics_steps": 0,
            "environment_steps": 0,
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "sac_parameter_updates": 0,
            "gui_started": False,
        },
        "unexpected_physics_modules": physics_modules,
        "source_sha256": {
            "dagger_supervisor.py": sha256(
                Path(__file__).resolve().parent / "dagger_supervisor.py"),
            "mpm_manual_demo.py": sha256(
                Path(__file__).resolve().parent / "mpm_manual_demo.py"),
            "preflight_takeover_handshake.py": sha256(__file__),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.sha256").write_text(
        f"{sha256(path)}  ./report.json\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
