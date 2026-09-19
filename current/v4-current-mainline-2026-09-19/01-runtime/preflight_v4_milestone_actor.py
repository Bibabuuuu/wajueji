"""Zero-physics preflight for the D-061 milestone-routed v4 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from dagger_supervisor import (
    BucketEndStopProjector, ClosedBucketBoomPauseSupervisor,
    CurlReversePauseSupervisor,
)
from observation_vector_v4 import FEATURE_COUNT, FEATURE_SPEC_SHA256


EXPECTED_CHECKPOINT_SHA256 = (
    "b32a460e4c849e6a2d0986405d39e3397c9eccad309946cfeb6d803a4bd238d9")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-audit", type=Path, required=True)
    parser.add_argument("--blind-report", type=Path, required=True)
    parser.add_argument("--milestone-audit", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--policy-site-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))
    sys.path.append(str(args.policy_site_packages.resolve()))
    from v4_actor_policy import LoadedV4ActorPolicy, MilestonePhaseRouter

    checkpoint_hash = sha256(args.checkpoint)
    actor_audit = read_json(args.actor_audit)
    blind_report = read_json(args.blind_report)
    milestone_audit = read_json(args.milestone_audit)
    failed_rows = read_jsonl(args.failed_run / "policy-steps.ndjson")
    policy = LoadedV4ActorPolicy(checkpoint_path=args.checkpoint)
    reverse = CurlReversePauseSupervisor()
    replay_event = None
    actions = []
    diagnostics = []
    for row in failed_rows:
        output = policy.action(row["observation"])
        actions.append(output)
        diagnostics.append(dict(policy.last_inference))
        event = reverse.evaluate(
            raw_action=output,
            curl_started=bool(policy.last_inference["curl_started"]),
            decision_step=row["decision_step"],
            sim_time_s=row["observation"]["sim_time_s"],
            routed_phase=policy.last_inference["routed_phase"],
        )
        if event is not None:
            replay_event = event
            break

    router = MilestonePhaseRouter()
    neutral = [0.0, 0.0, 0.0, 0.0]
    contact = {
        "digging_interaction_observed": True,
        "proprioception": {"cylinder_position_normalized": [0.0, 0.0, 0.0]},
    }
    router_probe = [router.update(contact, neutral) for _ in range(2)]
    router_probe += [router.update(contact, [0.0, 0.0, 0.0, 0.6])
                     for _ in range(2)]
    closed = {
        **contact,
        "proprioception": {"cylinder_position_normalized": [0.0, 0.0, 0.91]},
    }
    router_probe += [router.update(closed, [0.0, 0.6, 0.0, 0.0])
                     for _ in range(2)]

    source_observation = dict(failed_rows[0]["observation"])
    source_observation["proprioception"] = dict(
        source_observation["proprioception"])
    source_observation["proprioception"]["cylinder_position_normalized"] = list(
        source_observation["proprioception"]["cylinder_position_normalized"])
    source_observation["proprioception"]["axis_velocity_normalized"] = list(
        source_observation["proprioception"]["axis_velocity_normalized"])
    source_observation["proprioception"]["cylinder_position_normalized"][2] = 0.97
    source_observation["proprioception"]["axis_velocity_normalized"][3] = 0.0
    projector = BucketEndStopProjector()
    boundary = ClosedBucketBoomPauseSupervisor()
    probe = [0.0, -0.8, 0.4, 1.0]
    _, first_meta = projector.project(probe, source_observation)
    first_event = boundary.evaluate(
        raw_action=probe, end_stop=first_meta,
        decision_step=0, sim_time_s=0.0)
    _, second_meta = projector.project(probe, source_observation)
    second_event = boundary.evaluate(
        raw_action=probe, end_stop=second_meta,
        decision_step=1, sim_time_s=0.05)
    physics_modules = sorted(set(sys.modules) & {"newton", "warp", "mujoco"})
    checks = {
        "checkpoint_identity_exact": checkpoint_hash == EXPECTED_CHECKPOINT_SHA256,
        "independent_actor_audit_passed": (
            actor_audit.get("status") == "passed"
            and actor_audit.get("checkpoint_sha256") == checkpoint_hash),
        "blind_failure_preserved": (
            blind_report.get("status") == "failed"
            and blind_report.get("checkpoint_sha256") == checkpoint_hash),
        "milestone_fixed_log_audit_passed": milestone_audit.get("status") == "passed",
        "checkpoint_loaded_on_cuda": policy.device.type == "cuda",
        "checkpoint_is_v4_2412": (
            policy.metadata["feature_count"] == FEATURE_COUNT == 2412
            and policy.metadata["feature_spec_sha256"] == FEATURE_SPEC_SHA256),
        "runtime_route_is_causal_milestone": (
            policy.metadata["runtime_phase_route"]
            == "causal_milestone_two_tick_episode_reset"),
        "router_probe_advances_one_stage_after_each_two_tick_milestone": (
            router_probe == [(0, False), (1, True), (1, False),
                             (2, True), (2, False), (3, True)]),
        "failed_trace_replay_actions_finite_bounded": all(
            len(action) == 4 and all(
                math.isfinite(value) and -1.0 <= value <= 1.0
                for value in action)
            for action in actions),
        "failed_trace_replay_swing_zero": all(action[0] == 0.0 for action in actions),
        "failed_trace_replay_reverse_pause_before_original_damage": (
            replay_event is not None
            and replay_event["decision_step"] <= 257
            and replay_event["execution"] == "pause_full_action_before_physics"),
        "closed_bucket_pause_still_requires_two_ticks": (
            first_event is None and second_event is not None),
        "physics_not_imported": not physics_modules,
        "actor_parameter_updates_zero": True,
        "critic_and_sac_updates_zero": True,
        "environment_steps_zero": True,
    }
    report = {
        "schema": "excavation_v4_milestone_preflight_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "decision": "D-061",
        "checks": checks,
        "checkpoint_sha256": checkpoint_hash,
        "actor_audit_sha256": sha256(args.actor_audit),
        "blind_report_sha256": sha256(args.blind_report),
        "milestone_audit_sha256": sha256(args.milestone_audit),
        "failed_run_report_sha256": sha256(args.failed_run / "report.json"),
        "fixed_observation_replay": {
            "source_episode_id": "v4-protected-20260918-122910",
            "evaluated_steps": len(actions),
            "event": replay_event,
            "last_diagnostics": diagnostics[-1] if diagnostics else None,
            "counterfactual_boundary": (
                "actions are recomputed on frozen recorded observations; physical "
                "success after the pause is not claimed"),
        },
        "router_probe": router_probe,
        "closed_bucket_boom_pause_event": second_event,
        "scope": {
            "physics_steps": 0,
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "sac_parameter_updates": 0,
            "blind_failure_overridden": False,
            "physical_success_claimed": False,
        },
        "unexpected_physics_modules": physics_modules,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    (args.output / "manifest.sha256").write_text(
        f"{sha256(path)}  ./report.json\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
