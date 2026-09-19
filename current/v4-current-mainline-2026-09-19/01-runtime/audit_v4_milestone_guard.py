"""Read-only D-061 audit of milestone routing and reverse-curl support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ACCEPTED_ROLES = {
    "train_success_seed", "train_success", "sealed_blind_holdout",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def action(row):
    value = row.get("raw_action", row.get("action"))
    return [float(item) for item in value["normalized_velocity"]]


def analyze_rows(rows):
    phase = 0
    confirmation = 0
    transitions = {}
    first_reverse = None
    reverse_confirmation = 0
    negative_after_curl = 0
    maximum_bucket_position = -1.0
    for row in rows:
        step = int(row["decision_step"])
        observation = row["observation"]
        values = action(row)
        bucket_position = float(observation["proprioception"]
                                ["cylinder_position_normalized"][2])
        maximum_bucket_position = max(maximum_bucket_position, bucket_position)
        if phase == 0:
            condition = observation["digging_interaction_observed"] is True
            name = "penetrate_deepen"
        elif phase == 1:
            condition = (
                observation["digging_interaction_observed"] is True
                and values[3] >= 0.5)
            name = "curl_load"
        elif phase == 2:
            condition = bucket_position >= 0.90 and values[1] >= 0.5
            name = "lift_carry"
        else:
            condition = False
            name = None
        confirmation = confirmation + 1 if condition else 0
        if phase < 3 and confirmation >= 2:
            phase += 1
            transitions[name] = {
                "first_condition_decision_step": step - 1,
                "confirmed_at_decision_step": step,
            }
            confirmation = 0
        if phase >= 2:
            reverse = values[3] < -0.1
            if reverse:
                negative_after_curl += 1
            reverse_confirmation = reverse_confirmation + 1 if reverse else 0
            if first_reverse is None and reverse_confirmation >= 2:
                first_reverse = {
                    "first_reverse_decision_step": step - 1,
                    "pause_before_decision_step": step,
                }
    return {
        "row_count": len(rows),
        "transitions": transitions,
        "final_phase_id": phase,
        "negative_bucket_steps_after_curl": negative_after_curl,
        "first_two_tick_reverse": first_reverse,
        "maximum_bucket_position_normalized": maximum_bucket_position,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-report", type=Path, required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))
    label_report = read_json(args.label_report)
    source_paths = {
        Path(path).parent.name: Path(path)
        for path in label_report["source_sha256"]
        if path.endswith("/policy-steps.ndjson")
    }
    references = []
    for source in label_report["recorded_sources"]:
        if source["role"] not in ACCEPTED_ROLES:
            continue
        path = source_paths[source["episode_id"]]
        observed_hash = sha256(path)
        references.append({
            "episode_id": source["episode_id"],
            "role": source["role"],
            "source_sha256": observed_hash,
            "declared_sha256": label_report["source_sha256"][str(path)],
            "analysis": analyze_rows(read_jsonl(path)),
        })
    failed_policy = args.failed_run / "policy-steps.ndjson"
    failed_report = read_json(args.failed_run / "report.json")
    failed = analyze_rows(read_jsonl(failed_policy))
    checks = {
        "label_report_passed": label_report.get("status") == "passed",
        "eight_accepted_success_references": len(references) == 8,
        "reference_hashes_match": all(
            item["source_sha256"] == item["declared_sha256"]
            for item in references),
        "reference_milestone_sequences_complete": all(
            item["analysis"]["final_phase_id"] == 3 for item in references),
        "reference_reverse_pause_false_positives_zero": all(
            item["analysis"]["first_two_tick_reverse"] is None
            and item["analysis"]["negative_bucket_steps_after_curl"] == 0
            for item in references),
        "failed_run_is_d060_actor_only": (
            failed_report.get("episode_id") == "v4-protected-20260918-122910"
            and failed_report.get("actor_parameter_updates") == 0
            and failed_report.get("critic_parameter_updates") == 0
            and failed_report.get("sac_parameter_updates") == 0),
        "failed_run_curl_started": "curl_load" in failed["transitions"],
        "failed_run_never_reached_lift_milestone": (
            "lift_carry" not in failed["transitions"]),
        "failed_run_reverse_pause_detected": (
            failed["first_two_tick_reverse"] is not None),
        "failed_run_reverse_begins_at_256_pauses_before_257": (
            failed["first_two_tick_reverse"] == {
                "first_reverse_decision_step": 256,
                "pause_before_decision_step": 257,
            }),
        "physics_and_training_updates_zero": True,
    }
    report = {
        "schema": "excavation_v4_milestone_guard_audit_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "decision": "D-061",
        "checks": checks,
        "configuration": {
            "confirmation_ticks": 2,
            "curl_action_threshold": 0.5,
            "lift_boom_threshold": 0.5,
            "lift_bucket_position_min": 0.90,
            "reverse_bucket_threshold": -0.1,
            "reverse_pause_confirmation_ticks": 2,
            "reverse_execution": "pause_full_action_before_physics",
        },
        "references": references,
        "failed_run": {
            "episode_id": failed_report["episode_id"],
            "policy_sha256": sha256(failed_policy),
            "report_sha256": sha256(args.failed_run / "report.json"),
            "analysis": failed,
            "final_fill_percent": (
                failed_report["final_truth"]["fill_fraction"] * 100.0),
        },
        "scope": {
            "physics_steps": 0,
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "sac_parameter_updates": 0,
            "claim": (
                "fixed-log causal milestone and reverse-curl counterfactual only; "
                "no physical-success claim"),
        },
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
