"""Short live MPM contact smoke with read-only separation and fill truth.

This replays the mean 20 Hz actions from the qualified manual extraction only
far enough to establish real soil contact.  It is not a force-dynamic replay,
an episode success evaluation, or a training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from contracts import ACTION_SCHEMA, AXES, FourAxisAction
from mpm_smoke import LiveFastTierBackend, actor_observation, _read
from runtime_adapter import POLICY_DT_S, RESET_SCHEMA, FourAxisRuntime, ResetRequest, step_request
from termination import SingleScoopTermination


SCHEMA = "excavation_four_axis_mpm_contact_truth_smoke_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manual_actions(path: Path, decisions: int) -> list[list[float]]:
    actions = []
    with path.open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            if expected_index >= decisions:
                break
            row = json.loads(line)
            if (row.get("schema") != "manual_four_axis_demo_sample_v1"
                    or row.get("sample_index") != expected_index):
                raise ValueError("manual_action_sequence_mismatch")
            action = row.get("executed_action", {})
            if action.get("axis_order") != list(AXES):
                raise ValueError("manual_action_axis_order_mismatch")
            values = action.get("normalized_velocity_mean")
            FourAxisAction.parse({
                "schema": ACTION_SCHEMA,
                "axis_order": list(AXES),
                "normalized_velocity": values,
            })
            actions.append([float(value) for value in values])
    if len(actions) != decisions:
        raise ValueError("insufficient_manual_actions")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--empty-report", type=Path, required=True)
    parser.add_argument("--fast-reference-report", type=Path, required=True)
    parser.add_argument("--fill-root", type=Path, required=True)
    parser.add_argument("--fill-calibration-report", type=Path, required=True)
    parser.add_argument("--fill-held-out-report", type=Path, required=True)
    parser.add_argument("--manual-demo", type=Path, required=True)
    parser.add_argument("--episode-id", default="four-axis-contact-truth-smoke-v1")
    parser.add_argument("--decisions", type=int, default=170)
    parser.add_argument("--contact-force-threshold-n", type=float, default=100.0)
    parser.add_argument("--max-wall", type=float, default=180.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_directory_exists:" + str(args.output))
    if args.decisions <= 0 or args.decisions > 400:
        raise ValueError("invalid_contact_smoke_decisions")
    if (not math.isfinite(args.contact_force_threshold_n)
            or args.contact_force_threshold_n <= 0.0):
        raise ValueError("invalid_contact_force_threshold")
    if not math.isfinite(args.max_wall) or args.max_wall <= 0.0:
        raise ValueError("invalid_max_wall")
    actions = load_manual_actions(args.manual_demo, args.decisions)
    source_root = Path(__file__).resolve().parent
    execution_source_paths = tuple(source_root / name for name in (
        "mpm_contact_smoke.py", "mpm_smoke.py", "online_truth.py",
        "runtime_adapter.py", "contracts.py", "termination.py"))
    execution_sources_before = {
        str(path): sha256(path) for path in execution_source_paths}
    geometry = _read(args.geometry_report)
    target = geometry["selection"]["entry_xy_m"] + [geometry["selection"]["surface_z_m"]]
    reset_value = {"schema": RESET_SCHEMA, "episode_id": args.episode_id,
                   "target_world_m": target}
    ResetRequest.parse(reset_value)

    made = []

    def factory(request):
        backend = LiveFastTierBackend(
            request, backend_root=args.backend_root, output=args.output,
            geometry_report=args.geometry_report, empty_report=args.empty_report,
            fast_reference_report=args.fast_reference_report, fill_root=args.fill_root,
            fill_calibration_report=args.fill_calibration_report,
            fill_held_out_report=args.fill_held_out_report)
        made.append(backend)
        return backend

    runtime = FourAxisRuntime(factory)
    termination = SingleScoopTermination()
    started = time.perf_counter()
    report = {
        "schema": SCHEMA,
        "status": "initializing",
        "episode_id": args.episode_id,
        "requested_decisions": args.decisions,
        "requested_sim_time_s": args.decisions * POLICY_DT_S,
        "max_wall_s": args.max_wall,
        "contact_force_threshold_n": args.contact_force_threshold_n,
        "manual_demo_sha256": sha256(args.manual_demo),
        "execution_source_sha256": execution_sources_before,
        "manual_action_semantics": (
            "mean of each six-command manual block; diagnostic contact approach, "
            "not force-dynamic replay for blocks whose internal commands changed"),
        "training_started": False,
        "task_success_evaluated": False,
        "M2_loose_v1_qualified": False,
    }
    logs = []
    peak_force_n = 0.0
    strong_contact_steps = 0
    valid_separation_steps = 0
    online_fill_samples = []
    last_termination = None
    last_terrain_sequence = -1
    try:
        frame = runtime.reset(reset_value)
        backend = made[0]
        policy_log = (args.output / "policy-steps.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        truth_log = (args.output / "truth-steps.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        terrain_log = (args.output / "terrain-frames.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        logs = [policy_log, truth_log, terrain_log]
        for decision, action in enumerate(actions):
            if time.perf_counter() - started > args.max_wall:
                raise TimeoutError("contact_smoke_wall_budget")
            if (args.output / "STOP").exists():
                raise InterruptedError("cooperative_stop")
            observation = actor_observation(frame, target)
            result = runtime.step(step_request(frame, action))
            next_frame = result["observation"]
            next_observation = actor_observation(next_frame, target)
            force_n = float(np.linalg.norm(backend.force))
            peak_force_n = max(peak_force_n, force_n)
            truth = backend.termination_inputs()
            if truth["soil_body_separation_valid"]:
                valid_separation_steps += 1
            strong_contact = bool(
                force_n >= args.contact_force_threshold_n
                and truth["terrain_contact"] is True)
            if strong_contact:
                strong_contact_steps += 1
            bucket_closure = next_observation[
                "proprioception"]["cylinder_position_normalized"][2]
            need_fill = termination.sample_required(
                next_frame["sim_time_s"],
                digging_contact=truth["digging_contact_current"],
                terrain_contact=truth["terrain_contact"],
                bucket_closure_normalized=bucket_closure)
            if strong_contact and not online_fill_samples:
                need_fill = True
            if need_fill:
                truth = backend.termination_inputs(require_fill=True)
                if truth["fill_measurement_valid"]:
                    online_fill_samples.append({
                        "decision_step": decision + 1,
                        "sim_time_s": next_frame["sim_time_s"],
                        **truth["fill_measurement"],
                    })
            if not truth["soil_body_separation_valid"]:
                raise RuntimeError("invalid_soil_body_separation")
            last_termination = termination.update(
                next_frame["sim_time_s"],
                digging_contact=truth["digging_contact_current"],
                terrain_contact=truth["terrain_contact"],
                bucket_closure_normalized=bucket_closure,
                fill_fraction=(truth["fill_fraction"] if need_fill else None),
            )
            policy_log.write(json.dumps({
                "decision_step": decision,
                "observation": observation,
                "action": {"schema": ACTION_SCHEMA, "axis_order": list(AXES),
                           "normalized_velocity": action},
                "plan": result["plan"],
                "next_observation": next_observation,
            }, separators=(",", ":"), allow_nan=False) + "\n")
            truth_log.write(json.dumps({
                "decision_step": decision + 1,
                "sim_time_s": next_frame["sim_time_s"],
                "force_norm_n": force_n,
                "strong_contact": strong_contact,
                "truth": truth,
                "termination": last_termination.as_dict(),
            }, separators=(",", ":"), allow_nan=False) + "\n")
            if next_frame["terrain_sequence"] != last_terrain_sequence:
                terrain_log.write(json.dumps({
                    "terrain_sequence": next_frame["terrain_sequence"],
                    "capture_sim_s": next_frame["terrain_capture_sim_s"],
                    "terrain": next_frame["terrain"],
                }, separators=(",", ":"), allow_nan=False) + "\n")
                last_terrain_sequence = next_frame["terrain_sequence"]
            frame = next_frame
            if last_termination.terminated or last_termination.truncated:
                raise RuntimeError("contact_smoke_unexpected_termination:" +
                                   str(last_termination.reason))
        for stream in logs:
            stream.flush()
        report["status"] = "passed"
        report["final_observation"] = actor_observation(frame, target)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
    finally:
        for stream in logs:
            try:
                stream.close()
            except BaseException:
                pass
        try:
            runtime.close()
        except BaseException as error:
            report["status"] = "failed"
            report["cleanup_error"] = repr(error)
        report["wall_s"] = time.perf_counter() - started
        report["runtime_lifecycle"] = runtime.lifecycle
        report["peak_force_n"] = peak_force_n
        report["strong_contact_steps"] = strong_contact_steps
        report["valid_separation_steps"] = valid_separation_steps
        report["online_fill_samples"] = online_fill_samples
        report["termination_state"] = (
            last_termination.as_dict() if last_termination is not None else None)
        execution_sources_after = {
            str(path): sha256(path) for path in execution_source_paths}
        report["execution_sources_unchanged"] = (
            execution_sources_before == execution_sources_after)
        if made:
            backend = made[0]
            report["runtime_counters"] = runtime.counters
            report["backend_counters"] = {
                "mechanism_steps": backend.mechanism_steps,
                "logged_mechanism_steps": backend.logged_mechanism_steps,
                "soil_steps": backend.soil_steps,
                "logged_soil_steps": backend.logged_soil_steps,
                "terrain_captures": backend.capture_calls,
                "separation_evaluations": backend.separation_evaluations,
                "fill_evaluations": backend.fill_evaluations,
                "maximum_scene_outliers": backend.maximum_scene_outliers,
            }
            report["qualification"] = backend.qualification
            report["sources_unchanged"] = (
                backend.source_before is not None
                and backend.source_before == backend.source_after)
            report["truth_sources_unchanged"] = (
                backend.truth_sources_before is not None
                and backend.truth_sources_before == backend.truth_sources_after)
            expected = {
                "policy_steps": args.decisions,
                "mechanism_steps": args.decisions * 6,
                "soil_steps": args.decisions * 3,
                "terrain_captures": 1 + args.decisions // 4,
            }
            counts_match = (
                runtime.counters == expected
                and backend.mechanism_steps == backend.logged_mechanism_steps
                == expected["mechanism_steps"]
                and backend.soil_steps == backend.logged_soil_steps
                == expected["soil_steps"])
            report["checks"] = {
                "exact_20_120_60_cadence": runtime.counters == expected,
                "physical_and_log_counts_match": counts_match,
                "genuine_nonzero_soil_contact": (
                    peak_force_n >= args.contact_force_threshold_n
                    and strong_contact_steps > 0),
                "soil_body_separation_valid_every_policy_step": (
                    valid_separation_steps == args.decisions),
                "online_2cm_fill_computed_and_valid": bool(online_fill_samples),
                "termination_did_not_claim_success": (
                    last_termination is not None and not last_termination.success
                    and not last_termination.terminated and not last_termination.truncated),
                "source_files_unchanged": report["sources_unchanged"],
                "truth_sources_unchanged": report["truth_sources_unchanged"],
                "execution_sources_unchanged": report["execution_sources_unchanged"],
                "training_not_started": True,
            }
            if report["status"] == "passed" and not all(report["checks"].values()):
                report["status"] = "failed"
                report["error"] = "post_run_check_failed"
        if args.output.exists():
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: report.get(key) for key in (
            "status", "error", "wall_s", "peak_force_n", "strong_contact_steps",
            "runtime_counters", "backend_counters", "checks")},
            indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
