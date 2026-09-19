"""Bounded live single-scoop episode for the four-axis fast-tier runtime.

The approved manual 20 Hz mean actions drive the digging portion.  If the
manual sequence ends before a terminal outcome, the runner applies a zero
velocity hold until the confirmed 60 simulated-second limit.  This is an
environment/termination validation entry, never a training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from contracts import ACTION_SCHEMA, AXES
from mpm_contact_smoke import load_manual_actions
from mpm_smoke import LiveFastTierBackend, actor_observation, _read
from runtime_adapter import POLICY_DT_S, RESET_SCHEMA, FourAxisRuntime, ResetRequest, step_request
from termination import EPISODE_LIMIT_S, FILL_SAMPLE_COUNT, SingleScoopTermination


SCHEMA = "excavation_four_axis_mpm_single_scoop_episode_v1"
ZERO_ACTION = [0.0, 0.0, 0.0, 0.0]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_action(manual_actions: list[list[float]], decision_step: int):
    """Return the fixed manual prior action, then a safe zero-speed hold."""
    if decision_step < 0:
        raise ValueError("negative_decision_step")
    if decision_step < len(manual_actions):
        return list(manual_actions[decision_step]), "manual_mean"
    return list(ZERO_ACTION), "zero_hold"


def fill_sample_required(machine: SingleScoopTermination, sim_time_s: float,
                         *, digging_contact: bool, terrain_contact: bool,
                         bucket_closure_normalized: float) -> bool:
    """Delegate same-tick sampling to the shared termination machine."""
    return machine.sample_required(
        sim_time_s,
        digging_contact=digging_contact,
        terrain_contact=terrain_contact,
        bucket_closure_normalized=bucket_closure_normalized,
    )


def advance_termination(machine: SingleScoopTermination, sim_time_s: float,
                        truth: dict, bucket_closure_normalized: float,
                        require_fill):
    """Acquire an online fill sample exactly when this update requires one."""
    required = fill_sample_required(
        machine,
        sim_time_s,
        digging_contact=truth["digging_contact_current"],
        terrain_contact=truth["terrain_contact"],
        bucket_closure_normalized=bucket_closure_normalized,
    )
    if required:
        truth = require_fill()
    result = machine.update(
        sim_time_s,
        digging_contact=truth["digging_contact_current"],
        terrain_contact=truth["terrain_contact"],
        bucket_closure_normalized=bucket_closure_normalized,
        fill_fraction=truth["fill_fraction"] if required else None,
    )
    return truth, result, required


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


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
    parser.add_argument("--manual-decisions", type=int, default=539)
    parser.add_argument("--max-decisions", type=int,
                        default=int(EPISODE_LIMIT_S / POLICY_DT_S))
    parser.add_argument("--episode-id", default="four-axis-single-scoop-v1")
    parser.add_argument("--max-wall", type=float, default=600.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("output_directory_exists:" + str(args.output))
    if args.manual_decisions <= 0 or args.manual_decisions > args.max_decisions:
        raise ValueError("invalid_manual_decisions")
    full_limit_decisions = int(EPISODE_LIMIT_S / POLICY_DT_S)
    if args.max_decisions <= 0 or args.max_decisions > full_limit_decisions:
        raise ValueError("invalid_max_decisions")
    if not math.isfinite(args.max_wall) or args.max_wall <= 0.0:
        raise ValueError("invalid_max_wall")

    manual_actions = load_manual_actions(args.manual_demo, args.manual_decisions)
    source_root = Path(__file__).resolve().parent
    execution_source_paths = tuple(source_root / name for name in (
        "mpm_episode.py", "mpm_contact_smoke.py", "mpm_smoke.py",
        "online_truth.py", "runtime_adapter.py", "contracts.py", "termination.py"))
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
        "manual_decisions": args.manual_decisions,
        "maximum_decisions": args.max_decisions,
        "maximum_sim_time_s": args.max_decisions * POLICY_DT_S,
        "max_wall_s": args.max_wall,
        "manual_demo_sha256": sha256(args.manual_demo),
        "execution_source_sha256": execution_sources_before,
        "policy_semantics": (
            "approved manual 20 Hz mean action prior followed by zero-speed hold; "
            "not force-dynamic replay for changed source blocks"),
        "training_started": False,
        "critic_transition_emitted": False,
        "M2_loose_v1_qualified": False,
    }
    logs = []
    executed_decisions = 0
    last_terrain_sequence = -1
    valid_separation_steps = 0
    peak_force_n = 0.0
    online_fill_samples = []
    first_digging_contact = None
    first_separated_after_digging = None
    last_termination = None

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

        for decision in range(args.max_decisions):
            if time.perf_counter() - started > args.max_wall:
                raise TimeoutError("single_scoop_wall_budget")
            if (args.output / "STOP").exists():
                raise InterruptedError("cooperative_stop")
            observation = actor_observation(frame, target)
            action, action_source = episode_action(manual_actions, decision)
            step_result = runtime.step(step_request(frame, action))
            next_frame = step_result["observation"]
            next_observation = actor_observation(next_frame, target)
            truth = backend.termination_inputs()
            if not truth["soil_body_separation_valid"]:
                raise RuntimeError("invalid_soil_body_separation")
            valid_separation_steps += 1
            force_n = float(np.linalg.norm(backend.force))
            peak_force_n = max(peak_force_n, force_n)
            if truth["digging_contact_current"] and first_digging_contact is None:
                first_digging_contact = {
                    "decision_step": decision + 1,
                    "sim_time_s": next_frame["sim_time_s"],
                    "force_norm_n": force_n,
                }
            if (termination.digging_interaction_observed
                    and truth["terrain_contact"] is False
                    and first_separated_after_digging is None):
                first_separated_after_digging = {
                    "decision_step": decision + 1,
                    "sim_time_s": next_frame["sim_time_s"],
                    "mouth_angle_deg": truth["mouth_angle_deg"],
                }

            truth, last_termination, fill_required = advance_termination(
                termination, next_frame["sim_time_s"], truth,
                next_observation["proprioception"]["cylinder_position_normalized"][2],
                lambda: backend.termination_inputs(require_fill=True))
            if fill_required:
                online_fill_samples.append({
                    "decision_step": decision + 1,
                    "sim_time_s": next_frame["sim_time_s"],
                    **(truth["fill_measurement"] or {}),
                })

            policy_log.write(json.dumps({
                "decision_step": decision,
                "observation": observation,
                "action": {"schema": ACTION_SCHEMA, "axis_order": list(AXES),
                           "normalized_velocity": action},
                "action_source": action_source,
                "plan": step_result["plan"],
                "next_observation": next_observation,
            }, separators=(",", ":"), allow_nan=False) + "\n")
            truth_log.write(json.dumps({
                "decision_step": decision + 1,
                "sim_time_s": next_frame["sim_time_s"],
                "force_norm_n": force_n,
                "fill_sample_required": fill_required,
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
            executed_decisions += 1
            frame = next_frame
            if last_termination.terminated or last_termination.truncated:
                break

        if last_termination is None or not (
                last_termination.terminated or last_termination.truncated):
            raise RuntimeError("single_scoop_finished_without_terminal_outcome")
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
        report["executed_decisions"] = executed_decisions
        report["executed_sim_time_s"] = executed_decisions * POLICY_DT_S
        report["peak_force_n"] = peak_force_n
        report["valid_separation_steps"] = valid_separation_steps
        report["first_digging_contact"] = first_digging_contact
        report["first_separated_after_digging"] = first_separated_after_digging
        report["online_fill_samples"] = online_fill_samples
        report["termination_state"] = (
            last_termination.as_dict() if last_termination is not None else None)
        execution_sources_after = {
            str(path): sha256(path) for path in execution_source_paths}
        report["execution_sources_unchanged"] = (
            execution_sources_before == execution_sources_after)

        if made:
            backend = made[0]
            runtime_counters = runtime.counters
            expected = {
                "policy_steps": executed_decisions,
                "mechanism_steps": executed_decisions * 6,
                "soil_steps": executed_decisions * 3,
                "terrain_captures": 1 + executed_decisions // 4,
            }
            report["runtime_counters"] = runtime_counters
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
            task_terminal = bool(last_termination is not None and (
                last_termination.terminated or last_termination.truncated))
            sample_count = (len(last_termination.evaluation_samples)
                            if last_termination is not None else 0)
            report["checks"] = {
                "exact_20_120_60_cadence": runtime_counters == expected,
                "physical_and_log_counts_match": (
                    backend.mechanism_steps == backend.logged_mechanism_steps
                    == expected["mechanism_steps"]
                    and backend.soil_steps == backend.logged_soil_steps
                    == expected["soil_steps"]),
                "terminal_outcome_recorded": task_terminal,
                "separation_valid_every_policy_step": (
                    valid_separation_steps == executed_decisions),
                "online_fill_count_matches_termination": (
                    len(online_fill_samples) == sample_count
                    == backend.fill_evaluations),
                "runtime_closed": runtime.lifecycle == "closed",
                "source_files_unchanged": report["sources_unchanged"],
                "truth_sources_unchanged": report["truth_sources_unchanged"],
                "execution_sources_unchanged": report["execution_sources_unchanged"],
                "training_not_started": True,
            }
            report["milestones"] = {
                "digging_contact_observed": first_digging_contact is not None,
                "separation_after_digging_observed": first_separated_after_digging is not None,
                "off_soil_confirmation_completed": bool(
                    last_termination is not None
                    and last_termination.evaluation_start_s is not None),
                "full_thirteen_sample_window_observed": sample_count == FILL_SAMPLE_COUNT,
                "task_success": bool(last_termination and last_termination.success),
                "first_eight_episode_gate_unblocked": sample_count == FILL_SAMPLE_COUNT,
            }
            if report["status"] == "passed" and not all(report["checks"].values()):
                report["status"] = "failed"
                report["error"] = "post_run_check_failed"

        if args.output.exists():
            _write_json(args.output / "report.json", report)
        print(json.dumps({key: report.get(key) for key in (
            "status", "error", "wall_s", "executed_decisions",
            "termination_state", "runtime_counters", "backend_counters",
            "checks", "milestones")}, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
