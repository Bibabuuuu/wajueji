#!/usr/bin/env python3
"""Bounded, non-formal 120/60 Hz comparison; never changes practice defaults."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

import run_layered_47_interactive_proxy_resume_audit as audit
import run_layered_47_interactive_proxy_practice as practice
from run_layered_mound_three_rock_cover_release import _stable_particle


def schedule(hz):
    if hz not in (60, 120):
        raise ValueError("frequency_probe_only_60_or_120")
    return [("hold", hz, np.zeros(4)),
            ("swing", hz // 2, np.array([0.03, 0., 0., 0.])),
            ("stop", hz // 2, np.zeros(4))]


def probe_config(base, hz):
    schedule(hz)
    result = copy.deepcopy(base)
    result["probe"]["output_dt_s"] = 1. / hz
    result["probe"]["mujoco_internal_substeps"] = 4800 // hz
    return result


def strict_row(row, probe):
    return bool(row["safe"] and _stable_particle(row, probe)
                and row["maximum_rock_linear_speed_m_s"] <= probe[
                    "maximum_stable_rock_linear_speed_m_s"]
                and row["maximum_rock_angular_speed_rad_s"] <= probe[
                    "maximum_stable_rock_angular_speed_rad_s"]
                and row["bucket_mpm_contact_samples"] == 0
                and row["rock_bucket_contact_substeps"] == 0)


def diagnostic_safe(row):
    """Abort limits only, NOT relaxed qualification limits."""
    return bool(row["safe"] and row["bucket_mpm_contact_samples"] == 0
                and row["rock_bucket_contact_substeps"] == 0
                and row["maximum_rock_linear_speed_m_s"] <= .1
                and row["maximum_rock_angular_speed_rad_s"] <= 1.
                and row["particle_max_speed_m_s"] <= .5
                and row["particle_p99_speed_m_s"] <= .05
                and row["maximum_rock_displacement_from_source_m"] <= .005)


def timing(rows, hz):
    seconds = sum(r["total_frame_wall_seconds"] for r in rows)
    duration = len(rows) / hz
    return {"frames": len(rows), "simulation_seconds": duration,
            "wall_seconds": seconds,
            "simulation_seconds_per_wall_second": duration / seconds if seconds else None,
            "wall_seconds_per_simulation_second": seconds / duration if duration else None,
            **{key: sum(r[key] for r in rows) / duration if duration else None
               for key in ("mpm_wall_seconds", "mujoco_wall_seconds")}}


def subcycled_bridge_step(bridge, dt, substeps, command, wrenches):
    """Hold F=J/dt across 1 or 2 valid bridge calls, never apply J twice."""
    blocks = round(dt * 120)
    if blocks not in (1, 2) or abs(dt - blocks / 120) > 1e-12 or substeps != blocks * 40:
        raise ValueError("frequency_bridge_contract")
    states = [practice._bridge_step(bridge, 1/120, 40, command, wrenches)
              for _ in range(blocks)]
    # Preserve diagnostics from BOTH half-steps; later zero must not hide a violation.
    merged = dict(states[-1])
    merged["contact_substeps"] = {key: sum(s.get("contact_substeps", {}).get(key, 0) for s in states)
        for key in set().union(*(s.get("contact_substeps", {}) for s in states))}
    merged["forbidden_contact_count"] = sum(s.get("forbidden_contact_count", 0) for s in states)
    merged["closure_m"] = max(s["closure_m"] for s in states)
    if not all(audit._finite_tree(s) for s in states):
        raise ValueError("nonfinite_bridge_half_step")
    return merged


class MeasuredFixture(audit.preflight.ExcavationFixture):
    def step(self, dt, origins):
        result = super().step(dt, origins)
        self.last_wrenches = result[0]
        return result


def run(args):
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    out = Path(args.output).resolve()
    journal = out.with_suffix(".progress.ndjson")
    # Reserve the report before starting expensive work; never overwrite evidence.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump({"status": "initializing", "hz": args.hz}, stream)
    rows, bridge, fixture, before = [], None, None, None
    report = {"schema": "layered_47_frequency_probe_v1", "hz": args.hz,
              "scope": {"training_eligible": False, "manual_collection_eligible": False,
                        "formal_a30_equivalent": False, "bucket_contact_qualified": False},
              "protocol": {"phase_seconds": [1., .5, .5], "swing_rad_s": .03,
                           "internal_dt_s": 1. / 4800,
                           "bridge_control_hz": 120,
                           "bridge_blocks_per_mpm_step": 120 // args.hz,
                           "warmstart": "exact_source_unscaled_dt_switch_no_hidden_settling",
                           "diagnostic_hold_only": args.diagnostic_hold,
                           "stop_on_first_strict_failure": not args.diagnostic_hold,
                           "diagnostic_abort_limits": {"rock_linear_m_s": .1, "rock_angular_rad_s": 1.,
                               "particle_max_m_s": .5, "particle_p99_m_s": .05, "rock_displacement_m": .005},
                           "timing_only_excludes_initial_sim_seconds": .25}}
    try:
        config, loaded = audit.load_config(args.config, root)
        checkpoint = audit._resolve(root, config["input_checkpoint"]["path"])
        metadata = audit._resolve(root, config["input_checkpoint"]["metadata"])
        arrays, _, before = audit.proxy_checkpoint.load_checkpoint_readonly(checkpoint, metadata)
        base = probe_config(loaded["base"], args.hz)
        report.update(source_checkpoint=before, effective_probe=base["probe"],
                      solver_profile=loaded["profiles"]["voxel_10cm_sparse"],
                      source_metadata_sha256=audit.sha256(metadata),
                      config_sha256=audit.sha256(Path(args.config)),
                      runner_sha256=audit.sha256(Path(__file__)),
                      helper_sha256={name: audit.sha256(Path(module.__file__)) for name, module in
                                     (("practice", practice), ("preflight", audit.preflight),
                                      ("checkpoint", audit.proxy_checkpoint))})
        practice._append(journal, {"event": "started", **report})
        manifest = json.loads(audit._resolve(root, base["physical_geometry"]["manifest_path"]).read_text())
        records = [r for r in manifest["rocks"] if r["layer_id"] in audit.preflight.LAYERS]
        if [r["rock_id"] for r in records] != list(audit.preflight.DEFAULT_PROFILE["expected_ids"]):
            raise ValueError("manifest_order")
        _, proxy = audit.preflight._load_bound_json(root, base["open_bucket_proxy"], "open_bucket_proxy")
        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with tempfile.TemporaryDirectory(prefix="frequency-probe-") as raw:
            model, binary, mm, bm, command = audit.build_model_and_bridge(
                args, root, root / base["capacity_bridge"]["config_path"],
                audit.records_at_qpos(records, arrays["rigid_qpos"]), Path(raw), "frequency-probe")
            report["build"] = {"model": mm, "bridge": bm, "command": command}
            bridge = audit.RestoredRuntimeBridge(binary, model, len(records),
                arrays["mechanism_qpos"].tolist(), arrays["mechanism_qvel"].tolist(),
                arrays["rigid_qpos"], arrays["rigid_qvel"])
            fixture = MeasuredFixture(base, root, records, arrays["grid_reference_particle_q"],
                proxy, bridge.state, args.device, solver_profile=report["solver_profile"])
            fields = audit.restore_fixture_prefix(fixture, arrays)
            fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            fixture.assign(bridge.state)
            solver = audit.proxy_checkpoint.restore_solver_continuation(fixture, arrays)
            report["restore"] = {"fields": fields, "solver": solver}
            exact = all(fields["field_readback_exact"].values()) and all(
                solver["grid_readback_exact"].values()) and all(solver["solver_field_readback_exact"].values())
            for field, actual in (("rigid_qpos", [r["qpos"] for r in bridge.state["rocks"]]),
                                  ("rigid_qvel", [r["qvel"] for r in bridge.state["rocks"]]),
                                  ("mechanism_qpos", bridge.state["mechanism_qpos"]),
                                  ("mechanism_qvel", bridge.state["mechanism_qvel"])):
                error = float(np.max(np.abs(np.asarray(actual) - arrays[field])))
                report["restore"][field + "_max_error"] = error
                exact = exact and error <= 1e-12
            if not exact:
                raise ValueError("source_restore_not_exact")
            report["initialization_wall_seconds"] = time.perf_counter() - started
            dt = 1. / args.hz
            impulse = np.zeros(3)
            checkpoints = []
            phases = [("diagnostic_hold", 2 * args.hz, np.zeros(4))] if args.diagnostic_hold else schedule(args.hz)
            for phase, count, control in phases:
                for _ in range(count):
                    tick = time.perf_counter()
                    prior = np.array([r["qpos"][:3] for r in bridge.state["rocks"]])
                    row = practice._controlled_frame(fixture, bridge, base, control, len(rows) + 1,
                                                     bridge_step=subcycled_bridge_step)
                    # Helper's merged particle metrics can mask its aggregate finite flag.
                    row["finite"] = bool(row["finite"] and audit._finite_tree(bridge.state))
                    row["safe"] = bool(row["safe"] and row["finite"])
                    row["phase"] = phase
                    row["strict"] = strict_row(row, base["probe"])
                    position = np.array([r["qpos"][:3] for r in bridge.state["rocks"]])
                    row["maximum_rock_position_step_m"] = float(np.linalg.norm(position - prior, axis=1).max())
                    row["maximum_rock_displacement_from_source_m"] = float(np.linalg.norm(
                        position - arrays["rigid_qpos"][:, :3], axis=1).max())
                    row["diagnostic_safe"] = diagnostic_safe(row)
                    force = sum((np.asarray(w["force"]) for name, w in fixture.last_wrenches.items()
                                 if name != "bucket"), start=np.zeros(3))
                    impulse += force * dt
                    row["net_mpm_rock_force_n"] = force.tolist()
                    row["elapsed_simulation_seconds"] = (len(rows) + 1) * dt
                    row["total_frame_wall_seconds"] = time.perf_counter() - tick
                    rows.append(row)
                    practice._append(journal, {"event": "frame", "row": row})
                    if len(rows) == 1 or len(rows) % (args.hz // 4) == 0 or not row["strict"]:
                        print(f"FREQUENCY_PROBE hz={args.hz} frame={len(rows)}/{2*args.hz} phase={phase} strict={row['strict']} particle_max={row['particle_max_speed_m_s']:.6g}", flush=True)
                    if not (row["diagnostic_safe"] if args.diagnostic_hold else row["strict"]):
                        break
                # Phase/failure snapshot is diagnostic-only: never accepted as practice input.
                if rows and rows[-1]["finite"]:
                    snapshot = audit.proxy_checkpoint.capture_arrays(fixture, bridge.state,
                        float(arrays["simulation_time_s"][0]) + len(rows) * dt)
                    identity = audit.proxy_checkpoint.write_checkpoint_atomic(
                        out.with_name(out.stem + f"-{phase}.diagnostic.npz"), snapshot)
                    identity.update(phase=phase, frame=len(rows), restart_qualified=False)
                    checkpoints.append(identity)
                    audit._write_json(Path(identity["path"] + ".json"), {
                        "status": "frequency_probe_diagnostic_only_not_qualified_for_resume",
                        "hz": args.hz, "source": before, "scope": report["scope"], **identity})
                    report["diagnostic_snapshots"] = checkpoints
                    practice._append(journal, {"event": "phase_saved", **identity})
                if not (rows[-1]["diagnostic_safe"] if args.diagnostic_hold else rows[-1]["strict"]):
                    break
            report["net_mpm_rock_impulse_ns"] = impulse.tolist()
            report["endpoint_mechanism_qpos"] = bridge.state["mechanism_qpos"]
            report["initial_mechanism_qpos"] = arrays["mechanism_qpos"].tolist()
            report["maximum_rock_displacement_m"] = float(np.linalg.norm(
                np.array([r["qpos"][:3] for r in bridge.state["rocks"]]) - arrays["rigid_qpos"][:, :3], axis=1).max())
            report["checks"] = {"restore_exact": exact, "all_requested_frames": len(rows) == 2 * args.hz,
                "all_frames_strict": all(r["strict"] for r in rows),
                "source_unchanged": audit.checkpoint_identity(checkpoint) == before,
                "metadata_unchanged": audit.sha256(metadata) == report["source_metadata_sha256"],
                "first_step_position_continuous": rows[0]["maximum_rock_position_step_m"] <=
                    base["probe"]["maximum_stable_rock_linear_speed_m_s"] * dt * 1.01}
            report["status"] = "short_probe_passed" if all(report["checks"].values()) else "short_probe_rejected"
            if args.diagnostic_hold:
                tail = 0
                for row in reversed(rows):
                    if not row["strict"]:
                        break
                    tail += 1
                report["diagnostic_summary"] = {"strict_failure_frames": sum(not r["strict"] for r in rows),
                    "final_strict_tail_frames": tail, "final_strict_tail_sim_seconds": tail / args.hz,
                    "completed_requested_duration": len(rows) == 2 * args.hz,
                    "all_abort_guards_preserved": all(r["diagnostic_safe"] for r in rows)}
                report["status"] = "diagnostic_complete_not_qualified" if len(rows) == 2 * args.hz else "diagnostic_aborted"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "short_probe_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if bridge is not None:
            bridge.close(kill=report.get("status") == "short_probe_failed")
        report.update(trace=rows, timing_all=timing(rows, args.hz),
                      timing_after_initial_quarter_second=timing(rows[args.hz // 4:], args.hz),
                      total_wall_seconds=time.perf_counter() - started)
        # Non-finite diagnostic values are explicit nulls, not a second reporting failure.
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return None if isinstance(value, float) and not np.isfinite(value) else value
        audit._write_json(out, clean(report))
    return report


if __name__ == "__main__":
    parser = audit._parser()
    parser.add_argument("--hz", type=int, choices=(60, 120), required=True)
    parser.add_argument("--diagnostic-hold", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("status", "hz", "timing_all")}), flush=True)
    raise SystemExit(0 if result["status"] == "short_probe_passed" else 1)
