#!/usr/bin/env python3
"""Read-only 47/65-rock grid/frequency candidates, not excavation."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

import run_layered_47_excavation_preflight as p47
import run_layered_65_excavation_preflight as p65
import run_layered_47_interactive_proxy_benchmark as bench
import layered_47_interactive_proxy_checkpoint as snapshots
from layered_mound_checkpoint_runtime import restore_runtime
from layered_frequency_bridge import HoldBridge


REQUIRED_CHECKS = ('source_unchanged', 'metadata_unchanged', 'all_frames_safe',
                   'warmup_strict_tail_achieved', 'all_measured_frames_strict')


def policy(hz, transition="grid"):
    if hz not in (60, 120):
        raise ValueError('frequency_must_be_60_or_120')
    if transition not in ("grid", "frequency") or (transition == "frequency" and hz != 60):
        raise ValueError('unsupported_transition')
    return dict(hz=hz, warmup_min=hz // 2, warmup_max=hz * 2,
                required_strict_tail=hz // 2, measured_frames=hz,
                all_transition_frames_logged=True,
                warmup_protocol=("frequency_from_audited_120hz_full_state_v3" if transition == "frequency"
                    else "historical_local_tail_with_retained_hard_guards_v2"))


def select(count):
    if count == 47:
        return p47.load_config, p47.DEFAULT_PROFILE
    if count == 65:
        return p65.load_config, p65.PROFILE
    raise ValueError("rock_count_must_be_47_or_65")


def profile(base, name):
    result = p47.mpm_solver_profile(base)
    if name == "sparse10":
        result.update(voxel_size_m=.1, grid_type="sparse", grid_padding=2)
    elif name != "fixed5":
        raise ValueError("unknown_solver_candidate")
    return p47.mpm_solver_profile(base, result)


def guard_checks(row, phase="measured"):
    if phase not in ("warmup", "measured"):
        raise ValueError("unknown_guard_phase")
    limits = {"forbidden_contact_count": 0, "external_rigid_support_substeps": 0,
        "unmapped_active_collider_sample_count": 0, "nonfinite_collider_sample_count": 0,
        "bucket_mpm_contact_samples": 0, "rock_bucket_contact_substeps": 0,
        "closure_m": .001, "maximum_rock_linear_speed_m_s": .1,
        "maximum_rock_angular_speed_rad_s": 1., "particle_max_speed_m_s": .5,
        "rock_displacement_from_source_m": .005}
    # Historical successful grid conversion allowed transient p99 violations.
    # Only omit its diagnostic abort in the bounded warmup; row['stable'] still
    # uses the ORIGINAL strict particle/rock thresholds in EVERY frame.
    if phase != "warmup":
        limits["particle_p99_speed_m_s"] = .05
    return {"finite": bool(row["finite"]), **{
        key: bool(np.isfinite(row[key]) and 0 <= row[key] <= limit)
        for key, limit in limits.items()}}


def guard(row, phase="measured"):
    return all(guard_checks(row, phase).values())


def classify_frame(row, phase, tail, transition="grid"):
    guard_phase = "measured" if transition == "frequency" else phase
    row["guard_failures"] = [k for k, passed in guard_checks(row, guard_phase).items() if not passed]
    row["guard"] = not row["guard_failures"]
    row["strict"] = bool(row["guard"] and row["stable"])
    return tail + 1 if row["strict"] else 0


def validate_seed_request(args):
    seed = getattr(args, "seed_candidate_report", None)
    audit = getattr(args, "seed_resume_report", None)
    if bool(seed) != bool(audit):
        raise ValueError("seed_requires_candidate_and_independent_audit")
    if seed and (args.hz != 60 or args.solver != "sparse10"):
        raise ValueError("seed_only_for_sparse_120_to_60_transition")
    if args.hz == 60 and args.solver == "sparse10" and not seed:
        raise ValueError("sparse_60hz_requires_audited_120hz_seed")


def load_seed(args, base, original_identity):
    import run_layered_sparse_resume_audit as resume
    path = Path(args.seed_candidate_report).resolve()
    audit_path = Path(args.seed_resume_report).resolve()
    arrays, source, identity = resume.load_candidate(path, base, args.rocks, 120)
    audit = json.loads(audit_path.read_text())
    resume.validate_seed_audit(audit, path, source, identity, args.rocks, 120)
    if (source['config_sha256'] != bench.sha256(Path(args.config))
            or source['source_checkpoint'] != original_identity
            or source['metadata_sha256'] != bench.sha256(Path(args.checkpoint_metadata))
            or source.get('transition', 'grid') != 'grid'):
        raise ValueError('seed_original_source_binding')
    return arrays, dict(checkpoint=identity,
        metadata_sha256=source['diagnostic_snapshots'][-1]['metadata_sha256'],
        candidate_report=str(path), candidate_report_sha256=bench.sha256(path),
        resume_report=str(audit_path), resume_report_sha256=bench.sha256(audit_path),
        input_hz=120, output_hz=60)


def seed_unchanged(seed):
    return (bench.checkpoint_identity(seed['checkpoint']['path']) == seed['checkpoint']
        and bench.sha256(Path(seed['checkpoint']['path'] + '.json')) == seed['metadata_sha256']
        and bench.sha256(Path(seed['candidate_report'])) == seed['candidate_report_sha256']
        and bench.sha256(Path(seed['resume_report'])) == seed['resume_report_sha256'])


def run(args):
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump({"status": "initializing"}, stream)
    journal = out.with_suffix(".progress.ndjson")
    started = time.perf_counter()
    bridge = source_bridge = fixture = before = None
    warm, measured = [], []
    transition = "frequency" if getattr(args, "seed_candidate_report", None) else "grid"
    selected_policy = None
    report = {"schema": "layered_sparse_a30_probe_v2", "status": "initializing",
              "rocks": args.rocks, "solver": args.solver, "device": args.device,
              "scope": {"formal_a30_equivalent": False, "training_eligible": False,
                        "manual_collection_eligible": False, "digging_executed": False},
              "policy": selected_policy, "transition": transition}
    def save_snapshot(label):
        if fixture is None or bridge is None or args.solver != "sparse10":
            return
        values = snapshots.capture_arrays(fixture, bridge.state,
            float(arrays["simulation_time_s"][0]) + (len(warm) + len(measured)) / args.hz)
        info = snapshots.write_checkpoint_atomic(out.with_name(out.stem + f"-{label}.diagnostic.npz"), values)
        meta = {"status": "grid_candidate_diagnostic_only", "scope": report["scope"],
                "solver_profile": report["solver_profile"], "hz": args.hz,
                "source": before, "transition": transition, "seed": report.get("seed"),
                "independent_restore_qualified": False, **info}
        bench._write_json(Path(info["path"] + ".json"), meta)
        info["metadata_sha256"] = bench.sha256(Path(info["path"] + ".json"))
        report.setdefault("diagnostic_snapshots", []).append(info)
        bench._append_journal(journal, {"event": "diagnostic_saved", **info})
    try:
        validate_seed_request(args)
        selected_policy = policy(args.hz, transition)
        report['policy'] = selected_policy
        root = Path(args.repository_root).resolve()
        loader, identity = select(args.rocks)
        base = loader(args.config, root)
        if base["probe"]["output_dt_s"] != 1/120 or base["probe"]["mujoco_internal_substeps"] != 40:
            raise ValueError("source_config_not_frozen_120hz")
        arrays, metadata, before = bench.load_checkpoint_readonly(
            args.checkpoint, args.checkpoint_metadata, base["input_checkpoint"])
        original_positions = arrays['rigid_qpos'][:, :3].copy()
        report.update(source_checkpoint=before, config_sha256=bench.sha256(Path(args.config)),
            metadata_sha256=bench.sha256(Path(args.checkpoint_metadata)),
            runner_sha256=bench.sha256(Path(__file__)), solver_profile=profile(base, args.solver))
        manifest = json.loads((root / base["physical_geometry"]["manifest_path"]).read_text())
        records = [r for r in manifest["rocks"] if r["layer_id"] in identity["layers"]]
        if [r["rock_id"] for r in records] != identity["expected_ids"]:
            raise ValueError("source_manifest_order")
        _, operator = p47._load_bound_json(root, base["operator_start"], "operator")
        _, proxy = p47._load_bound_json(root, base["open_bucket_proxy"], "proxy")
        if transition == "frequency":
            arrays, report['seed'] = load_seed(args, base, before)
        report['displacement_reference'] = ("audited_120hz_seed" if transition == "frequency" else "original_checkpoint")
        bench._append_journal(journal, {"event": "started", **report})
        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with tempfile.TemporaryDirectory(prefix="sparse-a30-probe-") as raw:
            work = Path(raw)
            if transition == "frequency":
                from run_layered_sparse_resume_audit import restore_candidate_runtime
                fixture, bridge, report['seed_restore'] = restore_candidate_runtime(
                    args, base, root, records, arrays, proxy, report['solver_profile'], work)
                report['grid_transition'] = "none_exact_sparse_grid_and_warmstart_restored"
            else:
                # Verify the source's complete fixed-grid continuation BEFORE conversion.
                source_fixture, source_bridge, restored = restore_runtime(
                    args, base, root, records, arrays, work, "source")
                report["source_restore"] = restored
                source_bridge.close()
                source_bridge = None
                model = work / "source.xml"
                bridge = bench.RestoredRuntimeBridge(Path(restored["binary"]), model, len(records),
                    operator["qpos"], [0.] * 18, arrays["rigid_qpos"], arrays["rigid_qvel"])
                fixture = p47.ExcavationFixture(base, root, records, arrays["grid_reference_particle_q"],
                    proxy, bridge.state, args.device, solver_profile=report["solver_profile"])
                report["particle_restore"] = bench.restore_fixture_prefix(fixture, arrays)
                if not all(report["particle_restore"]["field_readback_exact"].values()):
                    raise ValueError("particle_restore_inexact")
                fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                fixture.assign(bridge.state)
                if args.solver == "fixed5":
                    report["grid_transition"] = p47.transfer_solver_warmstart(source_fixture, fixture)
                else:
                    report["grid_transition"] = "new_10cm_sparse_grid_cold_grid_warmstart_particle_history_preserved"
                del source_fixture
                bridge = HoldBridge(bridge)
            bucket = p47._bucket_world_vertices(proxy, bridge.state)
            top = max(float(arrays["particle_q"][:, 2].max()) + .45 * base["mpm"]["particle_spacing_m"],
                p47._rock_world_max_z(records, arrays["rigid_qpos"], root))
            report["initial_vertical_clearance_m"] = float(bucket[:, 2].min() - top)
            if report["initial_vertical_clearance_m"] < .02:
                raise ValueError("initial_clearance")
            # Source restore used its ORIGINAL 120 Hz config; only now change the outer step.
            base["probe"] = {**base["probe"], "output_dt_s": 1 / args.hz,
                             "mujoco_internal_substeps": 4800 // args.hz}
            report["initialization_wall_seconds"] = time.perf_counter() - started
            tail = 0
            required = selected_policy["required_strict_tail"]
            for phase, limit in (("warmup", selected_policy["warmup_max"]), ("measured", args.hz)):
                if phase == "measured" and tail < required:
                    break
                target = warm if phase == "warmup" else measured
                for _ in range(limit):
                    row, _ = bench._advance_frame(fixture, bridge, base, len(warm) + len(measured) + 1)
                    row["finite"] = bool(row["finite"] and bench._finite_tree(bridge.state))
                    positions = np.array([r["qpos"][:3] for r in bridge.state["rocks"]])
                    row["rock_displacement_from_source_m"] = float(np.linalg.norm(
                        positions - arrays["rigid_qpos"][:, :3], axis=1).max())
                    row["rock_displacement_from_original_m"] = float(np.linalg.norm(
                        positions - original_positions, axis=1).max())
                    tail = classify_frame(row, phase, tail, transition)
                    target.append(row)
                    bench._append_journal(journal, {"event": phase, "row": row, "strict_tail": tail})
                    if len(target) == 1 or len(target) % 15 == 0 or not row["guard"]:
                        print(f"SPARSE_PROBE rocks={args.rocks} solver={args.solver} phase={phase} frame={len(target)}/{limit} strict={row['strict']} tail={tail}", flush=True)
                    if not row["guard"] or (phase == "measured" and not row["strict"]):
                        report["rejection"] = {"phase": phase, "frame": row["frame"],
                            "reason": "guard_failed" if not row["guard"] else "strict_stability_failed",
                            "row": row}
                        break
                    if phase == "warmup" and len(warm) >= selected_policy["warmup_min"] and tail >= required:
                        break
                if target and target[-1]["finite"]:
                    save_snapshot(phase)
                if not target or not target[-1]["guard"]:
                    break
            report["checks"] = {"source_unchanged": bench.checkpoint_identity(args.checkpoint) == before,
                "metadata_unchanged": bench.sha256(Path(args.checkpoint_metadata)) == report["metadata_sha256"],
                "all_frames_safe": all(r["guard"] for r in warm + measured),
                "warmup_strict_tail_achieved": len(warm) >= required and all(r["strict"] for r in warm[-required:]),
                "all_measured_frames_strict": len(measured) == args.hz and all(r["strict"] for r in measured)}
            if transition == "frequency":
                report['checks']['audited_seed_unchanged'] = seed_unchanged(report['seed'])
                report['checks']['seed_restore_exact'] = report['seed_restore']['exact']
            report["all_transition_frames_strict"] = all(r["strict"] for r in warm)
            report["status"] = "candidate_short_hold_passed" if all(report["checks"].values()) else "candidate_rejected"
            if report["status"] == "candidate_rejected" and "rejection" not in report:
                report["rejection"] = {"reason": "qualification_checks_failed",
                    "failed_checks": [k for k, passed in report["checks"].items() if not passed],
                    "strict_tail": tail, "last_warmup_row": warm[-1] if warm else None}
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "probe_failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        for active in (bridge, source_bridge):
            if active is not None:
                try:
                    active.close(kill=report["status"] == "probe_failed")
                except Exception as exc:
                    report["cleanup_error"] = str(exc)
        report.update(warmup_trace=warm, measured_trace=measured,
            execution={"wall_seconds": time.perf_counter() - started, "warmup_frames": len(warm),
                       "measured_frames": len(measured), "journal": str(journal)})
        total = sum(r["wall_seconds"] for r in measured)
        report["timing"] = {"measured_wall_seconds": total,
            "measured_simulation_seconds": len(measured) / args.hz,
            "simulation_seconds_per_wall_second": len(measured) / args.hz / total if total else None,
            "wall_seconds_per_simulation_second": total * args.hz / len(measured) if measured else None}
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return None if isinstance(v, float) and not np.isfinite(v) else v
        bench._write_json(out, clean(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "repository-root", "output", "checkpoint", "checkpoint-metadata",
                 "mujoco-include", "mujoco-library-dir", "mechanism-header-dir", "controller-header-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--rocks", type=int, choices=(47, 65), required=True)
    parser.add_argument("--solver", choices=("fixed5", "sparse10"), required=True)
    parser.add_argument("--hz", type=int, choices=(60, 120), required=True)
    parser.add_argument("--seed-candidate-report")
    parser.add_argument("--seed-resume-report")
    args = parser.parse_args()
    result = run(args)
    print("COPY_RESULT_BEGIN", flush=True)
    print(json.dumps({k: result[k] for k in ("status", "rocks", "solver", "timing", "execution")}
        | {"checks": result.get("checks"), "failure": result.get("failure"),
           "rejection": result.get("rejection"), "policy": result["policy"],
           "transition": result["transition"], "seed": result.get("seed"),
           "report": str(Path(args.output).resolve()), "report_exists": Path(args.output).is_file(),
           "runner_exit_code": 0 if result["status"] == "candidate_short_hold_passed" else 1,
           "source": result.get("source_checkpoint"), "snapshots": result.get("diagnostic_snapshots"),
           "all_transition_frames_strict": result.get("all_transition_frames_strict")}, indent=2), flush=True)
    print("COPY_RESULT_END", flush=True)
    raise SystemExit(0 if result["status"] == "candidate_short_hold_passed" else 1)
