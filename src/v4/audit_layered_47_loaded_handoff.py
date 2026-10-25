#!/usr/bin/env python3
"""New-process zero-step identity and three-frame hold audit for a loaded handoff."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

import layered_47_loaded_handoff_checkpoint as handoff
from layered_47_bucket_integral_candidate import bucket_integral_bridge
from layered_47_optimized_bridge import optimized_bridge
import layered_47_rigid_frequency_candidate as rigid_frequency
from layered_excavation_approach_planner import _load_kinematics
from layered_mound_checkpoint import checkpoint_identity
import run_layered_47_sparse_manual_contact_episode as manual
import run_layered_47_sparse_motion as motion
from soil_hencky_update_guard import installed_material_guard


HZ = 60
RIGID_INTERNAL_HZ = 4800
HOLD_FRAMES = 3


def cavity_mask(points: np.ndarray, polygon_xz: np.ndarray,
                y_limits: np.ndarray, front_rim_xz: np.ndarray) -> np.ndarray:
    """Dependency-free copy of the fixed CAD cavity predicate."""
    q = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    front = np.asarray(front_rim_xz, dtype=np.float64)
    x, z = q[:, 0], q[:, 2]
    inside = np.zeros(len(q), dtype=bool)
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(second[1] - first[1]) > 1.0e-14:
            inside ^= ((first[1] > z) != (second[1] > z)) & (
                x < ((second[0] - first[0]) * (z - first[1])
                     / (second[1] - first[1]) + first[0]))
    return (inside & (q[:, 1] >= y_limits[0])
            & (q[:, 1] <= y_limits[1])
            & (x >= np.interp(z, front[:, 1], front[:, 0],
                              left=-10.0, right=-10.0)))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "independent-restore-audit.json"
    report = {
        "schema": "layered_47_loaded_handoff_restore_audit_v1",
        "status": "initializing",
        "scope": {
            "s07_development_handoff_eligible": False,
            "final_full_chain_evidence": False,
            "training_eligible": False,
        },
        "protocol": {"zero_step_identity": True, "zero_command": True,
                     "hold_frames": HOLD_FRAMES, "hz": HZ,
                     "rigid_internal_hz": RIGID_INTERNAL_HZ},
    }
    _write(report_path, report)
    bridge = None
    rows = []
    started = time.perf_counter()
    try:
        root = Path(args.repository_root).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        metadata_path = Path(args.metadata).resolve()
        packaging_path = Path(args.packaging_report).resolve()
        manual_path = Path(args.manual_report).resolve()
        posthoc_path = Path(args.posthoc_report).resolve()
        approach_path = Path(args.approach_report).resolve()
        local_source = Path(args.local_source_config).resolve()
        packaging = json.loads(packaging_path.read_text(encoding="utf-8"))
        packaging_sha256 = motion.probe.bench.sha256(packaging_path)
        source_manual = json.loads(manual_path.read_text(encoding="utf-8"))
        approach = json.loads(approach_path.read_text(encoding="utf-8"))
        arrays, metadata, before = handoff.load_checkpoint_readonly(
            checkpoint_path, metadata_path)
        metadata_before = checkpoint_identity(metadata_path)
        if (packaging.get("status") != "loaded_handoff_saved_unaudited"
                or packaging.get("handoff_checkpoint") != before
                or packaging.get("handoff_metadata") != metadata_before
                or metadata["manual_report_sha256"] !=
                    motion.probe.bench.sha256(manual_path)
                or metadata["posthoc_measurement_report_sha256"] !=
                    motion.probe.bench.sha256(posthoc_path)
                or source_manual.get("status") !=
                    "manual_contact_pilot_recorded"
                or not all(source_manual.get("checks", {}).values())
                or approach.get("status") !=
                    "sparse_approach_handoff_saved_unaudited"):
            raise ValueError("loaded_handoff_audit_lineage")

        loader, identity = motion.probe.select(47)
        base = loader(Path(args.config), root)
        base["input_checkpoint"] = json.loads(
            local_source.read_text(encoding="utf-8"))["input_checkpoint"]
        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity["expected_ids"]:
            raise ValueError("loaded_handoff_audit_manifest_order")
        _, proxy = motion.probe.p47._load_bound_json(
            root, base["open_bucket_proxy"], "proxy")
        base["probe"] = {**base["probe"], "output_dt_s": 1.0 / HZ,
                         "mujoco_internal_substeps": RIGID_INTERNAL_HZ // HZ}
        assembly, chain = _load_kinematics(
            args.kinematics_python_dir, args.kinematics_model,
            args.kinematics_mesh_dir)
        projector = manual.ContactCommandProjector(chain)

        contract = json.loads(Path(args.inventory_contract).read_text(
            encoding="utf-8"))
        with np.load(Path(args.front_rim), allow_pickle=False) as front_archive:
            front = front_archive["front"].copy()
        polygon = np.asarray(contract["bucket"]["polygon_xz_m"])
        y_limits = np.asarray(contract["bucket"]["y_limits_m"])
        source_pose = assembly.assemble(
            arrays["mechanism_qpos"], include_meshes=False)[1]["bucket"]
        source_local = ((arrays["particle_q"] - source_pose[:3, 3])
                        @ source_pose[:3, :3])
        source_payload = cavity_mask(source_local, polygon, y_limits, front)
        if int(source_payload.sum()) != 6494:
            raise ValueError("loaded_handoff_source_payload_identity")

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with ExitStack() as contexts:
            optimization = contexts.enter_context(optimized_bridge(
                args.mechanism_header_dir, output / "optimized-build"))
            integral_policy = contexts.enter_context(bucket_integral_bridge(
                True, output / "bucket-integral-build"))
            import soil_hencky_update_candidate as hencky
            if motion.probe.bench.sha256(Path(hencky.__file__)) != metadata[
                    "hencky_candidate_sha256"]:
                raise ValueError("loaded_handoff_hencky_binding")
            contexts.enter_context(hencky.installed_candidate())
            raw = contexts.enter_context(tempfile.TemporaryDirectory(
                prefix="loaded-handoff-audit-"))
            with handoff.restored_sparse_grid_topology(arrays) as grid_restore:
                fixture, held, restored = motion.audit.restore_candidate_runtime(
                    args, base, root, records, arrays, proxy,
                    approach["solver_profile"], Path(raw),
                    metadata["controller_state"])
            restored["saved_sparse_grid_constructor_calls"] = grid_restore["count"]
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            material_guard = contexts.enter_context(
                installed_material_guard(fixture.solver))
            zero_step = {
                "restore_exact": bool(restored["exact"]),
                "errors": restored["errors"],
                "saved_sparse_grid_constructor_calls":
                    restored["saved_sparse_grid_constructor_calls"],
                "saved_sparse_grid_cell_count": int(len(
                    arrays["solver_grid_cell_ijk"])),
                "saved_sparse_grid_node_count": int(len(
                    arrays["solver_grid_node_ijk"])),
                "controller_target": bridge.state["controller_target"],
                "controller_target_velocity":
                    bridge.state["controller_target_velocity"],
                "bucket_integral_effort":
                    bridge.state["bucket_integral_effort"],
            }
            captured = {}
            original_step = fixture.step

            def recording_step(dt, origins):
                measured, particles = original_step(dt, origins)
                captured["wrenches"] = measured
                return measured, particles

            fixture.step = recording_step
            initial_bucket = np.asarray(
                bridge.state["bucket_origin"], dtype=np.float64)
            initial_mechanism = np.asarray(
                bridge.state["mechanism_qpos"], dtype=np.float64)
            initial_particles = arrays["particle_q"].copy()
            for frame in range(HOLD_FRAMES):
                bridge.command = np.zeros(4, dtype=np.float64)
                material_guard.prepare_frame()
                row, _ = motion.probe.bench._advance_frame(
                    fixture, bridge, base, frame + 1)
                row["hencky_material_guard"] = material_guard.summarize(
                    fixture.state)
                qpos = np.asarray(bridge.state["mechanism_qpos"])
                row.update(
                    applied_command_si=[0.0] * 4,
                    rigid_contact_substeps={name: int(value) for name, value in
                        bridge.state.get("contact_substeps", {}).items()},
                    joint_bounds_ok=projector.physical_bounds_ok(qpos),
                    bucket_mpm_wrench_finite=all(np.isfinite(value).all()
                        for item in captured["wrenches"]["bucket"].values()
                        for value in [np.asarray(item)]),
                )
                row["finite"] = bool(
                    row["finite"] and motion.probe.bench._finite_tree(bridge.state)
                    and row["hencky_material_guard"]["safe"])
                row["hard_guard_checks"] = manual.hard_guard_checks(row)
                row["hard_guard"] = all(row["hard_guard_checks"].values())
                rows.append(row)
                if not row["hard_guard"]:
                    break

            final_pose = assembly.assemble(
                np.asarray(bridge.state["mechanism_qpos"]),
                include_meshes=False)[1]["bucket"]
            final_particles = fixture.state.particle_q.numpy()
            final_local = ((final_particles - final_pose[:3, 3])
                           @ final_pose[:3, :3])
            final_cavity = cavity_mask(final_local, polygon, y_limits, front)
            retained = int(np.count_nonzero(source_payload & final_cavity))
            source_count = int(source_payload.sum())
            local_delta = np.linalg.norm(
                final_local[source_payload] - source_local[source_payload], axis=1)
            particle_delta = np.linalg.norm(
                final_particles - initial_particles, axis=1)
            hold = {
                "frames": len(rows),
                "source_cavity_soil_particles": source_count,
                "same_particles_still_in_cavity": retained,
                "retention_fraction": retained / source_count,
                "payload_local_displacement_max_m": float(local_delta.max()),
                "payload_local_displacement_p99_m": float(np.quantile(
                    local_delta, 0.99)),
                "all_particle_displacement_max_m": float(particle_delta.max()),
                "bucket_origin_displacement_m": float(np.linalg.norm(
                    np.asarray(bridge.state["bucket_origin"]) - initial_bucket)),
                "mechanism_qpos_max_abs_delta": float(np.max(np.abs(
                    np.asarray(bridge.state["mechanism_qpos"])
                    - initial_mechanism))),
                "final_controller_target": bridge.state["controller_target"],
                "final_controller_target_velocity":
                    bridge.state["controller_target_velocity"],
                "final_bucket_integral_effort":
                    bridge.state["bucket_integral_effort"],
            }
            hold["controller_target_max_abs_delta"] = float(np.max(np.abs(
                np.asarray(hold["final_controller_target"], dtype=np.float64)
                - np.asarray(metadata["controller_state"][
                    "controller_target"], dtype=np.float64))))
            hold["bucket_integral_effort_max_abs_delta_N"] = float(np.max(
                np.abs(np.asarray(hold["final_bucket_integral_effort"],
                                  dtype=np.float64)
                       - np.asarray(metadata["controller_state"][
                           "bucket_integral_effort"], dtype=np.float64))))
            diagnostic = motion.probe.snapshots.capture_arrays(
                fixture, bridge.state,
                float(arrays["simulation_time_s"][0]) + len(rows) / HZ)
            diagnostic_path = output / "hold-terminal.diagnostic.npz"
            diagnostic_identity = motion.probe.snapshots.write_checkpoint_atomic(
                diagnostic_path, diagnostic)
            diagnostic_metadata = {
                "schema": "layered_47_loaded_handoff_hold_diagnostic_v1",
                "status": "diagnostic_only_not_a_continuation_checkpoint",
                "source": before, "frames": len(rows),
                "training_eligible": False, **diagnostic_identity,
            }
            _write(Path(str(diagnostic_path) + ".json"), diagnostic_metadata)
            checks = {
                "zero_step_restore_exact": zero_step["restore_exact"],
                "all_hold_frames_executed": len(rows) == HOLD_FRAMES,
                "all_hold_frames_hard_safe": bool(rows) and all(
                    row["hard_guard"] for row in rows),
                "same_payload_retained": hold["retention_fraction"] >= 0.99,
                "payload_local_motion_bounded":
                    hold["payload_local_displacement_max_m"] <= 0.03,
                "bucket_motion_continuous":
                    hold["bucket_origin_displacement_m"] <= 0.02,
                "controller_reference_preserved":
                    hold["controller_target_max_abs_delta"] <= 1.0e-12,
                "bucket_integral_effort_preserved":
                    hold["bucket_integral_effort_max_abs_delta_N"] <= 1.0e-9,
                "source_checkpoint_unchanged":
                    checkpoint_identity(checkpoint_path) == before,
                "source_metadata_unchanged":
                    checkpoint_identity(metadata_path) == metadata_before,
                "packaging_report_unchanged":
                    motion.probe.bench.sha256(packaging_path) ==
                    packaging_sha256,
            }
            report.update(
                status=("loaded_handoff_restore_hold_passed"
                        if all(checks.values()) else
                        "loaded_handoff_restore_hold_rejected"),
                source_checkpoint=before,
                source_metadata=metadata_before,
                zero_step=zero_step,
                hold=hold,
                checks=checks,
                trace=rows,
                optimization=optimization.copy(),
                bucket_integral_candidate=integral_policy.copy(),
                diagnostic_snapshot=diagnostic_identity,
            )
            report["scope"]["s07_development_handoff_eligible"] = bool(
                all(checks.values()))
            bridge.close(kill=not all(checks.values()))
            bridge = None
    except Exception as error:
        report.update(status="loaded_handoff_restore_hold_failed", failure={
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        })
    finally:
        if bridge is not None:
            try:
                bridge.close(kill=True)
            except Exception as error:
                report["cleanup_error"] = str(error)
        report["execution"] = {
            "frames": len(rows), "wall_seconds": time.perf_counter() - started}
        _write(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
            "output", "repository-root", "config", "local-source-config",
            "checkpoint", "metadata", "packaging-report", "manual-report",
            "posthoc-report", "approach-report", "device", "mujoco-include",
            "mujoco-library-dir", "mechanism-header-dir",
            "controller-header-dir", "kinematics-python-dir",
            "kinematics-model", "kinematics-mesh-dir",
            "inventory-contract", "front-rim"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cxx", default="g++")
    result = run(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in (
        "status", "scope", "checks", "zero_step", "hold", "failure")}, indent=2))
    raise SystemExit(0 if result["status"] ==
                     "loaded_handoff_restore_hold_passed" else 1)
