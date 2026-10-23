#!/usr/bin/env python3
"""Independent zero-command ownership gate for the S07 truck colliders."""

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

import audit_layered_47_loaded_handoff as loaded
import layered_47_loaded_handoff_checkpoint as handoff
from layered_47_bucket_integral_candidate import bucket_integral_bridge
from layered_47_optimized_bridge import optimized_bridge
import layered_47_rigid_frequency_candidate as rigid_frequency
import layered_47_truck_physics_candidate as truck
from layered_excavation_approach_planner import _load_kinematics
from layered_mound_checkpoint import checkpoint_identity
import run_layered_47_sparse_manual_contact_episode as manual
import run_layered_47_sparse_motion as motion
from soil_hencky_update_guard import installed_material_guard


HZ = 60
RIGID_INTERNAL_HZ = 4800


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "truck-physics-gate.json"
    report = {
        "schema": "layered_47_truck_physics_static_gate_v1",
        "status": "initializing",
        "scope": {
            "transport_unload_development_eligible": False,
            "complete_single_scoop_qualified": False,
            "final_full_chain_evidence": False,
            "training_eligible": False,
        },
        "protocol": {
            "zero_step_identity": True,
            "zero_command": True,
            "hold_frames": 3,
            "hz": HZ,
            "rigid_internal_hz": RIGID_INTERNAL_HZ,
            "soil_truck_owner": "Newton",
            "rock_truck_owner": "MuJoCo",
            "inventory_observer": "TruckReceiver",
        },
    }
    _write(report_path, report)
    bridge = None
    rows = []
    started = time.perf_counter()
    try:
        root = Path(args.repository_root).resolve()
        truck_root = Path(args.truck_repository_root).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        metadata_path = Path(args.metadata).resolve()
        loaded_audit_path = Path(args.loaded_audit).resolve()
        loaded_audit_sha = motion.probe.bench.sha256(loaded_audit_path)
        loaded_audit = json.loads(loaded_audit_path.read_text(encoding="utf-8"))
        arrays, metadata, before = handoff.load_checkpoint_readonly(
            checkpoint_path, metadata_path)
        metadata_before = checkpoint_identity(metadata_path)
        if (loaded_audit.get("status") != "loaded_handoff_restore_hold_passed"
                or not all(loaded_audit.get("checks", {}).values())
                or loaded_audit.get("source_checkpoint") != before
                or loaded_audit.get("source_metadata") != metadata_before
                or loaded_audit.get("scope", {}).get(
                    "s07_development_handoff_eligible") is not True):
            raise ValueError("truck_gate_loaded_audit_binding")

        truck_document, fixtures = truck.load_candidate(
            args.truck_config, truck_root)
        acceptance = truck_document["acceptance"]
        particle_clearance = truck.clearance(arrays["particle_q"], fixtures)
        rock_center_clearance = truck.clearance(
            arrays["rigid_qpos"][:, :3], fixtures)

        loader, identity = motion.probe.select(47)
        base = loader(Path(args.config), root)
        base["input_checkpoint"] = json.loads(
            Path(args.local_source_config).read_text(encoding="utf-8"))[
                "input_checkpoint"]
        manifest_path = root / base["physical_geometry"]["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [record for record in manifest["rocks"]
                   if record["layer_id"] in identity["layers"]]
        if [record["rock_id"] for record in records] != identity["expected_ids"]:
            raise ValueError("truck_gate_manifest_order")
        _, proxy = motion.probe.p47._load_bound_json(
            root, base["open_bucket_proxy"], "proxy")
        approach = json.loads(Path(args.approach_report).read_text(
            encoding="utf-8"))
        if approach.get("handoff_checkpoint", {}).get("sha256") != metadata[
                "approach_handoff_sha256"]:
            raise ValueError("truck_gate_approach_binding")
        base["probe"] = {
            **base["probe"], "output_dt_s": 1.0 / HZ,
            "mujoco_internal_substeps": RIGID_INTERNAL_HZ // HZ,
        }
        assembly, chain = _load_kinematics(
            args.kinematics_python_dir, args.kinematics_model,
            args.kinematics_mesh_dir)
        projector = manual.ContactCommandProjector(chain)
        inventory = json.loads(Path(args.inventory_contract).read_text(
            encoding="utf-8"))
        with np.load(Path(args.front_rim), allow_pickle=False) as archive:
            front = archive["front"].copy()
        polygon = np.asarray(inventory["bucket"]["polygon_xz_m"])
        y_limits = np.asarray(inventory["bucket"]["y_limits_m"])
        source_pose = assembly.assemble(
            arrays["mechanism_qpos"], include_meshes=False)[1]["bucket"]
        source_local = ((arrays["particle_q"] - source_pose[:3, 3])
                        @ source_pose[:3, :3])
        source_payload = loaded.cavity_mask(
            source_local, polygon, y_limits, front)
        if int(source_payload.sum()) != 6494:
            raise ValueError("truck_gate_payload_identity")

        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with ExitStack() as contexts:
            optimization = contexts.enter_context(optimized_bridge(
                args.mechanism_header_dir, output / "optimized-build"))
            integral_policy = contexts.enter_context(bucket_integral_bridge(
                True, output / "bucket-integral-build"))
            installed_fixtures = contexts.enter_context(
                truck.installed_candidate(args.truck_config, truck_root))
            import soil_hencky_update_candidate as hencky
            if motion.probe.bench.sha256(Path(hencky.__file__)) != metadata[
                    "hencky_candidate_sha256"]:
                raise ValueError("truck_gate_hencky_binding")
            contexts.enter_context(hencky.installed_candidate())
            raw = contexts.enter_context(tempfile.TemporaryDirectory(
                prefix="truck-physics-gate-"))
            with handoff.restored_sparse_grid_topology(arrays) as grid_restore:
                fixture, held, restored = motion.audit.restore_candidate_runtime(
                    args, base, root, records, arrays, proxy,
                    approach["solver_profile"], Path(raw),
                    metadata["controller_state"], installed_fixtures)
            bridge = rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, RIGID_INTERNAL_HZ)
            material_guard = contexts.enter_context(
                installed_material_guard(fixture.solver))
            model_candidate = restored["runtime_build"]["model"].get(
                "truck_physics_candidate", {})
            zero_step = {
                "restore_exact": bool(restored["exact"]),
                "errors": restored["errors"],
                "saved_sparse_grid_constructor_calls": grid_restore["count"],
                "newton_truck_fixture_names": sorted(
                    fixture.truck_collider_ids),
                "newton_truck_collider_ids": {
                    name: value.tolist() for name, value in
                    fixture.truck_collider_ids.items()},
                "newton_ground_collider_ids":
                    fixture.ground_collider_ids.tolist(),
                "mujoco_truck_candidate": model_candidate,
                "controller_target": bridge.state["controller_target"],
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
            for frame in range(int(acceptance["hold_frames"])):
                bridge.command = np.zeros(4, dtype=np.float64)
                material_guard.prepare_frame()
                row, _ = motion.probe.bench._advance_frame(
                    fixture, bridge, base, frame + 1)
                row["hencky_material_guard"] = material_guard.summarize(
                    fixture.state)
                row.update(
                    applied_command_si=[0.0] * 4,
                    rigid_contact_substeps={
                        name: int(value) for name, value in
                        bridge.state.get("contact_substeps", {}).items()},
                    joint_bounds_ok=projector.physical_bounds_ok(np.asarray(
                        bridge.state["mechanism_qpos"])),
                    bucket_mpm_wrench_finite=all(
                        np.isfinite(np.asarray(value)).all()
                        for value in captured["wrenches"]["bucket"].values()),
                )
                row["finite"] = bool(
                    row["finite"]
                    and motion.probe.bench._finite_tree(bridge.state)
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
            final_cavity = loaded.cavity_mask(
                final_local, polygon, y_limits, front)
            retained = int(np.count_nonzero(source_payload & final_cavity))
            source_count = int(source_payload.sum())
            target_delta = float(np.max(np.abs(
                np.asarray(bridge.state["controller_target"])
                - np.asarray(metadata["controller_state"][
                    "controller_target"]))))
            integral_delta = float(np.max(np.abs(
                np.asarray(bridge.state["bucket_integral_effort"])
                - np.asarray(metadata["controller_state"][
                    "bucket_integral_effort"]))))
            hold = {
                "frames": len(rows),
                "source_cavity_soil_particles": source_count,
                "same_particles_still_in_cavity": retained,
                "retention_fraction": retained / source_count,
                "maximum_newton_truck_contact_samples": max(
                    (int(row["truck_mpm_contact_samples"]) for row in rows),
                    default=0),
                "maximum_mujoco_rock_truck_contact_substeps": max(
                    (int(row["rigid_contact_substeps"]["rock_truck"])
                     for row in rows), default=0),
                "controller_target_max_abs_delta": target_delta,
                "bucket_integral_effort_max_abs_delta_N": integral_delta,
            }
            expected_names = sorted(truck.FIXTURE_NAMES)
            checks = {
                "loaded_handoff_audit_passed": True,
                "particle_initially_outside_truck":
                    particle_clearance["inside_any_fixture_count"] == 0,
                "particle_initial_clearance":
                    particle_clearance["minimum_point_center_clearance_m"] >=
                    float(acceptance["minimum_initial_particle_clearance_m"]),
                "rock_centers_initially_outside_truck":
                    rock_center_clearance["inside_any_fixture_count"] == 0,
                "rock_center_initial_clearance":
                    rock_center_clearance["minimum_point_center_clearance_m"] >=
                    float(acceptance[
                        "minimum_initial_rock_center_clearance_m"]),
                "zero_step_restore_exact": zero_step["restore_exact"],
                "saved_sparse_grid_consumed": grid_restore["count"] == 1,
                "newton_has_exact_truck_fixtures":
                    zero_step["newton_truck_fixture_names"] == expected_names,
                "mujoco_has_exact_truck_fixtures":
                    sorted(model_candidate.get("fixture_names", [])) ==
                    expected_names,
                "contact_ownership_unique":
                    truck_document["contact_ownership"][
                        "duplicate_soil_contact_backend"] is False
                    and truck_document["contact_ownership"][
                        "duplicate_rock_contact_backend"] is False,
                "all_hold_frames_executed": len(rows) == int(
                    acceptance["hold_frames"]),
                "all_hold_frames_hard_safe": bool(rows) and all(
                    row["hard_guard"] for row in rows),
                "newton_truck_contacts_zero_before_transport":
                    hold["maximum_newton_truck_contact_samples"] <= int(
                        acceptance["maximum_initial_truck_contact_samples"]),
                "mujoco_truck_contacts_zero_before_transport":
                    hold["maximum_mujoco_rock_truck_contact_substeps"] <= int(
                        acceptance[
                            "maximum_initial_rock_truck_contact_substeps"]),
                "same_payload_retained": hold["retention_fraction"] >= 0.99,
                "controller_reference_preserved": target_delta <= 1.0e-12,
                "bucket_integral_effort_preserved": integral_delta <= 1.0e-9,
                "loaded_audit_unchanged":
                    motion.probe.bench.sha256(loaded_audit_path) ==
                    loaded_audit_sha,
                "source_checkpoint_unchanged":
                    checkpoint_identity(checkpoint_path) == before,
                "source_metadata_unchanged":
                    checkpoint_identity(metadata_path) == metadata_before,
            }
            passed = all(checks.values())
            report.update(
                status=("truck_physics_static_gate_passed" if passed else
                        "truck_physics_static_gate_rejected"),
                source_checkpoint=before,
                source_metadata=metadata_before,
                loaded_handoff_audit={
                    "path": str(loaded_audit_path),
                    "sha256": loaded_audit_sha,
                },
                truck_candidate={
                    "config_path": str(Path(args.truck_config).resolve()),
                    "config_sha256": truck.sha256(args.truck_config),
                    "fixtures": fixtures,
                    "particle_initial_clearance": particle_clearance,
                    "rock_center_initial_clearance": rock_center_clearance,
                },
                zero_step=zero_step,
                hold=hold,
                checks=checks,
                trace=rows,
                optimization=optimization.copy(),
                bucket_integral_candidate=integral_policy.copy(),
            )
            report["scope"]["transport_unload_development_eligible"] = passed
            bridge.close(kill=not passed)
            bridge = None
    except Exception as error:
        report.update(status="truck_physics_static_gate_failed", failure={
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
            "output", "repository-root", "truck-repository-root", "config",
            "local-source-config",
            "checkpoint", "metadata", "loaded-audit", "approach-report",
            "truck-config", "device", "mujoco-include",
            "mujoco-library-dir", "mechanism-header-dir",
            "controller-header-dir", "kinematics-python-dir",
            "kinematics-model", "kinematics-mesh-dir", "inventory-contract",
            "front-rim"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cxx", default="g++")
    result = run(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in (
        "status", "scope", "checks", "zero_step", "hold", "failure")},
        indent=2))
    raise SystemExit(0 if result["status"] ==
                     "truck_physics_static_gate_passed" else 1)
