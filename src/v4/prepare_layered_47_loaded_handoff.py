#!/usr/bin/env python3
"""Package the D-129 final diagnostic state as an unaudited loaded handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

import layered_47_loaded_handoff_checkpoint as handoff
from layered_mound_checkpoint import checkpoint_identity
from build_mixed_material_capacity_preflight import sha256


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {
        "schema": "layered_47_loaded_handoff_packaging_v1",
        "status": "initializing",
        "scope": {
            "physics_advanced": False,
            "s07_development_handoff_eligible": False,
            "final_full_chain_evidence": False,
            "training_eligible": False,
        },
    }
    _write_json(report_path, report)
    try:
        snapshot = Path(args.snapshot).resolve()
        snapshot_metadata = Path(args.snapshot_metadata).resolve()
        manual_report_path = Path(args.manual_report).resolve()
        posthoc_path = Path(args.posthoc_report).resolve()
        manual = json.loads(manual_report_path.read_text(encoding="utf-8"))
        metadata = json.loads(snapshot_metadata.read_text(encoding="utf-8"))
        posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
        source_identity = checkpoint_identity(snapshot)
        if (manual.get("schema") != "layered_47_sparse_manual_contact_episode_v1"
                or manual.get("status") != "manual_contact_pilot_recorded"
                or manual.get("stop_reason") != "user_finished"
                or not all(manual.get("checks", {}).values())):
            raise ValueError("loaded_handoff_manual_report")
        listed = manual.get("diagnostic_snapshots", [])
        if not listed or any(
                listed[-1].get(key) != source_identity[key]
                for key in ("path", "sha256", "size_bytes", "mtime_ns")):
            raise ValueError("loaded_handoff_snapshot_binding")
        if (metadata.get("status") !=
                "diagnostic_only_not_resume_or_training_qualified"
                or metadata.get("stop_reason") != "user_finished"
                or metadata.get("sha256") != source_identity["sha256"]
                or sha256(snapshot_metadata) != listed[-1].get("metadata_sha256")):
            raise ValueError("loaded_handoff_snapshot_metadata")
        mixed = posthoc.get("load_geometry", {}).get(
            "mixed_soil_and_rock_0p0125m_samples", {})
        payload = posthoc.get("effective_payload", {})
        if (posthoc.get("schema") != "hencky_manual_single_scoop_posthoc_v1"
                or posthoc.get("status") !=
                    "posthoc_geometry_and_stability_audit_completed"
                or mixed.get("valid_estimate") is not True
                or mixed.get("bulk_touches_roi_boundary") is not False
                or payload.get("assessment") !=
                    "payload_lifted_from_initial_bank_and_retained_through_30_frame_hold"
                or posthoc.get("material_guard", {}).get(
                    "all_frames_safe") is not True):
            raise ValueError("loaded_handoff_posthoc_G3")
        with np.load(snapshot, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        # A sparse Nanogrid rebuilds its topology from this allocation seed.
        # The diagnostic snapshot retained the original prepoint seed even
        # though contact expanded the live grid.  Rebase only this constructor
        # input to the loaded endpoint; every evolving physics field remains
        # byte-for-byte copied from the source snapshot.
        source_grid_reference = arrays["grid_reference_particle_q"].copy()
        arrays["grid_reference_particle_q"] = arrays["particle_q"].copy()
        handoff.validate_arrays(arrays, 256441, 47)
        compact = metadata.get("controller_state_diagnostic_only", {})
        controller_state = {
            name: compact.get(name) for name in handoff.CONTROLLER_FIELDS}
        controller_state = handoff.validate_controller_state(controller_state)
        bindings = manual.get("input_bindings", {})
        approach = metadata.get("source_handoff", {})
        provenance = {
            "source_snapshot_sha256": source_identity["sha256"],
            "source_snapshot_metadata_sha256": sha256(snapshot_metadata),
            "manual_report_sha256": sha256(manual_report_path),
            "posthoc_measurement_report_sha256": sha256(posthoc_path),
            "approach_handoff_sha256": approach.get("sha256"),
            "hencky_candidate_sha256": manual[
                "hencky_update_candidate"]["candidate_source_sha256"],
            "bucket_integral_header_sha256": manual[
                "bucket_integral_candidate"]["header_sha256"],
            "manual_runner_sha256": manual["runner_sha256"],
            "source_manual_status": manual["status"],
            "source_domain": "local_diagnostic_G3_loaded_leave_pile",
        }
        if (bindings.get(str(Path(args.hencky_candidate).resolve())) !=
                provenance["hencky_candidate_sha256"]
                or sha256(Path(args.hencky_candidate)) !=
                provenance["hencky_candidate_sha256"]
                or bindings.get(str(Path(args.bucket_integral_header).resolve())) !=
                provenance["bucket_integral_header_sha256"]
                or sha256(Path(args.bucket_integral_header)) !=
                provenance["bucket_integral_header_sha256"]):
            raise ValueError("loaded_handoff_candidate_binding")
        target = output / "loaded-handoff.npz"
        identity = handoff.write_checkpoint_atomic(target, arrays)
        metadata_path = Path(str(target) + ".json")
        metadata_identity = handoff.write_metadata_atomic(
            metadata_path, handoff.metadata_document(
                identity, arrays, controller_state, provenance))
        checks = {
            "source_snapshot_unchanged": checkpoint_identity(snapshot) == source_identity,
            "all_source_checks_passed": all(manual["checks"].values()),
            "posthoc_G3_bound": True,
            "arrays_valid": True,
            "controller_state_valid": True,
            "physics_not_advanced": True,
            "grid_reference_rebased_without_advancing_physics": bool(
                not np.array_equal(source_grid_reference,
                                   arrays["grid_reference_particle_q"])),
            "handoff_still_unaudited": True,
        }
        report.update(
            status=("loaded_handoff_saved_unaudited"
                    if all(checks.values()) else "loaded_handoff_packaging_rejected"),
            source_snapshot=source_identity,
            source_snapshot_metadata={
                "path": str(snapshot_metadata), "sha256": sha256(snapshot_metadata)},
            manual_report={
                "path": str(manual_report_path), "sha256": sha256(manual_report_path)},
            posthoc_report={"path": str(posthoc_path), "sha256": sha256(posthoc_path)},
            handoff_checkpoint=identity,
            handoff_metadata=metadata_identity,
            controller_state=controller_state,
            provenance=provenance,
            checks=checks,
        )
    except Exception as error:
        report.update(status="loaded_handoff_packaging_failed", failure={
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        })
    _write_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
            "output", "snapshot", "snapshot-metadata", "manual-report",
            "posthoc-report", "hencky-candidate", "bucket-integral-header"):
        parser.add_argument("--" + name, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result.get(key) for key in (
        "status", "handoff_checkpoint", "handoff_metadata", "checks", "failure")},
        indent=2))
    raise SystemExit(0 if result["status"] ==
                     "loaded_handoff_saved_unaudited" else 1)
