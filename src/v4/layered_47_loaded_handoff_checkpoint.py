#!/usr/bin/env python3
"""Immutable development handoff for a qualified loaded leave-pile state.

The file is intentionally separate from the source diagnostic snapshot.  It
can only become eligible for S07 development after a new-process restore and
short zero-command hold audit; it never grants training or final full-chain
qualification by itself.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import warp as wp
import warp.fem as fem
from newton.solvers import SolverImplicitMPM

import layered_47_interactive_proxy_checkpoint as sparse
from layered_47_sparse_approach_handoff_checkpoint import (
    write_checkpoint_atomic, write_metadata_atomic,
)
from layered_mound_checkpoint import checkpoint_identity


CHECKPOINT_SCHEMA = "excavation_layered_47_loaded_handoff_checkpoint_v1"
CHECKPOINT_STATUS = "loaded_handoff_saved_unaudited"
CONTROLLER_FIELDS = (
    "controller_target", "controller_target_velocity",
    "bucket_integral_effort",
)
SHA256_FIELDS = (
    "source_snapshot_sha256", "source_snapshot_metadata_sha256",
    "manual_report_sha256", "posthoc_measurement_report_sha256",
    "approach_handoff_sha256", "hencky_candidate_sha256",
    "bucket_integral_header_sha256", "manual_runner_sha256",
)
PROVENANCE_FIELDS = (*SHA256_FIELDS, "source_manual_status", "source_domain")


def _sha256(name: str, value: object) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"loaded_handoff_sha256:{name}")


def validate_controller_state(state: dict) -> dict:
    if tuple(sorted(state)) != tuple(sorted(CONTROLLER_FIELDS)):
        raise ValueError("loaded_handoff_controller_fields")
    result = {}
    for name in CONTROLLER_FIELDS:
        value = np.asarray(state[name], dtype=np.float64)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError(f"loaded_handoff_controller_state:{name}")
        result[name] = value.tolist()
    integral = np.asarray(result["bucket_integral_effort"])
    if np.any(integral[:3] != 0.0) or abs(float(integral[3])) > 120000.0:
        raise ValueError("loaded_handoff_integral_envelope")
    return result


def validate_arrays(arrays: dict[str, np.ndarray], particle_count: int,
                    rock_count: int) -> None:
    sparse._validate(arrays, particle_count, rock_count)
    if float(arrays["simulation_time_s"][0]) <= 0.0:
        raise ValueError("loaded_handoff_simulation_time")


@contextmanager
def restored_sparse_grid_topology(arrays: dict[str, np.ndarray]):
    """Seed one fixture construction from the saved Nanogrid cell topology."""
    cells = np.asarray(arrays["solver_grid_cell_ijk"], dtype=np.int32)
    transform = np.asarray(
        arrays["solver_grid_transform"], dtype=np.float32).reshape(3, 3)
    original = SolverImplicitMPM._allocate_grid
    used = {"count": 0}

    def allocate(solver, positions, particle_flags, voxel_size,
                 temporary_store, padding_voxels=0):
        if used["count"] == 0 and solver.grid_type == "sparse":
            used["count"] += 1
            points = wp.array(cells, dtype=wp.vec3i, device=positions.device)
            volume = wp.Volume.allocate_by_voxels(
                points, voxel_size=None, transform=transform,
                device=positions.device)
            return fem.Nanogrid(volume, temporary_store=temporary_store)
        return original(
            solver, positions, particle_flags, voxel_size, temporary_store,
            padding_voxels)

    SolverImplicitMPM._allocate_grid = allocate
    try:
        yield used
        if used["count"] != 1:
            raise RuntimeError("loaded_handoff_saved_grid_not_consumed")
    finally:
        SolverImplicitMPM._allocate_grid = original


def metadata_document(identity: dict, arrays: dict[str, np.ndarray],
                      controller_state: dict, provenance: dict) -> dict:
    validate_arrays(arrays, len(arrays["particle_q"]), len(arrays["rigid_qpos"]))
    controller = validate_controller_state(controller_state)
    if tuple(sorted(provenance)) != tuple(sorted(PROVENANCE_FIELDS)):
        raise ValueError("loaded_handoff_provenance_fields")
    for name in SHA256_FIELDS:
        _sha256(name, provenance[name])
    if provenance["source_manual_status"] != "manual_contact_pilot_recorded":
        raise ValueError("loaded_handoff_source_manual_status")
    if provenance["source_domain"] != "local_diagnostic_G3_loaded_leave_pile":
        raise ValueError("loaded_handoff_source_domain")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": CHECKPOINT_STATUS,
        "checkpoint_file": Path(identity["path"]).name,
        "checkpoint_sha256": identity["sha256"],
        "particle_count": int(len(arrays["particle_q"])),
        "rock_count": int(len(arrays["rigid_qpos"])),
        "collider_body_count": int(len(arrays["rigid_qpos"]) + 1),
        "simulation_time_s": float(arrays["simulation_time_s"][0]),
        "solver_grid_type": "sparse",
        "solver_voxel_size_m": 0.1,
        "solver_grid_padding": 2,
        "grid_reference_rebased_to_loaded_endpoint": True,
        "grid_reference_is_physics_state": False,
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "controller_state_saved": True,
        "controller_state": controller,
        "independent_restore_audit_passed": False,
        "s07_development_handoff_eligible": False,
        "final_full_chain_evidence": False,
        "training_eligible": False,
        "formal_a30_equivalent": False,
        **provenance,
    }


def load_checkpoint_readonly(
        checkpoint_path: str | Path, metadata_path: str | Path,
        expected_provenance: dict | None = None,
        expected_particle_count: int = 256441,
        expected_rock_count: int = 47,
        ) -> tuple[dict[str, np.ndarray], dict, dict]:
    checkpoint = Path(checkpoint_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    before = checkpoint_identity(checkpoint)
    metadata_before = checkpoint_identity(metadata_file)
    document = json.loads(metadata_file.read_text(encoding="utf-8"))
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "status": CHECKPOINT_STATUS,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": before["sha256"],
        "particle_count": expected_particle_count,
        "rock_count": expected_rock_count,
        "collider_body_count": expected_rock_count + 1,
        "solver_grid_type": "sparse",
        "solver_voxel_size_m": 0.1,
        "solver_grid_padding": 2,
        "grid_reference_rebased_to_loaded_endpoint": True,
        "grid_reference_is_physics_state": False,
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "controller_state_saved": True,
        "independent_restore_audit_passed": False,
        "s07_development_handoff_eligible": False,
        "final_full_chain_evidence": False,
        "training_eligible": False,
        "formal_a30_equivalent": False,
    }
    for name, expected in required.items():
        if document.get(name) != expected:
            raise ValueError(
                f"loaded_handoff_metadata:{name}:"
                f"{document.get(name)!r}:{expected!r}")
    controller = validate_controller_state(document.get("controller_state", {}))
    provenance = {name: document.get(name) for name in PROVENANCE_FIELDS}
    for name in SHA256_FIELDS:
        _sha256(name, provenance[name])
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("loaded_handoff_provenance_mismatch")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_arrays(arrays, expected_particle_count, expected_rock_count)
    if float(arrays["simulation_time_s"][0]) != float(document["simulation_time_s"]):
        raise ValueError("loaded_handoff_metadata_time")
    if checkpoint_identity(checkpoint) != before:
        raise RuntimeError("loaded_handoff_checkpoint_changed")
    if checkpoint_identity(metadata_file) != metadata_before:
        raise RuntimeError("loaded_handoff_metadata_changed")
    document["controller_state"] = controller
    return arrays, document, before


__all__ = [
    "CHECKPOINT_SCHEMA", "CHECKPOINT_STATUS", "CONTROLLER_FIELDS",
    "PROVENANCE_FIELDS", "SHA256_FIELDS", "load_checkpoint_readonly",
    "metadata_document", "validate_arrays", "validate_controller_state",
    "restored_sparse_grid_topology",
    "write_checkpoint_atomic", "write_metadata_atomic",
]
