#!/usr/bin/env python3
"""Immutable sparse-grid handoff saved after the non-contact approach.

The source remains diagnostic-domain data.  Saving this state never grants
contact, collection, or training eligibility; a separate new-process restore
audit is required before a short CONTACT_HANDOFF experiment may be considered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

import layered_47_interactive_proxy_checkpoint as sparse
from layered_mound_checkpoint import checkpoint_identity


CHECKPOINT_SCHEMA = (
    "excavation_layered_47_sparse_approach_handoff_checkpoint_v1")
CHECKPOINT_STATUS = "sparse_approach_handoff_saved_unaudited"
SOURCE_DOMAIN = "local_47_rejected_ancestry_diagnostic_only"
SHA256_FIELDS = (
    "source_checkpoint_sha256",
    "source_checkpoint_metadata_sha256",
    "candidate_report_sha256",
    "resume_report_sha256",
    "preflight_config_sha256",
    "local_source_config_sha256",
    "approach_execution_report_sha256",
    "approach_runner_sha256",
    "plan_qpos_float64_sha256",
    "plan_velocity_float64_sha256",
    "observation_state_digest",
)
PROVENANCE_FIELDS = (
    *SHA256_FIELDS,
    "source_domain",
    "source_checkpoint_status",
    "hz",
    "rigid_internal_hz",
)


def capture_arrays(fixture, bridge_state: dict,
                   simulation_time_s: float) -> dict[str, np.ndarray]:
    arrays = sparse.capture_arrays(fixture, bridge_state, simulation_time_s)
    validate_arrays(arrays, len(arrays["particle_q"]),
                    len(arrays["rigid_qpos"]))
    return arrays


def validate_arrays(arrays: dict[str, np.ndarray], particle_count: int,
                    rock_count: int) -> None:
    sparse._validate(arrays, particle_count, rock_count)
    if float(arrays["simulation_time_s"][0]) <= 0.0:
        raise ValueError("sparse_approach_handoff_simulation_time")
    if arrays["solver_body_q_prev"].shape != (rock_count + 1, 7):
        raise ValueError("sparse_approach_handoff_bucket_history")


def _validate_sha256(name: str, value: object) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef"
                   for character in value)):
        raise ValueError(f"sparse_approach_handoff_sha256:{name}")


def write_checkpoint_atomic(path: str | Path,
                            arrays: dict[str, np.ndarray]) -> dict:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    validate_arrays(arrays, len(arrays["particle_q"]),
                    len(arrays["rigid_qpos"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp",
            delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return checkpoint_identity(target)


def metadata_document(identity: dict, arrays: dict[str, np.ndarray],
                      provenance: dict) -> dict:
    if tuple(sorted(provenance)) != tuple(sorted(PROVENANCE_FIELDS)):
        raise ValueError("sparse_approach_handoff_provenance_fields")
    for name in SHA256_FIELDS:
        _validate_sha256(name, provenance[name])
    if provenance["source_domain"] != SOURCE_DOMAIN:
        raise ValueError("sparse_approach_handoff_source_domain")
    if provenance["source_checkpoint_status"] != (
            "grid_candidate_diagnostic_only"):
        raise ValueError("sparse_approach_handoff_source_status")
    if provenance["hz"] != 60 or provenance["rigid_internal_hz"] != 4800:
        raise ValueError("sparse_approach_handoff_frequency")
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
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "independent_restore_audit_passed": False,
        "contact_handoff_eligible": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
        "formal_a30_equivalent": False,
        **provenance,
    }


def write_metadata_atomic(path: str | Path, document: dict) -> dict:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=target.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return checkpoint_identity(target)


def load_checkpoint_readonly(
        checkpoint_path: str | Path,
        metadata_path: str | Path,
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
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "independent_restore_audit_passed": False,
        "contact_handoff_eligible": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
        "formal_a30_equivalent": False,
    }
    for name, expected in required.items():
        if document.get(name) != expected:
            raise ValueError(
                f"sparse_approach_handoff_metadata:{name}:"
                f"{document.get(name)!r}:{expected!r}")
    provenance = {name: document.get(name) for name in PROVENANCE_FIELDS}
    metadata_document(before, {
        "particle_q": np.empty((int(document["particle_count"]), 3)),
        "rigid_qpos": np.empty((int(document["rock_count"]), 7)),
        "simulation_time_s": np.asarray([document["simulation_time_s"]]),
    }, provenance)
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("sparse_approach_handoff_provenance_mismatch")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_arrays(arrays, expected_particle_count, expected_rock_count)
    if float(arrays["simulation_time_s"][0]) != float(
            document["simulation_time_s"]):
        raise ValueError("sparse_approach_handoff_metadata_time")
    if checkpoint_identity(checkpoint) != before:
        raise RuntimeError("sparse_approach_handoff_checkpoint_changed")
    if checkpoint_identity(metadata_file) != metadata_before:
        raise RuntimeError("sparse_approach_handoff_metadata_changed")
    return arrays, document, before
