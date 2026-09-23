#!/usr/bin/env python3
"""Full-state checkpoint contract at the 47-rock APPROACH/S04 boundary.

This checkpoint is intentionally distinct from the layered-mound construction
checkpoint.  The excavation fixture has one additional kinematic bucket body
and the MuJoCo mechanism has already moved away from ``operator-start``.
Consequently an exact handoff must retain the 48-body Newton history together
with the mechanism qpos/qvel; restoring only the original 47 rocks or applying
the reported goal qpos is not a continuation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from layered_mound_checkpoint import (
    CHECKPOINT_FIELDS as LAYERED_CHECKPOINT_FIELDS,
    CHECKPOINT_SCHEMA as SOURCE_CHECKPOINT_SCHEMA,
    checkpoint_arrays as layered_checkpoint_arrays,
    checkpoint_identity,
)


CHECKPOINT_SCHEMA = "excavation_layered_47_approach_handoff_checkpoint_v1"
CHECKPOINT_STATUS = "approach_handoff_checkpoint_saved_unaudited"
MECHANISM_FIELDS = ("mechanism_qpos", "mechanism_qvel")
CHECKPOINT_FIELDS = (*LAYERED_CHECKPOINT_FIELDS, *MECHANISM_FIELDS)
SHA256_FIELDS = (
    "source_checkpoint_sha256",
    "approach_report_sha256",
    "preflight_config_sha256",
    "approach_config_sha256",
    "plan_qpos_float64_sha256",
    "plan_velocity_float64_sha256",
    "observation_state_digest",
)
PROVENANCE_FIELDS = (
    *SHA256_FIELDS,
    "source_domain",
    "source_checkpoint_status",
)


def capture_arrays(fixture, bridge_state: dict,
                   simulation_time_s: float) -> dict[str, np.ndarray]:
    """Capture every state needed to continue from the non-contact prepoint."""
    arrays = layered_checkpoint_arrays(
        fixture, bridge_state, simulation_time_s)
    arrays.update({
        "mechanism_qpos": np.asarray(
            bridge_state["mechanism_qpos"], dtype=np.float64),
        "mechanism_qvel": np.asarray(
            bridge_state["mechanism_qvel"], dtype=np.float64),
    })
    return {name: np.asarray(value).copy()
            for name, value in arrays.items()}


def _validate_sha256(name: str, value: object) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"approach_handoff_metadata_sha256:{name}")


def validate_arrays(arrays: dict[str, np.ndarray], particle_count: int,
                    rock_count: int) -> None:
    if tuple(sorted(arrays)) != tuple(sorted(CHECKPOINT_FIELDS)):
        raise ValueError("approach_handoff_checkpoint_field_set")
    expected = {
        "particle_q": (particle_count, 3),
        "particle_qd": (particle_count, 3),
        "particle_f": (particle_count, 3),
        "particle_flags": (particle_count,),
        "particle_elastic_strain": (particle_count, 3, 3),
        "particle_transform": (particle_count, 3, 3),
        "particle_qd_grad": (particle_count, 3, 3),
        "particle_stress": (particle_count, 3, 3),
        "particle_Jp": (particle_count,),
        "rigid_ids": (rock_count,),
        "rigid_qpos": (rock_count, 7),
        "rigid_qvel": (rock_count, 6),
        "rigid_reported_linear_velocity": (rock_count, 3),
        "rigid_reported_angular_velocity": (rock_count, 3),
        "mechanism_qpos": (19,),
        "mechanism_qvel": (18,),
        "simulation_time_s": (1,),
        "grid_reference_particle_q": (particle_count, 3),
        "solver_grid_bounds_lo": (3,),
        "solver_grid_bounds_hi": (3,),
        "solver_grid_resolution": (3,),
        # 47 dynamic rocks plus the kinematic open-bucket collider.
        "solver_body_q_prev": (rock_count + 1, 7),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"approach_handoff_checkpoint_shape:{name}:"
                f"{arrays[name].shape}:{shape}")
    for name, width in (("solver_ws_impulse", 3),
                        ("solver_ws_stress", 6)):
        shape = arrays[name].shape
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != width:
            raise ValueError(
                f"approach_handoff_checkpoint_shape:{name}:{shape}")
    if (not np.issubdtype(arrays["solver_grid_resolution"].dtype, np.integer)
            or np.any(arrays["solver_grid_resolution"] <= 0)
            or np.any(arrays["solver_grid_bounds_hi"] <=
                      arrays["solver_grid_bounds_lo"])):
        raise ValueError("approach_handoff_checkpoint_fixed_grid")
    if not all(np.isfinite(value).all() for name, value in arrays.items()
               if name != "rigid_ids"):
        raise ValueError("approach_handoff_checkpoint_nonfinite")
    expected_ids = np.asarray(
        [f"r1-rock-{index:03d}" for index in range(rock_count)])
    if not np.array_equal(arrays["rigid_ids"].astype(str), expected_ids):
        raise ValueError("approach_handoff_checkpoint_rigid_ids")
    if float(arrays["simulation_time_s"][0]) <= 0.0:
        raise ValueError("approach_handoff_checkpoint_simulation_time")


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
    os.replace(temporary, target)
    return checkpoint_identity(target)


def metadata_document(identity: dict, arrays: dict[str, np.ndarray],
                      provenance: dict) -> dict:
    if tuple(sorted(provenance)) != tuple(sorted(PROVENANCE_FIELDS)):
        raise ValueError("approach_handoff_metadata_provenance_fields")
    for name in SHA256_FIELDS:
        _validate_sha256(name, provenance[name])
    if provenance["source_checkpoint_status"] != (
            "four_layer_continuation_checkpoint_saved"):
        raise ValueError("approach_handoff_metadata_source_status")
    if provenance["source_domain"] not in {
            "a30_formal_47", "local_reconstruction_47"}:
        raise ValueError("approach_handoff_metadata_source_domain")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "status": CHECKPOINT_STATUS,
        "checkpoint_file": Path(identity["path"]).name,
        "checkpoint_sha256": identity["sha256"],
        "particle_count": int(len(arrays["particle_q"])),
        "rock_count": int(len(arrays["rigid_qpos"])),
        "collider_body_count": int(len(arrays["rigid_qpos"]) + 1),
        "simulation_time_s": float(arrays["simulation_time_s"][0]),
        "source_checkpoint_schema": SOURCE_CHECKPOINT_SCHEMA,
        "solver_grid_type": "fixed",
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "independent_restore_audit_passed": False,
        "contact_handoff_eligible": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
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
        "source_checkpoint_schema": SOURCE_CHECKPOINT_SCHEMA,
        "solver_grid_type": "fixed",
        "bucket_collider_history_saved": True,
        "mechanism_state_saved": True,
        "independent_restore_audit_passed": False,
        "contact_handoff_eligible": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
    }
    for name, expected in required.items():
        if document.get(name) != expected:
            raise ValueError(
                f"approach_handoff_metadata_contract:{name}:"
                f"{document.get(name)!r}:{expected!r}")
    provenance = {name: document.get(name) for name in PROVENANCE_FIELDS}
    metadata_document(before, {
        "particle_q": np.empty((int(document["particle_count"]), 3)),
        "rigid_qpos": np.empty((int(document["rock_count"]), 7)),
        "simulation_time_s": np.asarray([document["simulation_time_s"]]),
    }, provenance)
    if expected_provenance is not None and provenance != expected_provenance:
        raise ValueError("approach_handoff_metadata_provenance_mismatch")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_arrays(arrays, expected_particle_count, expected_rock_count)
    if float(arrays["simulation_time_s"][0]) != float(
            document["simulation_time_s"]):
        raise ValueError("approach_handoff_metadata_simulation_time")
    if checkpoint_identity(checkpoint) != before:
        raise RuntimeError("approach_handoff_checkpoint_changed")
    if checkpoint_identity(metadata_file) != metadata_before:
        raise RuntimeError("approach_handoff_metadata_changed")
    return arrays, document, before
