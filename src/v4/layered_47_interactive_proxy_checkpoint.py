#!/usr/bin/env python3
"""Checkpoint contract for the non-formal 47-rock interactive proxy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from build_mixed_material_capacity_preflight import sha256
from layered_mound_checkpoint import (
    MPM_STATE_FIELDS,
    checkpoint_identity,
)


CHECKPOINT_SCHEMA = "excavation_layered_47_interactive_proxy_checkpoint_v1"
CHECKPOINT_STATUS = "interactive_proxy_benchmark_recovery_saved"
PARTICLE_FIELDS = (
    "particle_q", "particle_qd", "particle_f", "particle_flags",
    *MPM_STATE_FIELDS,
)
SOLVER_FIELDS = (
    "grid_reference_particle_q", "solver_grid_transform",
    "solver_grid_cell_ijk", "solver_grid_node_ijk", "solver_ws_impulse",
    "solver_ws_stress", "solver_body_q_prev",
)
RIGID_FIELDS = (
    "rigid_ids", "rigid_qpos", "rigid_qvel",
    "rigid_reported_linear_velocity", "rigid_reported_angular_velocity",
)
MECHANISM_FIELDS = ("mechanism_qpos", "mechanism_qvel")
CHECKPOINT_FIELDS = (
    *PARTICLE_FIELDS, *RIGID_FIELDS, *MECHANISM_FIELDS,
    "simulation_time_s", *SOLVER_FIELDS,
)


def capture_arrays(fixture, bridge_state: dict,
                   simulation_time_s: float) -> dict[str, np.ndarray]:
    scratch = getattr(fixture.solver, "_scratchpad", None)
    grid = getattr(scratch, "grid", None)
    last = getattr(fixture.solver, "_last_step_data", None)
    if (grid is None or grid.__class__.__name__ != "Nanogrid"
            or last is None or last.ws_impulse_field is None
            or last.ws_stress_field is None or last.body_q_prev is None):
        raise RuntimeError("interactive_proxy_sparse_solver_state_unavailable")
    reference = np.asarray(fixture.grid_reference_particle_q).copy()
    return {
        "particle_q": fixture.state.particle_q.numpy(),
        "particle_qd": fixture.state.particle_qd.numpy(),
        "particle_f": fixture.state.particle_f.numpy(),
        "particle_flags": fixture.model.particle_flags.numpy(),
        **{name: getattr(fixture.state.mpm, name).numpy()
           for name in MPM_STATE_FIELDS},
        "rigid_ids": np.asarray(
            [rock["rock_id"] for rock in bridge_state["rocks"]]),
        "rigid_qpos": np.asarray(
            [rock["qpos"] for rock in bridge_state["rocks"]], dtype=np.float64),
        "rigid_qvel": np.asarray(
            [rock["qvel"] for rock in bridge_state["rocks"]], dtype=np.float64),
        "rigid_reported_linear_velocity": np.asarray([
            rock["linear_velocity_world_m_s"]
            for rock in bridge_state["rocks"]], dtype=np.float64),
        "rigid_reported_angular_velocity": np.asarray([
            rock["angular_velocity_world_rad_s"]
            for rock in bridge_state["rocks"]], dtype=np.float64),
        "mechanism_qpos": np.asarray(
            bridge_state["mechanism_qpos"], dtype=np.float64),
        "mechanism_qvel": np.asarray(
            bridge_state["mechanism_qvel"], dtype=np.float64),
        "simulation_time_s": np.asarray(
            [simulation_time_s], dtype=np.float64),
        "grid_reference_particle_q": reference,
        "solver_grid_transform": np.asarray(
            grid.transform, dtype=np.float32).reshape(3, 3),
        "solver_grid_cell_ijk": grid._cell_ijk.numpy().copy(),
        "solver_grid_node_ijk": grid._node_ijk.numpy().copy(),
        "solver_ws_impulse": (
            last.ws_impulse_field.dof_values.numpy().copy()),
        "solver_ws_stress": last.ws_stress_field.dof_values.numpy().copy(),
        "solver_body_q_prev": last.body_q_prev.numpy().copy(),
    }


def _validate(arrays: dict[str, np.ndarray], particle_count: int,
              rock_count: int) -> None:
    if tuple(sorted(arrays)) != tuple(sorted(CHECKPOINT_FIELDS)):
        raise ValueError("interactive_proxy_checkpoint_field_set")
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
        "solver_grid_transform": (3, 3),
        # 47 rocks plus the kinematic bucket collider body.
        "solver_body_q_prev": (rock_count + 1, 7),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"interactive_proxy_checkpoint_shape:{name}:"
                f"{arrays[name].shape}:{shape}")
    for name in ("solver_grid_cell_ijk", "solver_grid_node_ijk"):
        shape = arrays[name].shape
        if (len(shape) != 2 or shape[0] <= 0 or shape[1] != 3
                or not np.issubdtype(arrays[name].dtype, np.integer)):
            raise ValueError(
                f"interactive_proxy_checkpoint_shape:{name}:{shape}")
    for name, width in (("solver_ws_impulse", 3),
                        ("solver_ws_stress", 6)):
        shape = arrays[name].shape
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != width:
            raise ValueError(
                f"interactive_proxy_checkpoint_shape:{name}:{shape}")
    if not all(np.isfinite(value).all() for name, value in arrays.items()
               if name != "rigid_ids"):
        raise ValueError("interactive_proxy_checkpoint_nonfinite")
    expected_ids = np.asarray(
        [f"r1-rock-{index:03d}" for index in range(rock_count)])
    if not np.array_equal(arrays["rigid_ids"].astype(str), expected_ids):
        raise ValueError("interactive_proxy_checkpoint_rigid_ids")


def write_checkpoint_atomic(path: str | Path,
                            arrays: dict[str, np.ndarray]) -> dict:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    _validate(arrays, len(arrays["particle_q"]), len(arrays["rigid_qpos"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp",
            delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
    os.replace(temporary, target)
    return checkpoint_identity(target)


def write_rejected_diagnostic_atomic(path: str | Path,
                                     arrays: dict[str, np.ndarray],
                                     rejection: str) -> dict:
    """Preserve rejected raw state without granting checkpoint validity.

    NaN/Inf values must remain intact for diagnosis; do not sanitize them into
    a resumable checkpoint. The strict writer/loader contracts stay unchanged.
    """
    target = Path(path).resolve()
    metadata_path = Path(str(target) + ".json")
    if target.exists() or metadata_path.exists():
        raise FileExistsError(target)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("raw_diagnostic_object_dtype")
    nonfinite = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind not in "fc":
            continue
        bad = ~np.isfinite(value)
        if bad.any():
            nonfinite[name] = {
                "count": int(bad.sum()),
                "first_indices": np.argwhere(bad)[:16].tolist(),
                "shape": list(value.shape),
            }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=target.name + ".", suffix=".tmp",
                delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result = {
        **checkpoint_identity(target),
        "schema": "rejected_raw_diagnostic_not_checkpoint_v1",
        "status": "rejected_raw_diagnostic_only",
        "rejection": str(rejection),
        "resume_qualified": False,
        "training_eligible": False,
        "nonfinite_fields": nonfinite,
    }
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def restore_solver_continuation(fixture, arrays: dict[str, np.ndarray]) -> dict:
    """Restore sparse-grid warm-start fields and expanded collider history."""
    scratch = getattr(fixture.solver, "_scratchpad", None)
    grid = getattr(scratch, "grid", None)
    last = getattr(fixture.solver, "_last_step_data", None)
    if (grid is None or grid.__class__.__name__ != "Nanogrid"
            or last is None or last.ws_impulse_field is None
            or last.ws_stress_field is None or last.body_q_prev is None):
        raise RuntimeError("interactive_proxy_sparse_solver_restore_unavailable")
    actual_grid = {
        "solver_grid_transform": np.asarray(
            grid.transform, dtype=np.float32).reshape(3, 3),
        "solver_grid_cell_ijk": grid._cell_ijk.numpy().copy(),
        "solver_grid_node_ijk": grid._node_ijk.numpy().copy(),
    }
    grid_exact = {
        name: bool(np.array_equal(value, arrays[name]))
        for name, value in actual_grid.items()
    }
    if not all(grid_exact.values()):
        raise RuntimeError("interactive_proxy_sparse_grid_readback")
    targets = {
        "solver_ws_impulse": last.ws_impulse_field.dof_values,
        "solver_ws_stress": last.ws_stress_field.dof_values,
        "solver_body_q_prev": last.body_q_prev,
    }
    for name, target in targets.items():
        if target.numpy().shape != arrays[name].shape:
            raise ValueError(
                f"interactive_proxy_sparse_restore_shape:{name}:"
                f"{target.numpy().shape}:{arrays[name].shape}")
        target.assign(arrays[name])
    restored = {
        name: bool(np.array_equal(target.numpy(), arrays[name]))
        for name, target in targets.items()
    }
    if not all(restored.values()):
        raise RuntimeError("interactive_proxy_sparse_restore_readback")
    return {
        "grid_readback_exact": grid_exact,
        "solver_field_readback_exact": restored,
    }


def load_checkpoint_readonly(
        checkpoint_path: str | Path,
        metadata_path: str | Path) -> tuple[dict[str, np.ndarray], dict, dict]:
    checkpoint = Path(checkpoint_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    before = checkpoint_identity(checkpoint)
    metadata_before = checkpoint_identity(metadata_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    required = {
        "schema": CHECKPOINT_SCHEMA,
        "status": CHECKPOINT_STATUS,
        "checkpoint_sha256": before["sha256"],
        "checkpoint_file": checkpoint.name,
        "particle_count": 256441,
        "rock_count": 47,
        "solver_grid_type": "sparse",
        "solver_voxel_size_m": 0.1,
        "solver_grid_padding": 2,
        "manual_collection_eligible": False,
        "training_eligible": False,
        "formal_a30_equivalent": False,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"interactive_proxy_checkpoint_metadata:{name}:"
                f"{metadata.get(name)!r}:{expected!r}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _validate(arrays, int(metadata["particle_count"]),
              int(metadata["rock_count"]))
    if sha256(metadata_file) != metadata_before["sha256"]:
        raise RuntimeError("interactive_proxy_checkpoint_metadata_changed")
    if checkpoint_identity(checkpoint) != before:
        raise RuntimeError("interactive_proxy_checkpoint_changed")
    return arrays, metadata, before
