#!/usr/bin/env python3
"""Strict, read-only loading and full-state capture for layered MPM checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


CHECKPOINT_SCHEMA = "excavation_layered_mound_continuation_checkpoint_v2"
MPM_STATE_FIELDS = (
    "particle_elastic_strain",
    "particle_transform",
    "particle_qd_grad",
    "particle_stress",
    "particle_Jp",
)
SOLVER_STATE_FIELDS = (
    "grid_reference_particle_q",
    "solver_grid_bounds_lo",
    "solver_grid_bounds_hi",
    "solver_grid_resolution",
    "solver_ws_impulse",
    "solver_ws_stress",
    "solver_body_q_prev",
)
CHECKPOINT_FIELDS = (
    "particle_q",
    "particle_qd",
    "particle_f",
    "particle_flags",
    *MPM_STATE_FIELDS,
    "rigid_ids",
    "rigid_qpos",
    "rigid_qvel",
    "rigid_reported_linear_velocity",
    "rigid_reported_angular_velocity",
    "simulation_time_s",
    *SOLVER_STATE_FIELDS,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path) -> dict:
    target = Path(path)
    stat = target.stat()
    return {
        "path": str(target.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(target),
    }


def _finite_numeric_arrays(arrays: dict[str, np.ndarray]) -> bool:
    return all(
        np.isfinite(value).all()
        for name, value in arrays.items()
        if name != "rigid_ids"
    )


def _validate_shapes(arrays: dict[str, np.ndarray], particle_count: int,
                     rock_count: int) -> None:
    particle_shapes = {
        "particle_q": (particle_count, 3),
        "particle_qd": (particle_count, 3),
        "particle_f": (particle_count, 3),
        "particle_flags": (particle_count,),
        "particle_elastic_strain": (particle_count, 3, 3),
        "particle_transform": (particle_count, 3, 3),
        "particle_qd_grad": (particle_count, 3, 3),
        "particle_stress": (particle_count, 3, 3),
        "particle_Jp": (particle_count,),
    }
    for name, expected in particle_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"checkpoint_shape:{name}:{arrays[name].shape}:{expected}")
    rigid_shapes = {
        "rigid_ids": (rock_count,),
        "rigid_qpos": (rock_count, 7),
        "rigid_qvel": (rock_count, 6),
        "rigid_reported_linear_velocity": (rock_count, 3),
        "rigid_reported_angular_velocity": (rock_count, 3),
        "simulation_time_s": (1,),
        "grid_reference_particle_q": (particle_count, 3),
        "solver_grid_bounds_lo": (3,),
        "solver_grid_bounds_hi": (3,),
        "solver_grid_resolution": (3,),
        "solver_body_q_prev": (rock_count, 7),
    }
    for name, expected in rigid_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"checkpoint_shape:{name}:{arrays[name].shape}:{expected}")
    solver_field_shapes = {
        "solver_ws_impulse": 3,
        "solver_ws_stress": 6,
    }
    for name, width in solver_field_shapes.items():
        actual = arrays[name].shape
        if len(actual) != 2 or actual[0] <= 0 or actual[1] != width:
            raise ValueError(
                f"checkpoint_shape:{name}:{actual}:(positive,{width})")
    resolution = arrays["solver_grid_resolution"]
    if (not np.issubdtype(resolution.dtype, np.integer)
            or np.any(resolution <= 0)):
        raise ValueError("checkpoint_solver_grid_resolution")
    if np.any(arrays["solver_grid_bounds_hi"] <=
              arrays["solver_grid_bounds_lo"]):
        raise ValueError("checkpoint_solver_grid_bounds")


def load_checkpoint_readonly(
        checkpoint_path: str | Path,
        metadata_path: str | Path,
        contract: dict) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Validate and copy a checkpoint without opening it for write access."""
    checkpoint = Path(checkpoint_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint_missing:{checkpoint}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"checkpoint_metadata_missing:{metadata_file}")
    if contract.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint_contract_schema")
    before = checkpoint_identity(checkpoint)
    metadata_before = checkpoint_identity(metadata_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    expected_sha = contract["sha256"]
    if (metadata.get("checkpoint_sha256") != expected_sha
            or before["sha256"] != expected_sha):
        raise ValueError(
            f"checkpoint_sha256:{before['sha256']}:{expected_sha}")
    expected_metadata = {
        "schema": contract["checkpoint_schema"],
        "status": contract["checkpoint_status"],
        "checkpoint_sha256": expected_sha,
        "source_config_sha256": contract["source_config_sha256"],
        "source_runner_sha256": contract["source_runner_sha256"],
        "particle_count": int(contract["particle_count"]),
        "rock_count": int(contract["rock_count"]),
        "gravity_predictor_required_on_next_free_step": True,
        "same_solver_material_and_dt_required": True,
        "fixed_grid_reference_saved": True,
        "newton_solver_warmstart_saved": True,
        "newton_version": contract["newton_version"],
        "warp_version": contract["warp_version"],
        "solver_grid_type": "fixed",
        "solver_warmstart_mode": "grid",
        "collider_velocity_mode": "forward",
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"checkpoint_metadata_contract:{name}:"
                f"{metadata.get(name)!r}:{expected!r}")
    if metadata.get("checkpoint_file") != checkpoint.name:
        raise ValueError(
            f"checkpoint_metadata_file:{metadata.get('checkpoint_file')!r}:"
            f"{checkpoint.name!r}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        names = tuple(sorted(archive.files))
        if names != tuple(sorted(CHECKPOINT_FIELDS)):
            raise ValueError(f"checkpoint_field_set:{names}")
        arrays = {name: archive[name].copy() for name in CHECKPOINT_FIELDS}
    particle_count = int(contract["particle_count"])
    rock_count = int(contract["rock_count"])
    _validate_shapes(arrays, particle_count, rock_count)
    if not _finite_numeric_arrays(arrays):
        raise ValueError("checkpoint_nonfinite")
    expected_ids = np.asarray(
        [f"r1-rock-{index:03d}" for index in range(rock_count)])
    if not np.array_equal(arrays["rigid_ids"].astype(str), expected_ids):
        raise ValueError("checkpoint_rigid_ids")
    # The generalized free-joint qvel and the bridge's reported body velocity
    # are intentionally retained as separate audit fields.  MuJoCo may express
    # them at different reference points and/or in different frames, so numeric
    # equality is not a checkpoint-integrity requirement.  Preserve their
    # difference as a diagnostic instead of rejecting a valid saved state.
    metadata = dict(metadata)
    metadata["load_diagnostics"] = {
        "rigid_reported_velocity_fields_are_audit_only": True,
        "reported_linear_vs_qvel_max_abs_difference": float(np.max(np.abs(
            arrays["rigid_qvel"][:, :3]
            - arrays["rigid_reported_linear_velocity"]))),
        "reported_angular_vs_qvel_max_abs_difference": float(np.max(np.abs(
            arrays["rigid_qvel"][:, 3:]
            - arrays["rigid_reported_angular_velocity"]))),
    }
    if float(arrays["simulation_time_s"][0]) <= 0.0:
        raise ValueError("checkpoint_simulation_time")
    if float(metadata["simulation_time_s"]) != float(
            arrays["simulation_time_s"][0]):
        raise ValueError("checkpoint_metadata_simulation_time")
    after = checkpoint_identity(checkpoint)
    if after != before:
        raise RuntimeError("checkpoint_changed_while_loading")
    if checkpoint_identity(metadata_file) != metadata_before:
        raise RuntimeError("checkpoint_metadata_changed_while_loading")
    return arrays, metadata, before


def restore_fixture_prefix(fixture, arrays: dict[str, np.ndarray]) -> dict:
    """Restore a checkpoint prefix while preserving initialized appended particles."""
    prefix = len(arrays["particle_q"])
    total = len(fixture.state.particle_q.numpy())
    if prefix > total:
        raise ValueError("checkpoint_restore_prefix_larger_than_fixture")
    targets = {
        "particle_q": fixture.state.particle_q,
        "particle_qd": fixture.state.particle_qd,
        "particle_f": fixture.state.particle_f,
        **{name: getattr(fixture.state.mpm, name) for name in MPM_STATE_FIELDS},
    }
    for name, target in targets.items():
        values = target.numpy().copy()
        if values.shape[1:] != arrays[name].shape[1:]:
            raise ValueError(f"checkpoint_restore_shape:{name}")
        values[:prefix] = arrays[name]
        target.assign(values)
    flags = fixture.model.particle_flags.numpy().copy()
    flags[:prefix] = arrays["particle_flags"]
    if total > prefix:
        active_flag = 1
        inactive_mask = np.bitwise_not(np.asarray(active_flag, dtype=flags.dtype))
        flags[prefix:] = np.bitwise_and(flags[prefix:], inactive_mask)
    fixture.model.particle_flags.assign(flags)
    restored = {
        name: bool(np.array_equal(target.numpy()[:prefix], arrays[name]))
        for name, target in targets.items()
    }
    restored["particle_flags"] = bool(np.array_equal(
        fixture.model.particle_flags.numpy()[:prefix], arrays["particle_flags"]))
    if not all(restored.values()):
        raise RuntimeError("checkpoint_restore_readback")
    return {
        "prefix_particle_count": prefix,
        "fixture_particle_count": total,
        "appended_particle_count": total - prefix,
        "field_readback_exact": restored,
    }


def _solver_grid_arrays(fixture) -> dict[str, np.ndarray]:
    scratch = getattr(fixture.solver, "_scratchpad", None)
    if scratch is None or getattr(scratch, "grid", None) is None:
        raise RuntimeError("checkpoint_solver_grid_unavailable")
    grid = scratch.grid
    return {
        "solver_grid_bounds_lo": np.asarray(
            [float(value) for value in grid.bounds_lo], dtype=np.float32),
        "solver_grid_bounds_hi": np.asarray(
            [float(value) for value in grid.bounds_hi], dtype=np.float32),
        "solver_grid_resolution": np.asarray(
            [int(value) for value in grid.res], dtype=np.int32),
    }


def _solver_continuation_arrays(fixture) -> dict[str, np.ndarray]:
    reference = getattr(fixture, "grid_reference_particle_q", None)
    if reference is None:
        raise RuntimeError("checkpoint_grid_reference_particle_q_missing")
    reference = np.asarray(reference).copy()
    if reference.shape != fixture.state.particle_q.numpy().shape:
        raise ValueError("checkpoint_grid_reference_particle_q_shape")
    last = getattr(fixture.solver, "_last_step_data", None)
    if (last is None or last.ws_impulse_field is None
            or last.ws_stress_field is None or last.body_q_prev is None):
        raise RuntimeError("checkpoint_solver_last_step_data_unavailable")
    return {
        "grid_reference_particle_q": reference,
        **_solver_grid_arrays(fixture),
        "solver_ws_impulse": last.ws_impulse_field.dof_values.numpy().copy(),
        "solver_ws_stress": last.ws_stress_field.dof_values.numpy().copy(),
        "solver_body_q_prev": last.body_q_prev.numpy().copy(),
    }


def restore_solver_continuation(fixture, arrays: dict[str, np.ndarray]) -> dict:
    """Restore the private Newton state required at an exact step boundary."""
    actual_grid = _solver_grid_arrays(fixture)
    grid_exact = {
        name: bool(np.array_equal(actual_grid[name], arrays[name]))
        for name in actual_grid
    }
    if not all(grid_exact.values()):
        raise RuntimeError("checkpoint_solver_grid_readback")
    last = getattr(fixture.solver, "_last_step_data", None)
    if (last is None or last.ws_impulse_field is None
            or last.ws_stress_field is None or last.body_q_prev is None):
        raise RuntimeError("checkpoint_solver_last_step_data_unavailable")
    targets = {
        "solver_ws_impulse": last.ws_impulse_field.dof_values,
        "solver_ws_stress": last.ws_stress_field.dof_values,
        "solver_body_q_prev": last.body_q_prev,
    }
    for name, target in targets.items():
        if target.numpy().shape != arrays[name].shape:
            raise ValueError(
                f"checkpoint_solver_restore_shape:{name}:"
                f"{target.numpy().shape}:{arrays[name].shape}")
        target.assign(arrays[name])
    restored = {
        name: bool(np.array_equal(target.numpy(), arrays[name]))
        for name, target in targets.items()
    }
    if not all(restored.values()):
        raise RuntimeError("checkpoint_solver_restore_readback")
    return {
        "grid_readback_exact": grid_exact,
        "solver_field_readback_exact": restored,
    }


def transfer_solver_warmstart(source_fixture, target_fixture) -> dict:
    """Carry grid fields across an intentional fixture/grid expansion.

    Newton's grid warmstart path supports nonconforming interpolation.  The
    target keeps its own collider transform history because an expansion can
    change the rigid-body count; this project uses forward collider velocity,
    so that history is not a velocity input at the transition.
    """
    source = getattr(source_fixture.solver, "_last_step_data", None)
    target = getattr(target_fixture.solver, "_last_step_data", None)
    if (source is None or target is None
            or source.ws_impulse_field is None
            or source.ws_stress_field is None):
        raise RuntimeError("checkpoint_solver_warmstart_transfer_unavailable")
    mode = getattr(target_fixture.solver, "collider_velocity_mode", None)
    if mode != "forward":
        raise ValueError(f"checkpoint_solver_warmstart_transfer_mode:{mode}")
    source_shapes = {
        "solver_ws_impulse": list(
            source.ws_impulse_field.dof_values.numpy().shape),
        "solver_ws_stress": list(
            source.ws_stress_field.dof_values.numpy().shape),
    }
    target_shapes_before = {
        "solver_ws_impulse": list(
            target.ws_impulse_field.dof_values.numpy().shape),
        "solver_ws_stress": list(
            target.ws_stress_field.dof_values.numpy().shape),
    }
    target.ws_impulse_field = source.ws_impulse_field
    target.ws_stress_field = source.ws_stress_field
    return {
        "mode": "newton_nonconforming_grid_field_transfer",
        "collider_velocity_mode": mode,
        "source_shapes": source_shapes,
        "target_shapes_before": target_shapes_before,
        "body_q_prev_transferred": False,
    }


def checkpoint_arrays(fixture, bridge_state: dict,
                      simulation_time_s: float) -> dict[str, np.ndarray]:
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
        "simulation_time_s": np.asarray([simulation_time_s], dtype=np.float64),
        **_solver_continuation_arrays(fixture),
    }


def write_checkpoint_atomic(path: str | Path,
                            arrays: dict[str, np.ndarray]) -> dict:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    if tuple(sorted(arrays)) != tuple(sorted(CHECKPOINT_FIELDS)):
        raise ValueError("checkpoint_write_field_set")
    _validate_shapes(
        arrays, len(arrays["particle_q"]), len(arrays["rigid_qpos"]))
    if not _finite_numeric_arrays(arrays):
        raise ValueError("checkpoint_write_nonfinite")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=target.name + ".", suffix=".tmp",
            delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
    os.replace(temporary, target)
    return checkpoint_identity(target)
