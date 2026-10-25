#!/usr/bin/env python3
"""Isolate fixed-grid and implicit-MPM warmstart state at a restart boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import newton
import numpy as np
import warp as wp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/v4"))

from layered_mound_checkpoint import MPM_STATE_FIELDS  # noqa: E402
from run_layered_mound_three_rock_adaptive_press import MultiRockFixture  # noqa: E402


def physical_state(fixture: MultiRockFixture) -> dict[str, np.ndarray]:
    return {
        "particle_q": fixture.state.particle_q.numpy().copy(),
        "particle_qd": fixture.state.particle_qd.numpy().copy(),
        "particle_f": fixture.state.particle_f.numpy().copy(),
        "particle_flags": fixture.model.particle_flags.numpy().copy(),
        **{
            name: getattr(fixture.state.mpm, name).numpy().copy()
            for name in MPM_STATE_FIELDS
        },
        "body_q": fixture.state.body_q.numpy().copy(),
        "body_qd": fixture.state.body_qd.numpy().copy(),
        "body_f": fixture.state.body_f.numpy().copy(),
    }


def restore_physical(fixture: MultiRockFixture, saved: dict[str, np.ndarray]) -> None:
    fixture.state.particle_q.assign(saved["particle_q"])
    fixture.state.particle_qd.assign(saved["particle_qd"])
    fixture.state.particle_f.assign(saved["particle_f"])
    fixture.model.particle_flags.assign(saved["particle_flags"])
    for name in MPM_STATE_FIELDS:
        getattr(fixture.state.mpm, name).assign(saved[name])
    fixture.state.body_q.assign(saved["body_q"])
    fixture.state.body_qd.assign(saved["body_qd"])
    fixture.state.body_f.assign(saved["body_f"])
    fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)


def solver_state(fixture: MultiRockFixture) -> dict[str, np.ndarray]:
    data = fixture.solver._last_step_data
    return {
        "ws_impulse": data.ws_impulse_field.dof_values.numpy().copy(),
        "ws_stress": data.ws_stress_field.dof_values.numpy().copy(),
        "body_q_prev": data.body_q_prev.numpy().copy(),
    }


def restore_solver(fixture: MultiRockFixture, saved: dict[str, np.ndarray]) -> None:
    data = fixture.solver._last_step_data
    data.ws_impulse_field.dof_values.assign(saved["ws_impulse"])
    data.ws_stress_field.dof_values.assign(saved["ws_stress"])
    data.body_q_prev.assign(saved["body_q_prev"])


def snapshot(fixture: MultiRockFixture) -> dict[str, np.ndarray]:
    return {
        "q": fixture.state.particle_q.numpy().copy(),
        "qd": fixture.state.particle_qd.numpy().copy(),
        "stress": fixture.state.mpm.particle_stress.numpy().copy(),
        "transform": fixture.state.mpm.particle_transform.numpy().copy(),
    }


def max_differences(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(reference[name] - other[name])))
        for name in reference
    }


def grid_description(fixture: MultiRockFixture) -> dict[str, list[float] | list[int]]:
    grid = fixture.solver._scratchpad.grid
    return {
        "resolution": [int(value) for value in grid.res],
        "bounds_lo": [float(value) for value in grid.bounds_lo],
        "bounds_hi": [float(value) for value in grid.bounds_hi],
    }


def make_fixture(config: dict, record: dict, grid_q: np.ndarray) -> MultiRockFixture:
    return MultiRockFixture(config, ROOT, [copy.deepcopy(record)], grid_q, "cpu")


def main() -> None:
    wp.init()
    config = json.loads(
        (ROOT / "configs/v4/layered-mound-thirtythree-rock-three-layer-v1.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "results/layered-mound-physical-preflight-v1/scene-manifest.json")
        .read_text(encoding="utf-8")
    )
    record = manifest["rocks"][0]

    spacing = float(config["mpm"]["particle_spacing_m"])
    # Keep the fixture tiny while spanning enough cells to exercise grid-backed
    # stress/impulse warmstarts and a gravity-driven restart boundary.
    initial_q = np.asarray(
        [
            (x * spacing, y * spacing, 0.016 + z * spacing)
            for z in range(3)
            for y in range(5)
            for x in range(5)
        ],
        dtype=np.float32,
    )
    dt = float(config["probe"]["output_dt_s"])
    origins = [np.asarray(record["position_world_m"], dtype=np.float64)]

    source = make_fixture(config, record, initial_q)
    for _ in range(24):
        source.step(dt, origins)
    saved_physical = physical_state(source)
    saved_solver = solver_state(source)
    saved_grid_q = saved_physical["particle_q"].copy()
    reference_grid = grid_description(source)

    source.step(dt, origins)
    reference = snapshot(source)

    cold_same_grid = make_fixture(config, record, initial_q)
    restore_physical(cold_same_grid, saved_physical)
    cold_same_grid.step(dt, origins)

    warm_same_grid = make_fixture(config, record, initial_q)
    restore_physical(warm_same_grid, saved_physical)
    restore_solver(warm_same_grid, saved_solver)
    warm_same_grid.step(dt, origins)

    particle_stress_same_grid = make_fixture(config, record, initial_q)
    restore_physical(particle_stress_same_grid, saved_physical)
    particle_stress_same_grid.solver._stress_warmstart = "particles"
    particle_stress_same_grid.step(dt, origins)

    cold_rebuilt_grid = make_fixture(config, record, saved_grid_q)
    restore_physical(cold_rebuilt_grid, saved_physical)
    cold_rebuilt_grid.step(dt, origins)

    report = {
        "particle_count": int(len(initial_q)),
        "source_steps_before_boundary": 24,
        "saved_particle_max_speed_m_s": float(
            np.linalg.norm(saved_physical["particle_qd"], axis=1).max()
        ),
        "warmstart_shapes": {
            key: list(value.shape) for key, value in saved_solver.items()
        },
        "fixed_grid_source": reference_grid,
        "fixed_grid_same_initial_q": grid_description(cold_same_grid),
        "fixed_grid_rebuilt_from_saved_q": grid_description(cold_rebuilt_grid),
        "cold_same_grid_vs_reference": max_differences(
            reference, snapshot(cold_same_grid)
        ),
        "warm_same_grid_vs_reference": max_differences(
            reference, snapshot(warm_same_grid)
        ),
        "particle_stress_same_grid_vs_reference": max_differences(
            reference, snapshot(particle_stress_same_grid)
        ),
        "cold_rebuilt_grid_vs_reference": max_differences(
            reference, snapshot(cold_rebuilt_grid)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
