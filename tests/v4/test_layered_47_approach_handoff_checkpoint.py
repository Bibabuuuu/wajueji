#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import layered_47_approach_handoff_checkpoint as checkpoint  # noqa: E402


def arrays(particles: int = 3, rocks: int = 47) -> dict[str, np.ndarray]:
    result = {
        "particle_q": np.zeros((particles, 3), dtype=np.float32),
        "particle_qd": np.zeros((particles, 3), dtype=np.float32),
        "particle_f": np.zeros((particles, 3), dtype=np.float32),
        "particle_flags": np.ones((particles,), dtype=np.int32),
        "particle_elastic_strain": np.zeros(
            (particles, 3, 3), dtype=np.float32),
        "particle_transform": np.zeros(
            (particles, 3, 3), dtype=np.float32),
        "particle_qd_grad": np.zeros(
            (particles, 3, 3), dtype=np.float32),
        "particle_stress": np.zeros(
            (particles, 3, 3), dtype=np.float32),
        "particle_Jp": np.ones((particles,), dtype=np.float32),
        "rigid_ids": np.asarray(
            [f"r1-rock-{index:03d}" for index in range(rocks)]),
        "rigid_qpos": np.zeros((rocks, 7), dtype=np.float64),
        "rigid_qvel": np.zeros((rocks, 6), dtype=np.float64),
        "rigid_reported_linear_velocity": np.zeros(
            (rocks, 3), dtype=np.float64),
        "rigid_reported_angular_velocity": np.zeros(
            (rocks, 3), dtype=np.float64),
        "mechanism_qpos": np.zeros((19,), dtype=np.float64),
        "mechanism_qvel": np.zeros((18,), dtype=np.float64),
        "simulation_time_s": np.asarray([1.0], dtype=np.float64),
        "grid_reference_particle_q": np.zeros(
            (particles, 3), dtype=np.float32),
        "solver_grid_bounds_lo": np.zeros((3,), dtype=np.float32),
        "solver_grid_bounds_hi": np.ones((3,), dtype=np.float32),
        "solver_grid_resolution": np.ones((3,), dtype=np.int32),
        "solver_ws_impulse": np.zeros((2, 3), dtype=np.float32),
        "solver_ws_stress": np.zeros((2, 6), dtype=np.float32),
        "solver_body_q_prev": np.zeros((rocks + 1, 7), dtype=np.float32),
    }
    return result


def provenance() -> dict:
    names = checkpoint.SHA256_FIELDS
    result = {name: f"{index + 1:064x}"
              for index, name in enumerate(names)}
    result.update({
        "source_domain": "a30_formal_47",
        "source_checkpoint_status":
            "four_layer_continuation_checkpoint_saved",
    })
    return result


class ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value.copy()


def fake_fixture(values: dict[str, np.ndarray]):
    class Object:
        pass

    fixture = Object()
    fixture.state = Object()
    fixture.state.particle_q = ArrayValue(values["particle_q"])
    fixture.state.particle_qd = ArrayValue(values["particle_qd"])
    fixture.state.particle_f = ArrayValue(values["particle_f"])
    fixture.state.mpm = Object()
    for name in (
            "particle_elastic_strain", "particle_transform",
            "particle_qd_grad", "particle_stress", "particle_Jp"):
        setattr(fixture.state.mpm, name, ArrayValue(values[name]))
    fixture.model = Object()
    fixture.model.particle_flags = ArrayValue(values["particle_flags"])
    fixture.grid_reference_particle_q = values["grid_reference_particle_q"]
    fixture.solver = Object()
    fixture.solver._scratchpad = Object()
    fixture.solver._scratchpad.grid = Object()
    grid = fixture.solver._scratchpad.grid
    grid.bounds_lo = values["solver_grid_bounds_lo"]
    grid.bounds_hi = values["solver_grid_bounds_hi"]
    grid.res = values["solver_grid_resolution"]
    fixture.solver._last_step_data = Object()
    last = fixture.solver._last_step_data
    last.ws_impulse_field = Object()
    last.ws_impulse_field.dof_values = ArrayValue(
        values["solver_ws_impulse"])
    last.ws_stress_field = Object()
    last.ws_stress_field.dof_values = ArrayValue(
        values["solver_ws_stress"])
    last.body_q_prev = ArrayValue(values["solver_body_q_prev"])
    return fixture


class ApproachHandoffCheckpointTest(unittest.TestCase):
    def test_capture_adds_mechanism_to_expanded_layered_state(self):
        values = arrays()
        bridge_state = {
            "rocks": [{
                "rock_id": f"r1-rock-{index:03d}",
                "qpos": values["rigid_qpos"][index],
                "qvel": values["rigid_qvel"][index],
                "linear_velocity_world_m_s":
                    values["rigid_reported_linear_velocity"][index],
                "angular_velocity_world_rad_s":
                    values["rigid_reported_angular_velocity"][index],
            } for index in range(47)],
            "mechanism_qpos": np.arange(19, dtype=np.float64),
            "mechanism_qvel": np.arange(18, dtype=np.float64),
        }
        captured = checkpoint.capture_arrays(
            fake_fixture(values), bridge_state, 2.0)
        checkpoint.validate_arrays(captured, 3, 47)
        self.assertEqual(captured["solver_body_q_prev"].shape, (48, 7))
        np.testing.assert_array_equal(
            captured["mechanism_qpos"], bridge_state["mechanism_qpos"])

    def test_roundtrip_keeps_full_fixed_grid_and_mechanism_state(self):
        values = arrays()
        values["mechanism_qpos"] = np.arange(19, dtype=np.float64)
        values["mechanism_qvel"] = np.arange(18, dtype=np.float64) / 10.0
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(target, values)
            metadata = Path(raw) / "handoff.npz.json"
            document = checkpoint.metadata_document(
                identity, values, provenance())
            checkpoint.write_metadata_atomic(metadata, document)
            loaded, loaded_document, before = (
                checkpoint.load_checkpoint_readonly(
                    target, metadata, provenance(),
                    expected_particle_count=3))
            self.assertEqual(before["sha256"], identity["sha256"])
            self.assertEqual(
                loaded["solver_body_q_prev"].shape, (48, 7))
            np.testing.assert_array_equal(
                loaded["mechanism_qpos"], values["mechanism_qpos"])
            np.testing.assert_array_equal(
                loaded["mechanism_qvel"], values["mechanism_qvel"])
            self.assertFalse(loaded_document["contact_handoff_eligible"])
            self.assertFalse(loaded_document["training_eligible"])

    def test_rejects_layered_mound_history_without_bucket_body(self):
        values = arrays()
        values["solver_body_q_prev"] = np.zeros((47, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(
                    ValueError, "solver_body_q_prev"):
                checkpoint.write_checkpoint_atomic(
                    Path(raw) / "handoff.npz", values)

    def test_rejects_goal_only_mechanism_state(self):
        values = arrays()
        del values["mechanism_qvel"]
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "field_set"):
                checkpoint.write_checkpoint_atomic(
                    Path(raw) / "handoff.npz", values)

    def test_writer_refuses_to_overwrite_checkpoint_and_metadata(self):
        values = arrays()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(target, values)
            with self.assertRaises(FileExistsError):
                checkpoint.write_checkpoint_atomic(target, values)
            metadata = Path(raw) / "handoff.npz.json"
            document = checkpoint.metadata_document(
                identity, values, provenance())
            checkpoint.write_metadata_atomic(metadata, document)
            with self.assertRaises(FileExistsError):
                checkpoint.write_metadata_atomic(metadata, document)

    def test_formal_loader_defaults_to_the_frozen_particle_count(self):
        values = arrays()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(target, values)
            metadata = Path(raw) / "handoff.npz.json"
            checkpoint.write_metadata_atomic(
                metadata,
                checkpoint.metadata_document(
                    identity, values, provenance()))
            with self.assertRaisesRegex(ValueError, "particle_count"):
                checkpoint.load_checkpoint_readonly(target, metadata)

    def test_loader_rejects_metadata_that_claims_contact_qualification(self):
        values = arrays()
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(target, values)
            metadata = Path(raw) / "handoff.npz.json"
            document = checkpoint.metadata_document(
                identity, values, provenance())
            document["contact_handoff_eligible"] = True
            metadata.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "contact_handoff_eligible"):
                checkpoint.load_checkpoint_readonly(
                    target, metadata, expected_particle_count=3)


if __name__ == "__main__":
    unittest.main()
