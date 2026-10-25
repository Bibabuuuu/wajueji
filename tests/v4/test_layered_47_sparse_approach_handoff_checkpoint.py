#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import layered_47_sparse_approach_handoff_checkpoint as checkpoint  # noqa: E402


def arrays(particles=3, rocks=47):
    result = {
        "particle_q": np.zeros((particles, 3), np.float32),
        "particle_qd": np.zeros((particles, 3), np.float32),
        "particle_f": np.zeros((particles, 3), np.float32),
        "particle_flags": np.ones(particles, np.int32),
        "particle_elastic_strain": np.zeros((particles, 3, 3), np.float32),
        "particle_transform": np.zeros((particles, 3, 3), np.float32),
        "particle_qd_grad": np.zeros((particles, 3, 3), np.float32),
        "particle_stress": np.zeros((particles, 3, 3), np.float32),
        "particle_Jp": np.ones(particles, np.float32),
        "rigid_ids": np.asarray(
            [f"r1-rock-{index:03d}" for index in range(rocks)]),
        "rigid_qpos": np.zeros((rocks, 7), np.float64),
        "rigid_qvel": np.zeros((rocks, 6), np.float64),
        "rigid_reported_linear_velocity": np.zeros((rocks, 3), np.float64),
        "rigid_reported_angular_velocity": np.zeros((rocks, 3), np.float64),
        "mechanism_qpos": np.zeros(19, np.float64),
        "mechanism_qvel": np.zeros(18, np.float64),
        "simulation_time_s": np.asarray([2.0], np.float64),
        "grid_reference_particle_q": np.zeros((particles, 3), np.float32),
        "solver_grid_transform": np.eye(3, dtype=np.float32),
        "solver_grid_cell_ijk": np.zeros((2, 3), np.int32),
        "solver_grid_node_ijk": np.zeros((3, 3), np.int32),
        "solver_ws_impulse": np.zeros((3, 3), np.float32),
        "solver_ws_stress": np.zeros((3, 6), np.float32),
        "solver_body_q_prev": np.zeros((rocks + 1, 7), np.float32),
    }
    return result


def provenance():
    result = {name: f"{index + 1:064x}"
              for index, name in enumerate(checkpoint.SHA256_FIELDS)}
    result.update(source_domain=checkpoint.SOURCE_DOMAIN,
                  source_checkpoint_status="grid_candidate_diagnostic_only",
                  hz=60, rigid_internal_hz=4800)
    return result


class SparseApproachHandoffCheckpointTest(unittest.TestCase):
    def test_roundtrip_preserves_sparse_grid_mechanism_and_bucket_history(self):
        values = arrays()
        values["mechanism_qpos"] = np.arange(19, dtype=np.float64)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(path, values)
            metadata = Path(str(path) + ".json")
            checkpoint.write_metadata_atomic(
                metadata, checkpoint.metadata_document(
                    identity, values, provenance()))
            loaded, document, before = checkpoint.load_checkpoint_readonly(
                path, metadata, provenance(), expected_particle_count=3)
            self.assertEqual(before["sha256"], identity["sha256"])
            np.testing.assert_array_equal(
                loaded["solver_grid_node_ijk"],
                values["solver_grid_node_ijk"])
            np.testing.assert_array_equal(
                loaded["mechanism_qpos"], values["mechanism_qpos"])
            self.assertEqual(loaded["solver_body_q_prev"].shape, (48, 7))
            self.assertFalse(document["contact_handoff_eligible"])
            self.assertFalse(document["training_eligible"])

    def test_rejects_missing_sparse_grid_identity(self):
        values = arrays()
        del values["solver_grid_cell_ijk"]
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "field_set"):
                checkpoint.write_checkpoint_atomic(
                    Path(raw) / "handoff.npz", values)

    def test_rejects_mound_history_without_bucket(self):
        values = arrays()
        values["solver_body_q_prev"] = np.zeros((47, 7), np.float32)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "solver_body_q_prev"):
                checkpoint.write_checkpoint_atomic(
                    Path(raw) / "handoff.npz", values)

    def test_metadata_cannot_claim_audit_or_contact_eligibility(self):
        values = arrays()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(path, values)
            document = checkpoint.metadata_document(
                identity, values, provenance())
            document["contact_handoff_eligible"] = True
            metadata = Path(str(path) + ".json")
            checkpoint.write_metadata_atomic(metadata, document)
            with self.assertRaisesRegex(
                    ValueError, "contact_handoff_eligible"):
                checkpoint.load_checkpoint_readonly(
                    path, metadata, expected_particle_count=3)

    def test_fixed_grid_provenance_is_rejected(self):
        values = arrays()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "handoff.npz"
            identity = checkpoint.write_checkpoint_atomic(path, values)
            bad = provenance()
            bad["source_checkpoint_status"] = (
                "four_layer_continuation_checkpoint_saved")
            with self.assertRaisesRegex(ValueError, "source_status"):
                checkpoint.metadata_document(identity, values, bad)


if __name__ == "__main__":
    unittest.main()
