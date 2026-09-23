from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))
import layered_47_loaded_handoff_checkpoint as checkpoint  # noqa: E402


def arrays(particles=3, rocks=47):
    return {
        "particle_q": np.zeros((particles, 3), np.float32),
        "particle_qd": np.zeros((particles, 3), np.float32),
        "particle_f": np.zeros((particles, 3), np.float32),
        "particle_flags": np.ones(particles, np.int32),
        "particle_elastic_strain": np.zeros((particles, 3, 3), np.float32),
        "particle_transform": np.zeros((particles, 3, 3), np.float32),
        "particle_qd_grad": np.zeros((particles, 3, 3), np.float32),
        "particle_stress": np.zeros((particles, 3, 3), np.float32),
        "particle_Jp": np.ones(particles, np.float32),
        "rigid_ids": np.asarray([f"r1-rock-{i:03d}" for i in range(rocks)]),
        "rigid_qpos": np.zeros((rocks, 7), np.float64),
        "rigid_qvel": np.zeros((rocks, 6), np.float64),
        "rigid_reported_linear_velocity": np.zeros((rocks, 3), np.float64),
        "rigid_reported_angular_velocity": np.zeros((rocks, 3), np.float64),
        "mechanism_qpos": np.zeros(19, np.float64),
        "mechanism_qvel": np.zeros(18, np.float64),
        "simulation_time_s": np.asarray([20.2], np.float64),
        "grid_reference_particle_q": np.zeros((particles, 3), np.float32),
        "solver_grid_transform": np.eye(3, dtype=np.float32),
        "solver_grid_cell_ijk": np.zeros((2, 3), np.int32),
        "solver_grid_node_ijk": np.zeros((3, 3), np.int32),
        "solver_ws_impulse": np.zeros((3, 3), np.float32),
        "solver_ws_stress": np.zeros((2, 6), np.float32),
        "solver_body_q_prev": np.zeros((rocks + 1, 7), np.float32),
    }


def provenance():
    value = {name: f"{index + 1:064x}"
             for index, name in enumerate(checkpoint.SHA256_FIELDS)}
    value.update(source_manual_status="manual_contact_pilot_recorded",
                 source_domain="local_diagnostic_G3_loaded_leave_pile")
    return value


class LoadedHandoffCheckpointTest(unittest.TestCase):
    def test_roundtrip_keeps_controller_state_unaudited(self):
        values = arrays()
        controller = {
            "controller_target": [0.1, 0.2, 0.3, 0.4],
            "controller_target_velocity": [0.0] * 4,
            "bucket_integral_effort": [0.0, 0.0, 0.0, 59647.0],
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "loaded.npz"
            identity = checkpoint.write_checkpoint_atomic(path, values)
            document = checkpoint.metadata_document(
                identity, values, controller, provenance())
            metadata = Path(str(path) + ".json")
            checkpoint.write_metadata_atomic(metadata, document)
            loaded, restored, before = checkpoint.load_checkpoint_readonly(
                path, metadata, provenance(), expected_particle_count=3)
            self.assertEqual(before["sha256"], identity["sha256"])
            np.testing.assert_array_equal(loaded["particle_q"], values["particle_q"])
            self.assertEqual(restored["controller_state"], controller)
            self.assertFalse(restored["s07_development_handoff_eligible"])
            self.assertFalse(restored["training_eligible"])

    def test_rejects_integral_on_non_bucket_axis(self):
        controller = {
            "controller_target": [0.0] * 4,
            "controller_target_velocity": [0.0] * 4,
            "bucket_integral_effort": [1.0, 0.0, 0.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "integral_envelope"):
            checkpoint.validate_controller_state(controller)


if __name__ == "__main__":
    unittest.main()
