import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import layered_mound_checkpoint as checkpoint


class LayeredMoundCheckpointTest(unittest.TestCase):
    def _fixture(self, root: Path):
        particle_count, rock_count = 2, 1
        arrays = {
            "particle_q": np.zeros((particle_count, 3), np.float32),
            "particle_qd": np.zeros((particle_count, 3), np.float32),
            "particle_f": np.zeros((particle_count, 3), np.float32),
            "particle_flags": np.ones(particle_count, np.uint32),
            "particle_elastic_strain": np.zeros((particle_count, 3, 3), np.float32),
            "particle_transform": np.zeros((particle_count, 3, 3), np.float32),
            "particle_qd_grad": np.zeros((particle_count, 3, 3), np.float32),
            "particle_stress": np.zeros((particle_count, 3, 3), np.float32),
            "particle_Jp": np.ones(particle_count, np.float32),
            "rigid_ids": np.asarray(["r1-rock-000"]),
            "rigid_qpos": np.asarray([[1, 2, 3, 1, 0, 0, 0]], np.float64),
            "rigid_qvel": np.zeros((rock_count, 6), np.float64),
            "rigid_reported_linear_velocity": np.zeros((rock_count, 3), np.float64),
            "rigid_reported_angular_velocity": np.zeros((rock_count, 3), np.float64),
            "simulation_time_s": np.asarray([3.475], np.float64),
            "grid_reference_particle_q": np.asarray(
                [[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]], np.float32),
            "solver_grid_bounds_lo": np.asarray([-0.6, -0.6, -0.6], np.float32),
            "solver_grid_bounds_hi": np.asarray([0.65, 0.65, 0.65], np.float32),
            "solver_grid_resolution": np.asarray([25, 25, 25], np.int32),
            "solver_ws_impulse": np.zeros((7, 3), np.float32),
            "solver_ws_stress": np.zeros((5, 6), np.float32),
            "solver_body_q_prev": np.asarray(
                [[1, 2, 3, 1, 0, 0, 0]], np.float32),
        }
        path = root / "checkpoint.npz"
        np.savez(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        contract = {
            "sha256": digest,
            "checkpoint_schema": checkpoint.CHECKPOINT_SCHEMA,
            "checkpoint_status": "three_layer_continuation_checkpoint_saved",
            "source_config_sha256": "a" * 64,
            "source_runner_sha256": "b" * 64,
            "particle_count": particle_count,
            "rock_count": rock_count,
            "newton_version": "1.5.1",
            "warp_version": "1.17.0",
        }
        metadata = {
            "schema": checkpoint.CHECKPOINT_SCHEMA,
            "status": "three_layer_continuation_checkpoint_saved",
            "checkpoint_file": path.name,
            "checkpoint_sha256": digest,
            "source_config_sha256": contract["source_config_sha256"],
            "source_runner_sha256": contract["source_runner_sha256"],
            "particle_count": particle_count,
            "rock_count": rock_count,
            "simulation_time_s": 3.475,
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
        metadata_path = root / "checkpoint.npz.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return path, metadata_path, contract, arrays

    def test_readonly_load_checks_identity_and_all_fields(self):
        with tempfile.TemporaryDirectory() as raw:
            path, metadata, contract, expected = self._fixture(Path(raw))
            before = path.stat()
            arrays, loaded_metadata, identity = checkpoint.load_checkpoint_readonly(
                path, metadata, contract)
            after = path.stat()
            self.assertEqual(set(arrays), set(checkpoint.CHECKPOINT_FIELDS))
            self.assertTrue(all(np.array_equal(arrays[name], expected[name])
                                for name in expected))
            self.assertEqual(identity["sha256"], contract["sha256"])
            self.assertEqual(loaded_metadata["particle_count"], 2)
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(before.st_size, after.st_size)

    def test_hash_mismatch_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as raw:
            path, metadata, contract, _ = self._fixture(Path(raw))
            contract["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                checkpoint.load_checkpoint_readonly(path, metadata, contract)

    def test_contract_status_is_not_hardcoded_to_three_layers(self):
        with tempfile.TemporaryDirectory() as raw:
            path, metadata, contract, _ = self._fixture(Path(raw))
            contract["checkpoint_status"] = (
                "four_layer_continuation_checkpoint_saved")
            raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            raw_metadata["status"] = contract["checkpoint_status"]
            metadata.write_text(json.dumps(raw_metadata), encoding="utf-8")

            _, loaded_metadata, _ = checkpoint.load_checkpoint_readonly(
                path, metadata, contract)

            self.assertEqual(loaded_metadata["status"],
                             contract["checkpoint_status"])

    def test_metadata_file_must_bind_to_checkpoint_filename(self):
        with tempfile.TemporaryDirectory() as raw:
            path, metadata, contract, _ = self._fixture(Path(raw))
            raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            raw_metadata["checkpoint_file"] = "different-checkpoint.npz"
            metadata.write_text(json.dumps(raw_metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError,
                                        "checkpoint_metadata_file"):
                checkpoint.load_checkpoint_readonly(path, metadata, contract)

    def test_reported_body_velocity_may_differ_from_generalized_qvel(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, metadata, contract, arrays = self._fixture(root)
            arrays["rigid_qvel"][0] = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0]
            arrays["rigid_reported_linear_velocity"][0] = [7.0, 8.0, 9.0]
            arrays["rigid_reported_angular_velocity"][0] = [-1.0, -2.0, -3.0]
            np.savez(path, **arrays)
            contract["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            raw_metadata["checkpoint_sha256"] = contract["sha256"]
            metadata.write_text(json.dumps(raw_metadata), encoding="utf-8")

            loaded, loaded_metadata, _ = checkpoint.load_checkpoint_readonly(
                path, metadata, contract)

            self.assertTrue(np.array_equal(loaded["rigid_qvel"],
                                           arrays["rigid_qvel"]))
            diagnostics = loaded_metadata["load_diagnostics"]
            self.assertTrue(
                diagnostics["rigid_reported_velocity_fields_are_audit_only"])
            self.assertEqual(
                diagnostics["reported_linear_vs_qvel_max_abs_difference"],
                10.0)
            self.assertEqual(
                diagnostics["reported_angular_vs_qvel_max_abs_difference"],
                9.0)

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path, _, _, arrays = self._fixture(Path(raw))
            with self.assertRaises(FileExistsError):
                checkpoint.write_checkpoint_atomic(path, arrays)

    def test_v1_field_set_is_rejected_as_incomplete_continuation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, metadata, contract, arrays = self._fixture(root)
            for name in checkpoint.SOLVER_STATE_FIELDS:
                arrays.pop(name)
            np.savez(path, **arrays)
            contract["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
            raw_metadata["checkpoint_sha256"] = contract["sha256"]
            metadata.write_text(json.dumps(raw_metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_field_set"):
                checkpoint.load_checkpoint_readonly(path, metadata, contract)


if __name__ == "__main__":
    unittest.main()
