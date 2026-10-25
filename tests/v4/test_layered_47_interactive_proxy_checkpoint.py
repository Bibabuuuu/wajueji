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

import layered_47_interactive_proxy_checkpoint as checkpoint  # noqa: E402


def arrays(particles: int = 3, rocks: int = 47) -> dict[str, np.ndarray]:
    result = {
        "particle_q": np.zeros((particles, 3), dtype=np.float32),
        "particle_qd": np.zeros((particles, 3), dtype=np.float32),
        "particle_f": np.zeros((particles, 3), dtype=np.float32),
        "particle_flags": np.ones((particles,), dtype=np.int32),
        "particle_elastic_strain": np.zeros(
            (particles, 3, 3), dtype=np.float32),
        "particle_transform": np.zeros((particles, 3, 3), dtype=np.float32),
        "particle_qd_grad": np.zeros((particles, 3, 3), dtype=np.float32),
        "particle_stress": np.zeros((particles, 3, 3), dtype=np.float32),
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
        "solver_grid_transform": np.eye(3, dtype=np.float32),
        "solver_grid_cell_ijk": np.zeros((2, 3), dtype=np.int32),
        "solver_grid_node_ijk": np.zeros((2, 3), dtype=np.int32),
        "solver_ws_impulse": np.zeros((2, 3), dtype=np.float32),
        "solver_ws_stress": np.zeros((2, 6), dtype=np.float32),
        "solver_body_q_prev": np.zeros((rocks + 1, 7), dtype=np.float32),
    }
    return result


class InteractiveProxyCheckpointTest(unittest.TestCase):
    def test_write_contract_accepts_bucket_expanded_solver_history(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "proxy.npz"
            identity = checkpoint.write_checkpoint_atomic(target, arrays())
            self.assertEqual(identity["path"], str(target.resolve()))
            with np.load(target, allow_pickle=False) as archive:
                self.assertEqual(archive["solver_body_q_prev"].shape, (48, 7))

    def test_readonly_loader_keeps_nonformal_scope(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "proxy.npz"
            values = arrays(particles=256441)
            identity = checkpoint.write_checkpoint_atomic(target, values)
            metadata = Path(raw) / "proxy.npz.json"
            metadata.write_text(json.dumps({
                "schema": checkpoint.CHECKPOINT_SCHEMA,
                "status": checkpoint.CHECKPOINT_STATUS,
                "checkpoint_file": target.name,
                "checkpoint_sha256": identity["sha256"],
                "particle_count": 256441,
                "rock_count": 47,
                "solver_grid_type": "sparse",
                "solver_voxel_size_m": 0.1,
                "solver_grid_padding": 2,
                "manual_collection_eligible": False,
                "training_eligible": False,
                "formal_a30_equivalent": False,
            }), encoding="utf-8")
            loaded, document, before = checkpoint.load_checkpoint_readonly(
                target, metadata)
            self.assertEqual(before["sha256"], identity["sha256"])
            self.assertEqual(loaded["mechanism_qpos"].shape, (19,))
            self.assertFalse(document["manual_collection_eligible"])

    def test_rejects_missing_bucket_solver_history(self):
        values = arrays()
        values["solver_body_q_prev"] = np.zeros((47, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(
                    ValueError, "solver_body_q_prev"):
                checkpoint.write_checkpoint_atomic(
                    Path(raw) / "proxy.npz", values)


if __name__ == "__main__":
    unittest.main()
