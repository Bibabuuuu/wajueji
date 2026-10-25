import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_mound_three_layer_checkpoint as candidate


class ThreeLayerCheckpointContractTests(unittest.TestCase):
    def test_checkpoint_carries_all_mutable_mpm_history(self):
        self.assertEqual(candidate.MPM_STATE_FIELDS, (
            "particle_elastic_strain", "particle_transform", "particle_qd_grad",
            "particle_stress", "particle_Jp"))
        self.assertEqual(candidate.SOLVER_STATE_FIELDS, (
            "grid_reference_particle_q",
            "solver_grid_bounds_lo", "solver_grid_bounds_hi",
            "solver_grid_resolution", "solver_ws_impulse",
            "solver_ws_stress", "solver_body_q_prev"))

    def test_checkpoint_schema_is_continuation_specific(self):
        self.assertEqual(
            candidate.CHECKPOINT_SCHEMA,
            "excavation_layered_mound_continuation_checkpoint_v2")


if __name__ == "__main__":
    unittest.main()
