from pathlib import Path
import unittest

import numpy as np

import run_layered_47_excavation_preflight as preflight47
import run_layered_65_excavation_preflight as preflight65


ROOT = Path(__file__).resolve().parents[2]


class LayeredExcavationPreflightProfilesTest(unittest.TestCase):
    def test_47_profile_keeps_the_validated_contract_and_record_order(self):
        config = preflight47.load_config(
            ROOT / "configs/v4/layered-47-excavation-preflight-v1.json",
            ROOT)
        self.assertEqual(config["input_checkpoint"]["rock_count"], 47)
        self.assertEqual(len(preflight47.DEFAULT_PROFILE["expected_ids"]), 47)
        self.assertEqual(
            preflight47.DEFAULT_PROFILE["expected_ids"][-1], "upper-013")
        self.assertEqual(
            preflight47.DEFAULT_PROFILE["passed_status"],
            "layered_47_excavation_preflight_passed")

    def test_65_profile_only_expands_the_checkpoint_and_layer_contract(self):
        config = preflight65.load_config(
            ROOT / "configs/v4/layered-65-excavation-preflight-v1.json",
            ROOT)
        source = config["input_checkpoint"]
        self.assertEqual(source["rock_count"], 65)
        self.assertEqual(source["particle_count"], 266091)
        self.assertEqual(
            source["sha256"],
            "22fe16dfecc178257754d6646142de29a231ff91e0f09bee03210a414d64e9b7")
        self.assertEqual(len(preflight65.PROFILE["expected_ids"]), 65)
        self.assertEqual(preflight65.PROFILE["expected_ids"][-1], "top-017")
        self.assertEqual(
            preflight65.PROFILE["passed_status"],
            "layered_65_excavation_preflight_passed")

    def test_profiles_share_the_same_precontact_gate_semantics(self):
        shared_keys = {
            "passed_next_gate", "failed_next_gate",
        }
        for key in shared_keys:
            self.assertEqual(
                preflight47.DEFAULT_PROFILE[key], preflight65.PROFILE[key])

    def test_postsolve_projection_records_correction_without_rigid_routing(self):
        before_q = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        before_qd = np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        after_q = before_q.copy()
        after_q[0, 2] += 0.02
        after_qd = before_qd.copy()
        after_qd[0, 2] = 1.0
        result = preflight47.postsolve_projection_diagnostics(
            before_q, before_qd, after_q, after_qd,
            particle_mass_kg=3.0, base_query_gap_m=0.1, dt_s=0.1,
            applied=True)
        self.assertTrue(result["enabled"])
        self.assertTrue(result["applied"])
        self.assertEqual(result["corrected_particle_count"], 1)
        self.assertAlmostEqual(result["query_gap_m"], 0.25)
        self.assertEqual(
            result["reaction_routing"],
            "unattributed_diagnostic_not_applied_to_mujoco")
        self.assertFalse(result["continuous_collision_proven"])
        self.assertAlmostEqual(
            result["particle_momentum_delta_kg_m_s"][2], 3.0)


if __name__ == "__main__":
    unittest.main()
