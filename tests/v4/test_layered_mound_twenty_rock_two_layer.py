from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_mound_twenty_rock_two_layer as candidate


class TwentyRockTwoLayerContractTests(unittest.TestCase):
    def test_contract_freezes_two_layers_without_full_scene_claim(self):
        config = candidate.load_config(
            ROOT / "configs/v4/layered-mound-twenty-rock-two-layer-v1.json", ROOT)
        self.assertEqual(len(config["targets"]), 20)
        self.assertEqual(sum(x["active_count"] for x in config["particle_bands"]), 180554)
        self.assertEqual(config["scope"]["completed_layers"], 2)
        self.assertFalse(config["scope"]["full_scene"])

    def test_only_second_layer_receives_clearance_lifts(self):
        config = candidate.load_config(
            ROOT / "configs/v4/layered-mound-twenty-rock-two-layer-v1.json", ROOT)
        lower = [x for x in config["targets"] if x["stage"] == "lower"]
        second = [x for x in config["targets"] if x["stage"] == "lower_middle"]
        self.assertEqual(len(lower), 9)
        self.assertEqual(len(second), 11)
        self.assertTrue(all(x["vertical_clearance_lift_m"] == 0.0 for x in lower))
        self.assertGreater(max(x["vertical_clearance_lift_m"] for x in second), 0.0)


if __name__ == "__main__":
    unittest.main()
