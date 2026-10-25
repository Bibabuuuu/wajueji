import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_mound_thirtythree_rock_three_layer as candidate


class ThirtyThreeRockThreeLayerContractTests(unittest.TestCase):
    def test_contract_freezes_three_layers_without_full_scene_claim(self):
        config = candidate.load_config(
            ROOT / "configs/v4/layered-mound-thirtythree-rock-three-layer-v1.json", ROOT)
        self.assertEqual(len(config["targets"]), 33)
        self.assertEqual(sum(x["active_count"] for x in config["particle_bands"]), 221882)
        self.assertEqual(config["scope"]["completed_layers"], 3)
        self.assertFalse(config["scope"]["full_scene"])

    def test_prior_layers_are_unchanged_and_only_three_middle_rocks_lift(self):
        config = candidate.load_config(
            ROOT / "configs/v4/layered-mound-thirtythree-rock-three-layer-v1.json", ROOT)
        third = [x for x in config["targets"] if x["stage"] == "middle"]
        self.assertEqual(len(third), 13)
        self.assertEqual(sum(x["vertical_clearance_lift_m"] > 0 for x in third), 3)
        previous = json.loads((
            ROOT / "configs/v4/layered-mound-twenty-rock-two-layer-v1.json"
        ).read_text())
        self.assertEqual(config["targets"][:20], previous["targets"])


if __name__ == "__main__":
    unittest.main()
