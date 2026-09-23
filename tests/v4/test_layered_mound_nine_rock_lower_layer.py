import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))

import run_layered_mound_nine_rock_lower_layer as candidate


class NineRockLowerLayerContractTests(unittest.TestCase):
    def test_frozen_contract_has_full_lower_layer_without_full_scene_claim(self):
        config = candidate.load_config(
            ROOT / "configs/v4/layered-mound-nine-rock-lower-layer-v1.json", ROOT)
        self.assertEqual(
            [item["manifest_rock_id"] for item in config["targets"]],
            candidate.TARGET_IDS)
        self.assertEqual(
            [(item["press_distance_m"], item["press_frames"])
             for item in config["targets"]], candidate.PRESS_SCHEDULE)
        self.assertEqual(config["particles"]["combined_particle_count"], 128881)
        self.assertTrue(config["scope"]["full_lower_layer"])
        self.assertFalse(config["scope"]["full_scene"])
        self.assertFalse(config["scope"]["stable_scene_claimed"])

    def test_press_speeds_stay_close_to_qualified_three_rock_schedule(self):
        config = json.loads((
            ROOT / "configs/v4/layered-mound-nine-rock-lower-layer-v1.json"
        ).read_text())
        dt = config["probe"]["output_dt_s"]
        speeds = [distance / (frames * dt)
                  for distance, frames in candidate.PRESS_SCHEDULE]
        self.assertGreaterEqual(min(speeds), 0.076)
        self.assertLessEqual(max(speeds), 0.08)


if __name__ == "__main__":
    unittest.main()
