import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from run_layered_mound_three_rock_adaptive_press import load_config  # noqa: E402


CONFIG = ROOT / "configs/v4/layered-mound-three-rock-adaptive-press-v1.json"


class LayeredMoundThreeRockAdaptivePressTests(unittest.TestCase):
    def test_variable_press_and_free_stability_scope(self):
        value = load_config(CONFIG, ROOT)
        self.assertEqual(
            [item["press_distance_m"] for item in value["targets"]],
            [0.012, 0.008, 0.016])
        self.assertTrue(value["acceptance"]["rock_translation_during_release_allowed"])
        self.assertTrue(value["acceptance"]["rock_rotation_during_release_allowed"])
        self.assertFalse(value["acceptance"]["final_manifest_pose_required"])
        self.assertEqual(value["scope"]["active_rock_count"], 3)
        self.assertFalse(value["scope"]["full_lower_layer"])

    def test_rejects_gate_target_and_acceptance_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["single_press_gate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "single_press_gate_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["targets"][1]["press_distance_m"] = 0.012
        attacks.append((candidate, "target_contract"))
        candidate = copy.deepcopy(base)
        candidate["acceptance"]["rock_rotation_during_release_allowed"] = False
        attacks.append((candidate, "acceptance_contract"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
