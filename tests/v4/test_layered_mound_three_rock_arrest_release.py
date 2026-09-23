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

from run_layered_mound_three_rock_arrest_release import load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-three-rock-arrest-release-v1.json"


class LayeredMoundThreeRockArrestReleaseTests(unittest.TestCase):
    def test_single_arrest_and_uncontrolled_window_contract(self):
        value, bases = load_config(CONFIG, ROOT)
        correction = value["correction"]
        self.assertEqual(correction["arrest_frame_zero_based"], 60)
        self.assertEqual(correction["arrest_steps"], 1)
        self.assertEqual(
            correction["required_fully_uncontrolled_frames_after_arrest"], 120)
        self.assertFalse(correction["recurring_velocity_feedback"])
        self.assertEqual(bases["slow"]["correction"]["gravity_release_frames"], 60)
        self.assertEqual(bases["cover"]["targets"][2]["press_distance_m"], 0.016)

    def test_rejects_binding_and_correction_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["base_candidate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "base_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["correction"]["recurring_velocity_feedback"] = True
        attacks.append((candidate, "correction_contract"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
