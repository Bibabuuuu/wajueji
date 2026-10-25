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

from run_layered_mound_three_rock_slow_release import load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-three-rock-slow-release-v1.json"


class LayeredMoundThreeRockSlowReleaseTests(unittest.TestCase):
    def test_only_release_ramp_scope_changes(self):
        value, base = load_config(CONFIG, ROOT)
        correction = value["correction"]
        self.assertEqual(correction["gravity_release_frames"], 60)
        self.assertEqual(correction["required_fully_released_stable_frames"], 60)
        self.assertFalse(correction["mpm_material_changed"])
        self.assertFalse(correction["rigid_damping_changed"])
        self.assertFalse(correction["stability_thresholds_changed"])
        self.assertEqual(base["targets"][0]["press_distance_m"], 0.012)
        self.assertTrue(value["acceptance"][
            "rock_rotation_during_release_allowed"])

    def test_rejects_binding_and_correction_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["base_candidate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "base_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["correction"]["rigid_damping_changed"] = True
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
