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

from run_layered_mound_one_rock_free_deposition_v2 import load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-one-rock-free-deposition-v2.json"


class LayeredMoundOneRockFreeDepositionV2Tests(unittest.TestCase):
    def test_only_presettle_collider_parking_changes(self):
        value, base = load_config(CONFIG, ROOT)
        correction = value["correction"]
        self.assertEqual(correction["presettle_parking_position_world_m"],
                         [0.0, 0.0, 5.0])
        self.assertEqual(
            correction["parked_assign_calls_including_fixture_initialization"], 61)
        self.assertFalse(correction["material_changed"])
        self.assertFalse(correction["stability_thresholds_changed"])
        self.assertEqual(base["deposition"]["surface_clearance_m"], 0.005)

    def test_rejects_binding_and_correction_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["base_candidate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "base_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["correction"]["material_changed"] = True
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
