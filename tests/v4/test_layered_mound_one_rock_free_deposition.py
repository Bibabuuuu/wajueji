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

from run_layered_mound_one_rock_free_deposition import load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-one-rock-free-deposition-v1.json"


class LayeredMoundOneRockFreeDepositionTests(unittest.TestCase):
    def test_free_from_first_dynamic_contact_contract(self):
        value, base = load_config(CONFIG, ROOT)
        deposition = value["deposition"]
        self.assertEqual(deposition["surface_clearance_m"], 0.005)
        self.assertTrue(deposition["rock_dynamic_from_first_mujoco_step"])
        self.assertFalse(deposition["kinematic_press"])
        self.assertFalse(deposition["held_contact_phase"])
        self.assertFalse(deposition["construction_support_force"])
        self.assertFalse(deposition["gravity_ramp"])
        self.assertEqual(deposition["required_continuous_stable_frames"], 120)
        self.assertEqual(base["targets"][0]["manifest_rock_id"], "lower-000")

    def test_rejects_binding_and_construction_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["failed_hold_release_gate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "report_binding"))
        candidate = copy.deepcopy(base)
        candidate["deposition"]["kinematic_press"] = True
        attacks.append((candidate, "deposition_contract"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
