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

from run_layered_mound_time_zero import load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-time-zero-v1.json"


class LayeredMoundTimeZeroTests(unittest.TestCase):
    def test_config_binds_65_rock_geometry_and_zero_step_scope(self):
        value = load_config(CONFIG, ROOT)
        self.assertEqual(value["expected"]["rock_count"], 65)
        self.assertEqual(value["expected"]["particle_count"], 263673)
        self.assertEqual(value["expected"]["activation_stage_count"], 6)
        self.assertEqual(value["scope"]["mj_step_calls"], 0)
        self.assertEqual(value["scope"]["newton_solver_step_calls"], 0)
        self.assertFalse(value["scope"]["stable_scene_claimed"])

    def test_rejects_geometry_and_scope_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["geometry"]["manifest_sha256"] = "0" * 64
        attacks.append((candidate, "geometry_manifest_binding"))
        candidate = copy.deepcopy(base)
        candidate["scope"]["mj_step_calls"] = 1
        attacks.append((candidate, "scope"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
