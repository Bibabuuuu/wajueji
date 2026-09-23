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

from run_layered_mound_single_rock_press_smoke import (  # noqa: E402
    _foundation_particles,
    load_config,
)


CONFIG = ROOT / "configs" / "v4" / "layered-mound-single-rock-press-smoke-v1.json"


class LayeredMoundSingleRockPressSmokeTests(unittest.TestCase):
    def test_config_binds_narrow_controlled_press_scope(self):
        value = load_config(CONFIG, ROOT)
        self.assertEqual(value["target"]["manifest_rock_id"], "lower-000")
        self.assertFalse(value["target"]["free_fall"])
        self.assertEqual(value["scope"]["active_rock_count"], 1)
        self.assertFalse(value["scope"]["other_64_rocks_active"])
        self.assertEqual(value["scope"]["newton_solver_step_calls"], 60)
        self.assertFalse(value["scope"]["stable_scene_claimed"])

    def test_unexcavated_foundation_is_larger_than_carved_stage(self):
        value = load_config(CONFIG, ROOT)
        q = _foundation_particles(value, ROOT)
        self.assertEqual(q.shape, (85620, 3))
        with __import__("numpy").load(
                ROOT / "results/layered-mound-physical-preflight-v1/layered-mound-bank.npz",
                allow_pickle=False) as bank:
            carved_count = int((bank["activation_stage"] == 0).sum())
        self.assertEqual(carved_count, 79529)
        self.assertGreater(len(q), carved_count)

    def test_rejects_binding_and_scope_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["time_zero"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "time_zero_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["target"]["free_fall"] = True
        attacks.append((candidate, "target_contract"))
        candidate = copy.deepcopy(base)
        candidate["scope"]["other_64_rocks_active"] = True
        attacks.append((candidate, "scope_contract"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
