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

from build_mixed_material_capacity_preflight import build_capacity_model  # noqa: E402
from run_mixed_material_capacity_upper_bound import load_and_validate_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "mixed-material-capacity-upper-bound-v1.json"


class CapacityUpperBoundTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_marks_887_as_advisory_upper_bound_only(self):
        config, base = load_and_validate_config(CONFIG, ROOT)
        self.assertEqual(config["reference_rock_count"], 887)
        self.assertTrue(base.is_file())
        self.assertFalse(config["scope"]["formal_scene_count_claim"])
        self.assertFalse(config["scope"]["physics_qualified"])

    def test_rejects_fixed_scene_claim_count_and_scope_attacks(self):
        attacks = []
        value = copy.deepcopy(self.value)
        value["reference_rock_count"] = 240
        attacks.append((value, "reference_count_drift"))
        value = copy.deepcopy(self.value)
        value["scope"]["formal_scene_count_claim"] = True
        attacks.append((value, "scope_expanded"))
        value = copy.deepcopy(self.value)
        value["frozen_capacity_config"]["sha256"] = "0" * 64
        attacks.append((value, "binding_mismatch"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, error) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        ValueError, error):
                    load_and_validate_config(path, ROOT)

    def test_887_synthetic_model_is_deterministic_and_dimensioned(self):
        _, base = load_and_validate_config(CONFIG, ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.xml"
            second = Path(temporary) / "second.xml"
            first_manifest = build_capacity_model(base, ROOT, 887, first)
            second_manifest = build_capacity_model(base, ROOT, 887, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest["expected_nq"], 6228)
            self.assertEqual(first_manifest["expected_nv"], 5340)
            self.assertEqual(first_manifest["rock_count"], 887)
            self.assertFalse(first_manifest["formal_scene_qualified"])


if __name__ == "__main__":
    unittest.main()
