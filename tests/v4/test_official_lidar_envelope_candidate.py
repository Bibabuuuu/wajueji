import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from build_official_lidar_envelope_candidate import build, load_and_validate, sha256


CONFIG = ROOT / "configs" / "v4" / "official-lidar-a-envelope-candidate-v1.json"


class OfficialLidarEnvelopeCandidateTests(unittest.TestCase):
    def test_frozen_candidate_arithmetic_and_boundaries(self):
        config = load_and_validate(CONFIG)
        self.assertEqual(config["source_roi"]["selected_component_cell_count"], 310)
        self.assertEqual(config["source_roi"]["selected_component_support_point_count"], 7513)
        self.assertAlmostEqual(config["target_envelope"]["slope_angle_deg"], 21.0622)
        self.assertLess(config["target_envelope"]["crest_height_band_m"]["p95"], 1.6)
        self.assertIsNone(config["material_layout_intent"]["exact_rigid_rock_count"])

    def test_rejects_nonuniform_scale_and_premature_rock_count(self):
        original = json.loads(CONFIG.read_text(encoding="utf-8"))
        for mutation, error in (
            (("uniform_scaling", "scale", 0.5), "uniform_scale_mismatch"),
            (("material_layout_intent", "exact_rigid_rock_count", 60),
             "rigid_rock_count_prematurely_fixed"),
        ):
            candidate = copy.deepcopy(original)
            candidate[mutation[0]][mutation[1]] = mutation[2]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "candidate.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    load_and_validate(path)

    def test_two_builds_are_byte_identical_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_report = build(CONFIG, first)
            second_report = build(CONFIG, second)
            self.assertEqual(first_report, second_report)
            for name in ("preview.svg", "report.json", "README.md"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(first_report["preview_sha256"], sha256(first / "preview.svg"))
            self.assertFalse(first_report["physical_scene_modified"])
            self.assertFalse(first_report["training_started"])


if __name__ == "__main__":
    unittest.main()
