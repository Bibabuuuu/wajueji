import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_layered_mound_physical_preflight import build, load_config  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-physical-preflight-v1.json"


class LayeredMoundPhysicalPreflightTests(unittest.TestCase):
    def test_build_is_nonintersecting_deterministic_and_zero_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            report = build(CONFIG, ROOT, first)
            second_report = build(CONFIG, ROOT, second)
            self.assertEqual(report, second_report)
            for name in ("layered-mound-bank.npz", "scene-manifest.json",
                         "metrics.json", "report.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertEqual(report["status"], "nonintersecting_geometry_preflight_passed")
            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(report["stable_scene_claimed"])
            manifest = json.loads((first / "scene-manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rock_count"], 65)
            self.assertGreaterEqual(metrics["minimum_rock_rock_sat_gap_m"], 0.001 - 1e-12)
            self.assertEqual(metrics["retained_particle_center_inside_rock_count"], 0)
            with np.load(first / "layered-mound-bank.npz", allow_pickle=False) as archive:
                self.assertEqual(len(archive["q"]), metrics["retained_particle_count"])
                self.assertEqual(set(archive["activation_stage"].tolist()), set(range(6)))

    def test_rejects_visual_hash_and_scope_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["visual_config"]["sha256"] = "0" * 64
        attacks.append((candidate, "visual_binding"))
        candidate = copy.deepcopy(base)
        candidate["scope"]["stable_scene_claimed"] = True
        attacks.append((candidate, "scope"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_config(path, ROOT)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                build(CONFIG, ROOT, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
