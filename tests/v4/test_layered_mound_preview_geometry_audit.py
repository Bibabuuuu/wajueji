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

from audit_layered_mound_preview_geometry import audit  # noqa: E402


CONFIG = ROOT / "configs" / "v4" / "layered-mound-viewer-preview-v1.json"


class LayeredMoundPreviewGeometryAuditTests(unittest.TestCase):
    def test_approved_visual_baseline_is_honestly_rejected_for_intersections(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            report = audit(CONFIG, output)
            self.assertEqual(report["status"],
                             "approved_visual_geometry_rejected_for_physics")
            self.assertEqual(report["rock_count"], 65)
            self.assertEqual(report["sat_intersection_count"], 61)
            self.assertGreater(report["maximum_penetration_lower_bound_m"], 0.14)
            self.assertFalse(report["checks"]["rock_rock_sat_no_intersection"])
            self.assertTrue(report["checks"]["retained_particle_center_inside_rock_zero"])
            self.assertFalse(report["physics_input_authorized"])
            self.assertFalse(report["physical_manifest_published"])
            self.assertTrue((output / "audit.json").is_file())

    def test_visual_review_tamper_is_rejected_before_output(self):
        value = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
        value["visual_review"]["physics_stability_included"] = True
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "tampered.json"
            output = Path(temporary) / "output"
            config.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "visual_review_contract"):
                audit(config, output)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output_exists"):
                audit(CONFIG, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
