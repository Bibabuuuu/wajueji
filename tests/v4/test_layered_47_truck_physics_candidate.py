import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import layered_47_truck_physics_candidate as candidate


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v4/layered-47-truck-physics-v1.json"


class TruckPhysicsCandidateTest(unittest.TestCase):
    def test_frozen_candidate_loads_five_cad_aligned_boxes(self):
        document, fixtures = candidate.load_candidate(CONFIG, ROOT)
        self.assertEqual(document["schema"], candidate.CONFIG_SCHEMA)
        self.assertEqual(tuple(item["name"] for item in fixtures),
                         candidate.FIXTURE_NAMES)
        for fixture in fixtures:
            rotation = np.asarray(fixture["rotation_world"])
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            self.assertAlmostEqual(np.linalg.norm(
                fixture["quaternion_wxyz"]), 1.0)

    def test_clearance_distinguishes_inside_and_outside(self):
        _, fixtures = candidate.load_candidate(CONFIG, ROOT)
        center = np.asarray(fixtures[0]["position_world_m"])
        result = candidate.clearance(np.asarray([
            center, center + np.asarray([20.0, 0.0, 0.0])]), fixtures)
        self.assertEqual(result["inside_any_fixture_count"], 1)
        self.assertEqual(result["minimum_point_center_clearance_m"], 0.0)

    def test_mujoco_injection_uses_reserved_truck_bit(self):
        _, fixtures = candidate.load_candidate(CONFIG, ROOT)
        xml = (
            "<mujoco><worldbody><body name='r1-rock-000'>"
            "<geom name='r1_rock_geom_000' contype='256' "
            "conaffinity='3840'/></body></worldbody></mujoco>")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "model.xml"
            path.write_text(xml, encoding="utf-8")
            manifest = candidate.append_mujoco_fixtures(path, fixtures, {
                "rock": 256, "truck_receiver_reserved": 2048})
            root = ET.parse(path).getroot()
            geoms = [geom for geom in root.iter("geom")
                     if (geom.get("name") or "").startswith("r1_truck_")]
        self.assertEqual(manifest["fixture_count"], 5)
        self.assertEqual(len(geoms), 5)
        self.assertTrue(all(geom.get("contype") == "2048" for geom in geoms))
        self.assertTrue(all(geom.get("conaffinity") == "256" for geom in geoms))

    def test_source_hash_drift_is_rejected(self):
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document["frozen_inputs"]["receiver_contract"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truck_source_mismatch"):
                candidate.load_candidate(path, ROOT)


if __name__ == "__main__":
    unittest.main()
