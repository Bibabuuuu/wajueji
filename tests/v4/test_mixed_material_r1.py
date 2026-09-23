import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from mixed_material_r1 import build_probe_xml, build_proxy  # noqa: E402
from mixed_material_r1_mechanism import build_candidate, sha256  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r1.json"
R0_CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-r0.json"
INVENTORY_PATH = Path(
    "/home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json")
SOURCE_MODEL = (ROOT / "evidence" / "autonomous-multiscoop-2026-09-14-v1"
                / "mechanism-auto8" / "mechanism.xml")


class OpenBucketProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        cls.proxy = build_proxy(cls.inventory, cls.config)

    def test_proxy_is_semantic_open_cavity_decomposition(self):
        self.assertEqual(len(self.proxy["shell_boxes"]), 11)
        self.assertEqual(len(self.proxy["side_plate_meshes"]), 2)
        self.assertEqual(len(self.proxy["tooth_boxes"]), 5)
        self.assertEqual(self.proxy["geom_count"], 18)
        self.assertFalse(self.proxy["has_mouth_cap"])
        self.assertEqual(self.proxy["construction"],
                         "semantic_open_cavity_segmented_convex_proxy")
        self.assertEqual(self.proxy["shell_curve_original_point_count"], 55)

    def test_side_plates_are_thin_closed_prisms_outside_inner_width(self):
        inner = self.inventory["bucket"]["y_limits_m"]
        negative, positive = self.proxy["side_plate_meshes"]
        self.assertAlmostEqual(negative["inner_y_m"], inner[0])
        self.assertLess(negative["outer_y_m"], negative["inner_y_m"])
        self.assertAlmostEqual(positive["inner_y_m"], inner[1])
        self.assertGreater(positive["outer_y_m"], positive["inner_y_m"])
        for mesh in (negative, positive):
            vertices = np.asarray(mesh["vertices_bucket_m"])
            faces = mesh["faces"]
            self.assertEqual(vertices.shape, (24, 3))
            self.assertEqual(len(faces), 44)
            edge_counts = {}
            for face in faces:
                for first, second in zip(face, face[1:] + face[:1]):
                    edge = tuple(sorted((first, second)))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            self.assertTrue(all(count == 2 for count in edge_counts.values()))

    def test_central_mouth_has_room_for_027m_probe(self):
        self.assertGreater(
            self.proxy["central_mouth_point_proxy_clearance_m"], 0.135)
        self.assertAlmostEqual(self.proxy["inner_clear_width_m"], 0.59949)

    def test_probe_fixtures_do_not_restore_whole_bucket_mesh(self):
        for diameter in (0.12, 0.27):
            xml = build_probe_xml(
                self.proxy, self.config, diameter,
                "../../../assets/v4/mixed-material-r1")
            self.assertIn('<freejoint name="rock_free"/>', xml)
            self.assertIn('name="mouth_inward"', xml)
            self.assertNotIn('mesh="bucket"', xml)
            self.assertNotIn('bucket.obj', xml)


class MainSceneDistributionTests(unittest.TestCase):
    def test_main_staging_scene_is_small_rock_dominant(self):
        config = json.loads(R0_CONFIG_PATH.read_text(encoding="utf-8"))
        scene = next(item for item in config["scenes"]
                     if item["size_bucket"] == "main_training")
        diameters = scene["equivalent_diameters_m"]
        small_fraction = sum(value <= 0.16 for value in diameters) / len(diameters)
        large_volume_fraction = (
            sum(value ** 3 for value in diameters if value >= 0.22)
            / sum(value ** 3 for value in diameters))
        self.assertGreaterEqual(small_fraction, 0.8)
        self.assertLessEqual(max(diameters), 0.23)
        self.assertEqual(sum(value >= 0.27 for value in diameters), 0)
        self.assertLessEqual(large_volume_fraction, 0.2)

    def test_large_rocks_are_rare_and_boundary_rocks_are_separate(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        policy = config["main_scene_distribution"]
        self.assertEqual(policy["large_rock_count_max_per_scene"], 1)
        self.assertLessEqual(
            policy["large_rock_rigid_solid_volume_fraction_max"], 0.2)
        self.assertLessEqual(policy["scenes_with_0_27_m_rock_fraction_max"], 0.25)
        self.assertTrue(
            policy["jam_boundary_is_separate_from_success_demonstrations"])
        self.assertGreater(policy["jam_boundary_min_equivalent_diameter_m"], 0.27)


class MechanismCandidateTests(unittest.TestCase):
    def test_candidate_is_isolated_and_has_one_rock_and_open_proxy(self):
        source_before = sha256(SOURCE_MODEL)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mechanism-r1.xml"
            manifest = build_candidate(
                SOURCE_MODEL, output, ROOT, INVENTORY_PATH, CONFIG_PATH)
            tree = ET.parse(output)
            root = tree.getroot()
            bucket_visual = next(item for item in root.iter("geom")
                                 if item.get("name") == "bucket")
            self.assertEqual(bucket_visual.get("contype"), "0")
            self.assertEqual(bucket_visual.get("conaffinity"), "0")
            proxy_geoms = [item for item in root.iter("geom")
                           if item.get("name", "").startswith("r1_bucket_")]
            self.assertEqual(len(proxy_geoms), 18)
            rock = next(item for item in root.iter("body")
                        if item.get("name") == "r1-rock-000")
            self.assertEqual(len(list(rock.iter("freejoint"))), 1)
            inertial = next(rock.iter("inertial"))
            self.assertAlmostEqual(float(inertial.get("mass")),
                                   manifest["rock_mass_kg"])
            self.assertEqual(manifest["expected_candidate_contract"]["nq"], 26)
            self.assertEqual(manifest["expected_candidate_contract"]["nv"], 24)
            self.assertTrue(manifest["frozen_source_modified"] is False)
            slope_output = Path(temporary) / "mechanism-r1-slope.xml"
            slope_manifest = build_candidate(
                SOURCE_MODEL, slope_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                "slope_roll")
            slope_root = ET.parse(slope_output).getroot()
            ramp = next(item for item in slope_root.iter("geom")
                        if item.get("name") == "r1_ground_ramp")
            self.assertEqual(ramp.get("contype"), "1024")
            self.assertEqual(ramp.get("conaffinity"), "256")
            self.assertEqual(slope_manifest["unit_scenario"], "slope_roll")
            self.assertEqual(slope_manifest["unit_fixture"]["slope_degrees"], 15.0)
            collision_output = Path(temporary) / "mechanism-r1-collision.xml"
            collision_manifest = build_candidate(
                SOURCE_MODEL, collision_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                "rock_collision")
            collision_root = ET.parse(collision_output).getroot()
            collision_rocks = [item for item in collision_root.iter("body")
                               if item.get("name", "").startswith("r1-rock-")]
            self.assertEqual(len(collision_rocks), 2)
            self.assertEqual(
                collision_manifest["expected_candidate_contract"]["nq"], 33)
            self.assertEqual(
                collision_manifest["expected_candidate_contract"]["nv"], 30)
            self.assertAlmostEqual(
                collision_manifest["total_rock_mass_kg"],
                sum(item["mass_kg"] for item in collision_manifest["rocks"]))
            for scenario_name in ("bucket_push", "bucket_push_clear"):
                push_output = Path(temporary) / f"mechanism-r1-{scenario_name}.xml"
                push_manifest = build_candidate(
                    SOURCE_MODEL, push_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                    scenario_name)
                push_root = ET.parse(push_output).getroot()
                shelf = next(item for item in push_root.iter("geom")
                             if item.get("name") == "r1_ground_push_shelf")
                self.assertEqual(shelf.get("contype"), "1024")
                self.assertEqual(shelf.get("conaffinity"), "256")
                self.assertEqual(push_manifest["unit_scenario"], scenario_name)
            entry_output = Path(temporary) / "mechanism-r1-entry.xml"
            entry_manifest = build_candidate(
                SOURCE_MODEL, entry_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                "mechanism_entry")
            entry_root = ET.parse(entry_output).getroot()
            entry_shelf = next(item for item in entry_root.iter("geom")
                               if item.get("name") == "r1_ground_entry_shelf")
            self.assertEqual(entry_shelf.get("contype"), "1024")
            self.assertEqual(entry_shelf.get("conaffinity"), "256")
            self.assertEqual(entry_manifest["unit_scenario"], "mechanism_entry")
            self.assertEqual(len(entry_manifest["mouth_center_bucket_m"]), 3)
            self.assertEqual(
                len(entry_manifest["mouth_inward_direction_bucket"]), 3)
            hold_output = Path(temporary) / "mechanism-r1-hold.xml"
            hold_manifest = build_candidate(
                SOURCE_MODEL, hold_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                "mechanism_hold")
            hold_root = ET.parse(hold_output).getroot()
            hold_shelf = next(item for item in hold_root.iter("geom")
                              if item.get("name") == "r1_ground_hold_shelf")
            self.assertEqual(hold_shelf.get("contype"), "1024")
            self.assertEqual(hold_shelf.get("conaffinity"), "256")
            self.assertEqual(hold_manifest["unit_scenario"], "mechanism_hold")
            dump_output = Path(temporary) / "mechanism-r1-dump.xml"
            dump_manifest = build_candidate(
                SOURCE_MODEL, dump_output, ROOT, INVENTORY_PATH, CONFIG_PATH,
                "mechanism_dump")
            dump_root = ET.parse(dump_output).getroot()
            truck_geoms = [item for item in dump_root.iter("geom")
                           if item.get("name", "").startswith("r1_truck_")]
            self.assertEqual(len(truck_geoms), 5)
            self.assertTrue(all(item.get("contype") == "2048"
                                for item in truck_geoms))
            self.assertTrue(all(item.get("conaffinity") == "256"
                                for item in truck_geoms))
            self.assertEqual(dump_manifest["unit_scenario"], "mechanism_dump")
        self.assertEqual(sha256(SOURCE_MODEL), source_before)

    def test_contact_bits_are_pairwise_distinct(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        values = list(config["mechanism_candidate"]["contact_bits"].values())
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(value > 0 and value & (value - 1) == 0
                            for value in values))


if __name__ == "__main__":
    unittest.main()
