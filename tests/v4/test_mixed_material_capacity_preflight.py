import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_material_capacity_preflight import (  # noqa: E402
    build_capacity_model,
    load_and_validate_config,
    render_capacity_bridge,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-capacity-preflight-v1.json"


class CapacityConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_freezes_new_capacity_ladder_without_physics_claim(self):
        config = load_and_validate_config(CONFIG, ROOT)
        self.assertEqual(config["capacity"]["preflight_rock_counts"],
                         [64, 256, 512])
        self.assertEqual(config["capacity"]["maximum_formal_rock_count"], 1024)
        self.assertFalse(config["scope"]["one_clear_step"])
        self.assertFalse(config["scope"]["mpm_or_gpu_run"])
        self.assertFalse(config["scope"]["physics_qualified"])

    def test_rejects_old_ladder_fixed_count_hash_and_scope_attacks(self):
        attacks = []
        value = copy.deepcopy(self.config)
        value["capacity"]["preflight_rock_counts"] = [64, 128, 240]
        attacks.append((value, "capacity_contract_drift"))
        value = copy.deepcopy(self.config)
        value["capacity"]["runtime_count_source"] = "fixed_240"
        attacks.append((value, "capacity_contract_drift"))
        value = copy.deepcopy(self.config)
        value["frozen_inputs"]["two_rock_bridge"]["sha256"] = "0" * 64
        attacks.append((value, "frozen_input_mismatch"))
        value = copy.deepcopy(self.config)
        value["scope"]["physics_qualified"] = True
        attacks.append((value, "scope_expanded"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, error) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(error=error), self.assertRaisesRegex(
                        ValueError, error):
                    load_and_validate_config(path, ROOT)


class CapacityGenerationTests(unittest.TestCase):
    def test_models_are_deterministic_dimensioned_and_contiguously_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count in (64, 256, 512):
                first = root / f"first-{count}.xml"
                second = root / f"second-{count}.xml"
                first_manifest = build_capacity_model(CONFIG, ROOT, count, first)
                second_manifest = build_capacity_model(CONFIG, ROOT, count, second)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                self.assertEqual(first_manifest, second_manifest)
                self.assertEqual(first_manifest["expected_nq"], 19 + 7 * count)
                self.assertEqual(first_manifest["expected_nv"], 18 + 6 * count)
                tree = ET.parse(first)
                bodies = [body for body in tree.getroot().find("worldbody")
                          .findall("body")
                          if (body.get("name") or "").startswith("r1-rock-")]
                self.assertEqual([body.get("name") for body in bodies],
                                 [f"r1-rock-{index:03d}"
                                  for index in range(count)])
                self.assertTrue(all(body.find("freejoint").get("name") ==
                                    f"r1_rock_free_{index:03d}"
                                    for index, body in enumerate(bodies)))

    def test_bridge_uses_runtime_vectors_and_dynamic_named_wrench_contract(self):
        first, first_manifest = render_capacity_bridge(CONFIG, ROOT)
        second, second_manifest = render_capacity_bridge(CONFIG, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertIn("rock_count < 1 || rock_count > 1024", first)
        self.assertIn("applied_rock_mpm_force(rock_count)", first)
        self.assertIn("std::unordered_map<std::string, int> rock_index_by_name", first)
        self.assertIn("rock_index_by_name.find(body_name)", first)
        self.assertIn("std::vector<bool> seen_rocks(rock_count, false)", first)
        self.assertIn("external_wrench_count_value > rock_count + 1", first)
        self.assertIn("unknown or absent capacity rock wrench", first)
        self.assertIn(
            r'maximum_external_wrench_count\":" << rock_count + 1', first)
        self.assertNotIn("rock_count != 2", first)
        self.assertNotIn("std::array<bool, 2>", first)
        self.assertFalse(first_manifest["physics_qualified"])

    def test_rejects_invalid_or_layout_exhausting_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.xml"
            for count, error in ((0, "out_of_range"),
                                 (1025, "out_of_range")):
                with self.subTest(count=count), self.assertRaisesRegex(
                        ValueError, error):
                    build_capacity_model(CONFIG, ROOT, count, output)


if __name__ == "__main__":
    unittest.main()
