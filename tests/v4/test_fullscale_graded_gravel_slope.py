import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "v4"))

from build_fullscale_graded_gravel_slope import (  # noqa: E402
    FROZEN_CLASSES,
    _descriptors,
    _deterministic_npz_bytes,
    _load_frozen_bank,
    _load_prototypes,
    build,
    load_config,
    validate_config,
)


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "v4" / "fullscale-graded-gravel-slope-v1.json"


class D060ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_canonical_config_is_valid_and_has_no_fixed_count(self):
        validate_config(copy.deepcopy(self.config))
        self.assertNotIn("total_count", self.config["rocks"])
        self.assertTrue(all("count" not in item
                            for item in self.config["rocks"]["size_classes"]))
        self.assertTrue(self.config["claims"]["geometry_only"])

    def test_rejects_old_240_recipe_old_ranges_and_fixed_count(self):
        mutations = [
            (lambda value: value["rocks"].update(total_count=240),
             "fixed_rock_count_forbidden"),
            (lambda value: value["rocks"]["size_classes"][0].update(
                diameter_range_m=[0.08, 0.14]), "size_class_contract_drift"),
            (lambda value: value["rocks"]["size_classes"][2].update(
                diameter_range_m=[0.20, 0.27]), "size_class_contract_drift"),
            (lambda value: value["rocks"]["size_classes"][0].update(count=840),
             "size_class_contract_drift"),
        ]
        for mutate, error in mutations:
            candidate = copy.deepcopy(self.config)
            mutate(candidate)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                validate_config(candidate)

    def test_rejects_fraction_volume_depth_and_capability_drift(self):
        mutations = [
            (lambda value: value["material_recipe"].update(small_rigid_fraction=0.59),
             "material_recipe_fraction_drift"),
            (lambda value: value["rocks"]["size_classes"][1].update(
                target_volume_m3=0.5), "size_class_contract_drift"),
            (lambda value: value["placement"].update(deep_depth_threshold_m=0.0),
             "invalid_deep_depth_threshold"),
            (lambda value: value["placement"].update(minimum_large_deep_fraction=0.0),
             "invalid_placement_fraction"),
            (lambda value: value["material_recipe"].update(
                relative_total_volume_tolerance=0.03), "invalid_total_volume_tolerance"),
            (lambda value: value["claims"].update(physics_executed=True),
             "forbidden_capability_claim"),
            (lambda value: value["claims"].update(physics_qualified=True),
             "forbidden_capability_claim"),
            (lambda value: value["claims"].update(three_scoop_qualified=True),
             "forbidden_capability_claim"),
        ]
        for mutate, error in mutations:
            candidate = copy.deepcopy(self.config)
            mutate(candidate)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                validate_config(candidate)

    def test_rejects_input_asset_and_carve_policy_tampering(self):
        mutations = [
            (lambda value: value["inputs"]["initial_bank_npz"].update(sha256="0" * 64),
             "bank_input_hash_contract_drift"),
            (lambda value: value["inputs"]["initial_bank_npz"].update(
                q_array_sha256="0" * 64), "bank_q_hash_contract_drift"),
            (lambda value: value["inputs"]["solid_bank_source"].update(sha256="0" * 64),
             "solid_bank_hash_contract_drift"),
            (lambda value: value["rocks"]["assets"].update(angular_octa="0" * 64),
             "rock_asset_contract_drift"),
            (lambda value: value["placement"].update(carve_policy="delete_extra_fines"),
             "carve_policy_drift"),
        ]
        for mutate, error in mutations:
            candidate = copy.deepcopy(self.config)
            mutate(candidate)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                validate_config(candidate)


class D060DerivedRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        cls.descriptors = _descriptors(cls.config)

    def test_counts_are_derived_and_diameters_and_volumes_close(self):
        expected_counts = {
            "small_0p16_0p24": 840,
            "medium_0p24_0p32": 42,
            "large_0p32_0p40": 5,
        }
        self.assertEqual(len(self.descriptors), 887)
        for name, low, high, target, _ in FROZEN_CLASSES:
            selected = [item for item in self.descriptors if item["size_class"] == name]
            self.assertEqual(len(selected), expected_counts[name])
            self.assertTrue(all(low < item["diameter_m"] < high for item in selected))
            actual = sum(math.pi / 6.0 * item["diameter_m"] ** 3
                         for item in selected)
            self.assertLessEqual(abs(actual - target) / target, 5e-12)

    def test_prototypes_are_balanced_within_each_derived_class(self):
        for name, *_ in FROZEN_CLASSES:
            selected = [item for item in self.descriptors if item["size_class"] == name]
            counts = {prototype: sum(item["prototype_id"] == prototype
                                     for item in selected)
                      for prototype in self.config["rocks"]["assets"]}
            self.assertEqual(len(counts), 4)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_frozen_bank_and_assets_are_recomputed_from_hash_bound_inputs(self):
        q = _load_frozen_bank(self.config, REPOSITORY_ROOT)
        self.assertEqual(q.shape, (390400, 3))
        self.assertEqual(q.dtype, np.float32)
        prototypes = _load_prototypes(self.config, REPOSITORY_ROOT)
        self.assertEqual(set(prototypes), set(self.config["rocks"]["assets"]))
        for prototype in prototypes.values():
            self.assertAlmostEqual(prototype["unit_volume_m3"], math.pi / 6.0, places=12)

    def test_npz_serialization_has_fixed_zip_metadata(self):
        arrays = {"q": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
                  "material_id": np.asarray([0], dtype=np.uint8)}
        self.assertEqual(_deterministic_npz_bytes(arrays),
                         _deterministic_npz_bytes(arrays))


class D060BoundedPackingBlockerTests(unittest.TestCase):
    def test_current_layered_grid_sat_attempt_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "formal-output-must-not-exist"
            with self.assertRaisesRegex(
                    ValueError,
                    r"layered_grid_sat_relaxation_failed:rocks=887:iterations=60"):
                build(CONFIG_PATH, output, REPOSITORY_ROOT)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
