from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_mixed_coupled_bridge_r3_c import render_bridge as render_base  # noqa: E402
from build_mixed_coupled_bridge_r3_c_dynamics import (  # noqa: E402
    READY_SCHEMA,
    STATE_SCHEMA,
    render_bridge,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-r3-c.json"


class MixedMaterialR3CDynamicsBridgeTest(unittest.TestCase):
    def test_generation_is_deterministic_and_keeps_base_frozen(self):
        base_first, base_manifest_first = render_base(CONFIG, ROOT)
        first, first_manifest = render_bridge(CONFIG, ROOT)
        second, second_manifest = render_bridge(CONFIG, ROOT)
        base_second, base_manifest_second = render_base(CONFIG, ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(base_first, base_second)
        self.assertEqual(base_manifest_first, base_manifest_second)
        self.assertEqual(first_manifest["ready_schema"], READY_SCHEMA)
        self.assertEqual(first_manifest["state_schema"], STATE_SCHEMA)
        self.assertFalse(
            first_manifest["frozen_r1_r2_r3_a_r3_b_r3_c_base_modified"])
        self.assertFalse(first_manifest["physics_qualified"])

    def test_bridge_echoes_per_rock_reference_and_rigid_partition(self):
        source, manifest = render_bridge(CONFIG, ROOT)
        self.assertIn("applied_rock_mpm_reference_point", source)
        self.assertIn('array("reference_point_world_m"', source)
        self.assertIn("bucket_rigid_by_rock", source)
        self.assertIn("bucket_rock_impulse_by_rock", source)
        self.assertIn("bucket_rock_angular_impulse_by_rock", source)
        self.assertIn("contact_rock_index", source)
        self.assertIn(READY_SCHEMA, source)
        self.assertIn(STATE_SCHEMA, source)
        self.assertTrue(manifest["per_rock_applied_mpm_reference_point"])
        self.assertTrue(manifest["per_rock_bucket_rigid_impulse"])

    def test_bridge_preserves_exact_two_rocks_and_three_named_wrenches(self):
        source, manifest = render_bridge(CONFIG, ROOT)
        self.assertIn("rock_count != 2", source)
        self.assertIn('"maximum_external_wrench_count\\\":3"', source)
        self.assertIn('body_name == "r1-rock-000"', source)
        self.assertIn('body_name == "r1-rock-001"', source)
        self.assertEqual(manifest["required_rock_count"], 2)
        self.assertFalse(manifest["training_qualified"])


if __name__ == "__main__":
    unittest.main()
