import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from instrument_r2_bridge_contacts import (  # noqa: E402
    instrument_per_rock_contacts,
)
from mixed_material_r2_mechanism import sha256  # noqa: E402
from run_mixed_material_r2_effective_load import (  # noqa: E402
    _in_relaxed_neighborhood,
    _validate_config,
)


CONFIG = ROOT / "configs/v4/mixed-material-r2-effective-load-v1.json"
REPORT_ROOT = (
    ROOT / "reports/v4/mixed-material-r2-effective-load-2026-09-20-v1")
FROZEN_BRIDGE = (
    ROOT / "reports/v4/mixed-material-r2-stabilization-2026-09-20-v2/"
    "generated-bridge-r2.cc")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class EffectiveLoadConfigTests(unittest.TestCase):
    def test_preliminary_fill_rate_and_scope_are_explicit(self):
        config = read_json(CONFIG)
        _validate_config(config)
        self.assertAlmostEqual(config["bucket_capacity"]["value_m3"],
                               0.255164729950989, places=15)
        self.assertEqual(
            config["bucket_capacity"]["metric"],
            "stable_carried_rock_solid_volume_divided_by_bucket_capacity")
        self.assertTrue(config["scope"]["solid_volume_fill_rate_only"])
        self.assertFalse(config["scope"]["void_compensated_bulk_fill_rate"])
        self.assertFalse(config["scope"]["dump"])
        self.assertFalse(config["scope"]["physics_qualified"])

    def test_rejects_capacity_and_scope_drift(self):
        cases = []
        capacity = read_json(CONFIG)
        capacity["bucket_capacity"]["value_m3"] = 0.2777620975255621
        cases.append(capacity)
        bulk = read_json(CONFIG)
        bulk["scope"]["void_compensated_bulk_fill_rate"] = True
        cases.append(bulk)
        strict = read_json(CONFIG)
        strict["acceptance"]["strict_full_envelope_is_diagnostic_only"] = False
        cases.append(strict)
        truck = read_json(CONFIG)
        truck["scope"]["truck_cad"] = True
        cases.append(truck)
        boolean = read_json(CONFIG)
        boolean["acceptance"]["minimum_carried_rock_count"] = True
        cases.append(boolean)
        for candidate in cases:
            with self.assertRaises(ValueError):
                _validate_config(candidate)


class PerRockInstrumentationTests(unittest.TestCase):
    def test_instrumentation_is_deterministic_and_frozen_source_unchanged(self):
        before_hash = sha256(FROZEN_BRIDGE)
        source = FROZEN_BRIDGE.read_text(encoding="utf-8")
        first = instrument_per_rock_contacts(source)
        second = instrument_per_rock_contacts(source)
        self.assertEqual(first, second)
        self.assertIn("rock_ground_contact_substeps_by_rock", first)
        self.assertIn("rock_bucket_contact_substeps_by_rock", first)
        self.assertEqual(sha256(FROZEN_BRIDGE), before_hash)

    def test_rejects_already_instrumented_or_wrong_source(self):
        source = FROZEN_BRIDGE.read_text(encoding="utf-8")
        instrumented = instrument_per_rock_contacts(source)
        with self.assertRaises(ValueError):
            instrument_per_rock_contacts(instrumented)
        with self.assertRaises(ValueError):
            instrument_per_rock_contacts("not a bridge")


class RelaxedNeighborhoodTests(unittest.TestCase):
    def test_center_may_be_above_strict_envelope_but_not_far_away(self):
        inventory = read_json(
            "/home/qunwang/codex-work/compact-bank-2026-09-11/"
            "inventory-contract.json")
        near = {"center_bucket_m": [0.4, 0.0, 0.05]}
        far = {"center_bucket_m": [3.0, 0.0, 3.0]}
        self.assertTrue(_in_relaxed_neighborhood(near, inventory, 0.3))
        self.assertFalse(_in_relaxed_neighborhood(far, inventory, 0.3))


@unittest.skipUnless(REPORT_ROOT.is_dir(), "formal effective-load report pending")
class EffectiveLoadFormalReportTests(unittest.TestCase):
    def test_formal_report_is_bound_and_scoped(self):
        config = read_json(CONFIG)
        report = read_json(REPORT_ROOT / "report.json")
        self.assertEqual(report["scene_count"], 2)
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(report["bucket_capacity_m3"],
                         config["bucket_capacity"]["value_m3"])
        self.assertFalse(report["void_compensated_bulk_fill_rate_qualified"])
        self.assertFalse(report["dump_qualified"])
        self.assertFalse(report["truck_cad_qualified"])
        self.assertFalse(report["physics_qualified"])
        for scene in report["scenes"]:
            self.assertEqual(scene["carried_rock_count"], 2)
            expected = (scene["carried_solid_volume_m3"] /
                        report["bucket_capacity_m3"])
            self.assertAlmostEqual(
                scene["preliminary_solid_volume_fill_rate"], expected,
                places=15)

    def test_formal_report_counts_relaxed_carried_rocks_not_grounded_envelope(self):
        report = read_json(REPORT_ROOT / "report.json")
        saw_carried_outside_strict_envelope = False
        saw_grounded_strict_envelope_rejected = False
        for scene in report["scenes"]:
            scene_report = read_json(
                REPORT_ROOT / scene["scene_id"] / "report.json")
            for rock in scene_report["metrics"]["per_rock"]:
                if (rock["carried"]
                        and not rock[
                            "final_full_envelope_inside_diagnostic"]):
                    saw_carried_outside_strict_envelope = True
                if (not rock["carried"]
                        and rock[
                            "final_full_envelope_inside_diagnostic"]
                        and not rock["checks"]["continuously_off_ground"]):
                    saw_grounded_strict_envelope_rejected = True
        self.assertTrue(saw_carried_outside_strict_envelope)
        self.assertTrue(saw_grounded_strict_envelope_rejected)


if __name__ == "__main__":
    unittest.main()
