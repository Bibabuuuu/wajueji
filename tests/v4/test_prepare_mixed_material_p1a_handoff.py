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

from prepare_mixed_material_p1a_handoff import (  # noqa: E402
    build_receipt,
    load_config,
    sha256,
)


CONFIG = ROOT / "configs" / "v4" / "mixed-material-p1a-handoff-v1.json"


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_fixture(root, *, fullscale_pass=True):
    scene = root / "scene-manifest.json"
    write_json(scene, {"schema": "fixture_scene", "rock_count": 22})
    scaled = root / "scaled-audit.json"
    write_json(scaled, {
        "schema": "excavation_layered_mixed_scene_staged_settling_audit_report_v1",
        "status": "admitted_for_fullscale_scaleup",
        "derived_qualifications": {
            "staged_construction_qualified": True,
            "nominal_hold_qualified": True,
            "fullscale_scaleup_authorized": True,
            "p1a_single_scoop_authorized": False,
            "fullscale_scene_qualified": False,
            "three_cycle_qualified": False,
        },
    })
    bundle = root / "geometry-bundle-report.json"
    write_json(bundle, {
        "schema": "excavation_formal_scene_geometry_bundle_report_v1",
        "status": "formal_scene_geometry_bundle_candidate",
        "scene_manifest_sha256": sha256(scene),
        "checks": {"single_scene_manifest_bound": True,
                   "model_and_fill_rock_count_match": True},
        "time_zero_executed": False,
        "physics_step_executed": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    })
    time_zero = root / "time-zero.json"
    settling = root / "settling.json"
    write_json(time_zero, {"schema": "fixture_time_zero"})
    write_json(settling, {"schema": "fixture_settling"})
    required_checks = [
        "stable_scene_manifest_valid", "geometry_bundle_cross_bound",
        "actual_n_time_zero_passed", "fullscale_staged_settling_passed",
        "nominal_hold_passed", "visual_review_passed",
        "mpm_connectivity_passed", "identity_and_mass_closed",
    ]
    checks = {name: True for name in required_checks}
    if not fullscale_pass:
        checks["nominal_hold_passed"] = False
    candidate = root / "fullscale-candidate.json"
    write_json(candidate, {"fixture": "candidate-producer-report"})
    fullscale = root / "fullscale-report.json"
    write_json(fullscale, {
        "schema": "excavation_fullscale_layered_mixed_scene_qualification_audit_report_v1",
        "status": ("fullscale_scene_admitted_for_p1a" if fullscale_pass
                   else "fullscale_scene_failure_preserved"),
        "candidate_report": {
            "path": str(candidate.resolve()),
            "sha256": sha256(candidate),
        },
        "inputs": {
            "scaled_audit": {"path": str(scaled.resolve()),
                             "sha256": sha256(scaled)},
            "scene_manifest": {"path": str(scene.resolve()),
                               "sha256": sha256(scene)},
            "geometry_bundle_report": {"path": str(bundle.resolve()),
                                       "sha256": sha256(bundle)},
            "time_zero_report": {"path": str(time_zero.resolve()),
                                 "sha256": sha256(time_zero)},
            "settling_report": {"path": str(settling.resolve()),
                                "sha256": sha256(settling)},
        },
        "checks": {
            "input_bindings_exact": True,
            "scaled_scaleup_gate_admitted": True,
            "scene_manifest_well_formed": True,
            "geometry_bundle_well_formed": True,
            "time_zero_report_well_formed": True,
            "settling_report_well_formed": True,
            "candidate_declaration_consistent": True,
        },
        "derived_checks": checks,
        "derived_qualifications": {
            "scaled_method_qualified": True,
            "fullscale_scene_qualified": fullscale_pass,
            "p1a_single_scoop_authorized": fullscale_pass,
            "three_cycle_qualified": False,
        },
        "execution": {
            "physics_executed_by_auditor": False,
            "gpu_run_by_auditor": False,
        },
    })
    return scaled, fullscale, bundle


class MixedMaterialP1aHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG, ROOT)

    def test_config_binds_existing_adapters_without_execution(self):
        self.assertTrue(self.config["scope"]["offline_contract_only"])
        self.assertFalse(self.config["scope"]["physics_execution"])
        self.assertEqual(
            self.config["adapter_chain"][-1],
            "existing_same_instance_three_cycle_loop")

    def test_fullscale_pass_builds_offline_handoff_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            scaled, fullscale, bundle = make_fixture(Path(temporary))
            receipt = build_receipt(
                self.config, scaled, fullscale, bundle)
        self.assertEqual(receipt["status"], "p1a_offline_handoff_ready")
        self.assertTrue(
            receipt["qualifications"]["p1a_single_scoop_authorized"])
        self.assertFalse(receipt["qualifications"]["three_cycle_qualified"])
        self.assertFalse(receipt["execution"]["physics_executed"])

    def test_honest_fullscale_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            scaled, fullscale, bundle = make_fixture(
                Path(temporary), fullscale_pass=False)
            receipt = build_receipt(
                self.config, scaled, fullscale, bundle)
        self.assertEqual(receipt["status"], "fullscale_failure_preserved")
        self.assertFalse(
            receipt["qualifications"]["p1a_single_scoop_authorized"])

    def test_scaled_pass_cannot_directly_or_falsely_authorize_p1a(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scaled, fullscale, bundle = make_fixture(root)
            value = json.loads(scaled.read_text(encoding="utf-8"))
            value["derived_qualifications"]["p1a_single_scoop_authorized"] = True
            write_json(scaled, value)
            fullscale_value = json.loads(fullscale.read_text(encoding="utf-8"))
            fullscale_value["inputs"]["scaled_audit"]["sha256"] = sha256(scaled)
            write_json(fullscale, fullscale_value)
            with self.assertRaisesRegex(ValueError, "scaled_audit_not_admitted"):
                build_receipt(self.config, scaled, fullscale, bundle)

    def test_hash_and_declaration_attacks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scaled, fullscale, bundle = make_fixture(root)
            value = json.loads(fullscale.read_text(encoding="utf-8"))
            value["inputs"]["scene_manifest"]["sha256"] = "0" * 64
            write_json(fullscale, value)
            with self.assertRaisesRegex(ValueError, "binding_mismatch"):
                build_receipt(self.config, scaled, fullscale, bundle)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scaled, fullscale, bundle = make_fixture(root)
            write_json(
                root / "fullscale-candidate.json",
                {"fixture": "mutated-after-independent-audit"})
            with self.assertRaisesRegex(ValueError, "candidate_binding"):
                build_receipt(self.config, scaled, fullscale, bundle)

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["scope"]["physics_execution"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-config.json"
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "scope_expanded"):
                load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
