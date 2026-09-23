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

from audit_fullscale_layered_mixed_scene_qualification import (  # noqa: E402
    audit_candidate,
    load_config,
    sha256,
)


CONFIG = (ROOT / "configs" / "v4" /
          "fullscale-layered-mixed-scene-qualification-audit-v1.json")


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binding(path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def make_candidate(root, config, *, pass_settling=True):
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
    scene_contract = config["artifact_contracts"]["scene_manifest"]
    scene = root / "scene.json"
    write_json(scene, {
        "schema": scene_contract["schema"],
        "status": scene_contract["status"],
        "rock_count": 2,
        "rocks": [{"rock_id": "rock-000"}, {"rock_id": "rock-001"}],
        "checks": {name: True for name in
                   scene_contract["required_true_checks"]},
    })
    bundle_contract = config["artifact_contracts"]["geometry_bundle_report"]
    bundle = root / "geometry-bundle.json"
    write_json(bundle, {
        "schema": bundle_contract["schema"],
        "status": bundle_contract["status"],
        "scene_manifest_sha256": sha256(scene),
        "rock_count": 2,
        "expected_nq": 14,
        "expected_nv": 12,
        "checks": {name: True for name in
                   bundle_contract["required_true_checks"]},
        "time_zero_executed": False,
        "physics_step_executed": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    })
    time_zero_contract = config["artifact_contracts"]["time_zero_report"]
    time_zero = root / "time-zero.json"
    write_json(time_zero, {
        "schema": time_zero_contract["schema"],
        "status": time_zero_contract["passing_status"],
        "inputs": {
            "scene_manifest": binding(scene),
            "geometry_bundle_report": binding(bundle),
        },
        "rock_count": 2,
        "expected_nq": 14,
        "expected_nv": 12,
        "checks": {name: True for name in
                   time_zero_contract["required_true_checks"]},
    })
    settling_contract = config["artifact_contracts"]["settling_report"]
    settling_checks = {name: True for name in
                       settling_contract["required_true_checks"]}
    if not pass_settling:
        settling_checks["nominal_hold_passed"] = False
    settling = root / "settling.json"
    write_json(settling, {
        "schema": settling_contract["schema"],
        "status": (settling_contract["passing_status"] if pass_settling else
                   settling_contract["failure_status"]),
        "inputs": {
            "scene_manifest": binding(scene),
            "geometry_bundle_report": binding(bundle),
            "time_zero_report": binding(time_zero),
        },
        "rock_count": 2,
        "expected_nq": 14,
        "expected_nv": 12,
        "checks": settling_checks,
    })
    paths = {
        "scaled_audit": scaled,
        "scene_manifest": scene,
        "geometry_bundle_report": bundle,
        "time_zero_report": time_zero,
        "settling_report": settling,
    }
    inputs = {name: binding(path) for name, path in paths.items()}
    derived_checks = {
        "stable_scene_manifest_valid": True,
        "geometry_bundle_cross_bound": True,
        "actual_n_time_zero_passed": True,
        "fullscale_staged_settling_passed": True,
        "nominal_hold_passed": pass_settling,
        "visual_review_passed": True,
        "mpm_connectivity_passed": True,
        "identity_and_mass_closed": True,
    }
    candidate = {
        "schema": config["candidate_report_schema"],
        "status": ("fullscale_scene_qualified_for_audit" if pass_settling else
                   "fullscale_scene_failed"),
        "inputs": inputs,
        "checks": derived_checks,
        "qualifications": {
            "fullscale_scene_qualified": pass_settling,
            "p1a_single_scoop_authorized": pass_settling,
            "three_cycle_qualified": False,
        },
    }
    return candidate, paths


def run_audit(root, config, candidate):
    candidate_path = root / "candidate.json"
    write_json(candidate_path, candidate)
    return audit_candidate(config, candidate, binding(candidate_path))


class FullscaleQualificationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG, ROOT)

    def test_config_is_read_only_and_binds_upstream_contracts(self):
        self.assertTrue(
            self.config["scope"]["independent_report_audit_only"])
        self.assertFalse(self.config["scope"]["physics_execution"])
        self.assertFalse(self.config["scope"]["gpu_run"])

    def test_pass_is_independently_admitted_for_p1a(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = make_candidate(root, self.config)
            report = run_audit(root, self.config, candidate)
        self.assertEqual(report["status"], "fullscale_scene_admitted_for_p1a")
        self.assertTrue(all(report["checks"].values()))
        self.assertTrue(all(report["derived_checks"].values()))
        self.assertTrue(
            report["derived_qualifications"]["p1a_single_scoop_authorized"])
        self.assertFalse(
            report["derived_qualifications"]["three_cycle_qualified"])

    def test_honest_physics_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = make_candidate(
                root, self.config, pass_settling=False)
            report = run_audit(root, self.config, candidate)
        self.assertEqual(
            report["status"], "fullscale_scene_failure_preserved")
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(report["derived_checks"]["nominal_hold_passed"])
        self.assertFalse(
            report["derived_qualifications"]["p1a_single_scoop_authorized"])

    def test_false_pass_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = make_candidate(
                root, self.config, pass_settling=False)
            candidate["status"] = "fullscale_scene_qualified_for_audit"
            candidate["checks"] = {
                name: True for name in self.config["required_candidate_checks"]}
            candidate["qualifications"]["fullscale_scene_qualified"] = True
            candidate["qualifications"]["p1a_single_scoop_authorized"] = True
            report = run_audit(root, self.config, candidate)
        self.assertEqual(report["status"], "candidate_report_rejected")
        self.assertFalse(report["checks"]["candidate_declaration_consistent"])

    def test_hash_rebinding_and_execution_scope_expansion_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, paths = make_candidate(root, self.config)
            scene_value = json.loads(
                paths["scene_manifest"].read_text(encoding="utf-8"))
            scene_value["extra"] = "mutated-after-binding"
            write_json(paths["scene_manifest"], scene_value)
            report = run_audit(root, self.config, candidate)
            self.assertEqual(report["status"], "candidate_report_rejected")
            self.assertFalse(report["checks"]["input_bindings_exact"])

            bad_config = copy.deepcopy(self.config)
            bad_config["scope"]["gpu_run"] = True
            path = root / "bad-config.json"
            write_json(path, bad_config)
            with self.assertRaisesRegex(ValueError, "scope_expanded"):
                load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
