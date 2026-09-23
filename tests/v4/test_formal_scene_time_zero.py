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

from run_formal_scene_time_zero import (  # noqa: E402
    build_report,
    load_config,
    sha256,
    validate_bundle,
)


CONFIG = ROOT / "configs" / "v4" / "formal-scene-time-zero-v1.json"
AUDIT_CONFIG = (ROOT / "configs" / "v4" /
                "fullscale-layered-mixed-scene-qualification-audit-v1.json")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(root):
    scene = root / "scene.json"
    write_json(scene, {"rock_count": 2, "rocks": [
        {"rock_id": "source-a"}, {"rock_id": "source-b"}]})
    directory = root / "bundle"
    model = directory / "actual-n-scene.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco model='fixture'/>\n", encoding="utf-8")
    manifest_path = directory / "actual-n-model-manifest.json"
    manifest = {
        "schema": "excavation_formal_scene_model_adapter_manifest_v1",
        "status": "formal_scene_actual_n_model_generated",
        "source_scene_manifest_sha256": sha256(scene),
        "candidate_model_sha256": sha256(model),
        "rock_count": 2,
        "expected_nq": 33,
        "expected_nv": 30,
        "time_zero_executed": False,
        "physics_step_executed": False,
    }
    write_json(manifest_path, manifest)
    files = {
        "actual-n-scene.xml": model,
        "actual-n-model-manifest.json": manifest_path,
        "mpm-fill/layered-bank.npz": directory / "mpm-fill/layered-bank.npz",
        "mpm-fill/metrics.json": directory / "mpm-fill/metrics.json",
        "mpm-fill/report.json": directory / "mpm-fill/report.json",
    }
    for relative, path in files.items():
        if relative.endswith(".npz"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-npz")
        elif not path.exists():
            write_json(path, {"fixture": relative})
    report = {
        "schema": "excavation_formal_scene_geometry_bundle_report_v1",
        "status": "formal_scene_geometry_bundle_candidate",
        "scene_manifest_sha256": sha256(scene),
        "rock_count": 2,
        "expected_nq": 33,
        "expected_nv": 30,
        "outputs": {relative: sha256(path) for relative, path in files.items()},
        "checks": {
            "single_scene_manifest_bound": True,
            "model_and_fill_rock_count_match": True,
            "mpm_particle_partition": True,
            "retained_particle_center_inside_rock_zero": True,
        },
        "time_zero_executed": False,
        "physics_step_executed": False,
        "formal_scene_qualified": False,
        "physics_qualified": False,
    }
    write_json(directory / "geometry-bundle-report.json", report)
    return scene, directory


def ready_and_state():
    ids = ["r1-rock-000", "r1-rock-001"]
    ready = {
        "schema": "excavation_mixed_coupled_bridge_capacity_v1",
        "rock_count": 2,
        "maximum_external_wrench_count": 3,
        "allowed_external_wrench_bodies": ["bucket", *ids],
    }
    state = {
        "time_s": 0,
        "mechanism_qpos": [0.0] * 19,
        "mechanism_qvel": [0.0] * 18,
        "rocks": [{"rock_id": rock_id, "qpos": [0.0] * 7,
                   "qvel": [0.0] * 6} for rock_id in ids],
        "applied_bucket_mpm_force_world_n": [0, 0, 0],
        "applied_bucket_mpm_torque_world_nm": [0, 0, 0],
        "applied_rocks_mpm": [{
            "rock_id": rock_id, "force_world_n": [0, 0, 0],
            "torque_world_nm": [0, 0, 0]} for rock_id in ids],
        "forbidden_contact_count": 0,
    }
    return ready, state


class FormalSceneTimeZeroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG, ROOT)

    def test_contract_matches_independent_fullscale_auditor(self):
        audit = json.loads(AUDIT_CONFIG.read_text(encoding="utf-8"))
        expected = audit["artifact_contracts"]["time_zero_report"]
        self.assertEqual(self.config["report_contract"]["schema"],
                         expected["schema"])
        self.assertEqual(self.config["report_contract"]["passing_status"],
                         expected["passing_status"])
        self.assertEqual(self.config["report_contract"]["checks"],
                         expected["required_true_checks"])
        self.assertFalse(self.config["scope"]["physics_step"])
        self.assertFalse(self.config["scope"]["mpm_or_gpu_run"])

    def test_valid_bundle_and_time_zero_state_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, directory = make_bundle(Path(temporary))
            bundle = validate_bundle(self.config, scene, directory)
            ready, state = ready_and_state()
            report = build_report(
                self.config, bundle, ready, state, {"fixture": True})
        self.assertEqual(report["status"], "fullscale_time_zero_passed")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["rock_count"], 2)
        self.assertEqual(report["execution"]["mj_step_calls"], 0)
        self.assertFalse(
            report["execution"]["p1a_single_scoop_authorized"])

    def test_nonfinite_or_nonzero_wrench_is_honest_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, directory = make_bundle(Path(temporary))
            bundle = validate_bundle(self.config, scene, directory)
            ready, state = ready_and_state()
            state["rocks"][0]["qvel"][0] = float("nan")
            state["applied_rocks_mpm"][0]["force_world_n"][0] = 1
            report = build_report(
                self.config, bundle, ready, state, {"fixture": True})
        self.assertEqual(report["status"], "fullscale_time_zero_failed")
        self.assertFalse(report["checks"]["finite_state"])
        self.assertFalse(report["checks"]["zero_external_wrenches"])

    def test_bundle_hash_and_scope_attacks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene, directory = make_bundle(root)
            (directory / "actual-n-scene.xml").write_text(
                "<mujoco model='mutated'/>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "geometry_bundle_invalid"):
                validate_bundle(self.config, scene, directory)

            config = copy.deepcopy(self.config)
            config["scope"]["physics_step"] = True
            bad = root / "bad-config.json"
            write_json(bad, config)
            with self.assertRaisesRegex(ValueError, "scope_expanded"):
                load_config(bad, ROOT)


if __name__ == "__main__":
    unittest.main()
