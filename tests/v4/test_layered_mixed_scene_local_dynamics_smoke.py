import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "v4" / "layered-mixed-scene-local-dynamics-smoke-v1.json"
REPORT = (ROOT / "reports" / "v4" /
          "layered-mixed-scene-local-dynamics-smoke-2026-09-20-v1" / "report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayeredMixedSceneLocalDynamicsSmokeTests(unittest.TestCase):
    def test_config_binds_geometry_and_keeps_smoke_scope_honest(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema"],
                         "excavation_layered_mixed_scene_local_dynamics_smoke_config_v1")
        for name in ("config", "manifest", "bank"):
            path = ROOT / config["geometry"][name + "_path"]
            self.assertEqual(sha256(path), config["geometry"][name + "_sha256"])
        capacity = ROOT / config["capacity_bridge"]["config_path"]
        self.assertEqual(sha256(capacity),
                         config["capacity_bridge"]["config_sha256"])
        self.assertTrue(config["scope"]["short_smoke_only"])
        self.assertFalse(config["scope"]["staged_activation"])
        self.assertFalse(config["scope"]["settling_qualified"])
        self.assertFalse(config["scope"]["wall_free_hold_qualified"])

    def test_persisted_smoke_has_contact_without_stability_overclaim(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["status"],
                         "final_assembled_short_mixed_dynamics_smoke_passed")
        self.assertTrue(report["short_smoke_qualified"])
        self.assertFalse(report["settling_qualified"])
        self.assertFalse(report["wall_free_hold_qualified"])
        self.assertFalse(report["formal_scene_qualified"])
        self.assertTrue(all(report["checks"].values()))
        metrics = report["metrics"]
        self.assertEqual(metrics["frames"], 12)
        self.assertEqual(metrics["matrix"]["particle_count"], 72415)
        self.assertGreater(metrics["mpm_rock_contact_samples"], 0)
        self.assertEqual(metrics["maximum_forbidden_contact_count"], 0)
        self.assertEqual(report["inputs"]["config_sha256"], sha256(CONFIG))
        self.assertEqual(report["inputs"]["model_manifest"]["rock_count"], 22)
        self.assertFalse(report["inputs"]["model_manifest"]["stable_scene_claimed"])


if __name__ == "__main__":
    unittest.main()
