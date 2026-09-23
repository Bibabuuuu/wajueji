from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))
import layered_mound_checkpoint_runtime as runtime
import layered_47_bucket_integral_candidate as candidate

SOURCE = '''    std::array<double, 4> unloaded_feedforward{};
      array("unloaded_feedforward", unloaded_feedforward.data(), 4);
      if (operation == "init_default" || operation == "init") {
        if (ready) throw std::runtime_error("init once only");
        if (operation == "init") {
          for (int index = 0; index < model->nq; ++index) data->qpos[index] = number();
          for (int index = 0; index < model->nv; ++index) data->qvel[index] = number();
        }
        std::string extra;
        ConfigureCandidate(model.get(), controller, 1, 1);
        const auto initial = feedforward.Evaluate(data->qpos, controller, {}, {});
        const double before = data->time;
        mj_step(model.get(), data.get());
'''


class BucketIntegralManualCandidateTest(unittest.TestCase):
    def test_frozen_d120_header_is_preserved(self):
        self.assertEqual(hashlib.sha256(candidate.HEADER.read_bytes()).hexdigest(),
                         candidate.HEADER_SHA256)

    def test_disabled_context_does_not_touch_renderer_or_files(self):
        original = runtime.render_capacity_bridge
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "disabled"
            with candidate.bucket_integral_bridge(False, target) as policy:
                self.assertFalse(policy["enabled"])
                self.assertIs(runtime.render_capacity_bridge, original)
            self.assertFalse(target.exists())

    def test_exception_restores_renderer_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runtime, "render_capacity_bridge",
                              return_value=(SOURCE, {"existing": "unchanged"})) as base:
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    with candidate.bucket_integral_bridge(True, Path(tmp) / "build") as policy:
                        code, manifest = runtime.render_capacity_bridge(None)
                        self.assertEqual(manifest["existing"], "unchanged")
                        self.assertEqual(manifest["bucket_integral_candidate"], policy)
                        self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),
                                         policy["generated_bridge_sha256"])
                        self.assertFalse(policy["stopped_qualified"])
                        self.assertTrue(policy["loaded_state_restore"])
                        self.assertIn('operation == "init_loaded"', code)
                        self.assertIn(
                            "bucket_integrator.effort = restored_bucket_integral_effort",
                            code)
                        self.assertIn(
                            "controller.reference[axis].Reset(restored_controller_target[axis])",
                            code)
                        raise RuntimeError("deliberate")
                self.assertIs(runtime.render_capacity_bridge, base)

    def test_duplicate_or_missing_anchors_fail_closed(self):
        for source in (SOURCE + SOURCE, SOURCE.replace("unloaded_feedforward{}", "other{}")):
            with self.assertRaisesRegex(ValueError, "candidate_anchor"):
                candidate.instrument_bucket_integral(source, candidate.HEADER)

    def test_unvalidated_header_does_not_install_renderer(self):
        original = runtime.render_capacity_bridge
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "changed.h"
            header.write_text("unvalidated")
            with patch.object(candidate, "HEADER", header):
                with self.assertRaisesRegex(ValueError, "unvalidated_bucket_integral_header"):
                    with candidate.bucket_integral_bridge(True, Path(tmp) / "build"):
                        self.fail("unvalidated header entered context")
            self.assertIs(runtime.render_capacity_bridge, original)


if __name__ == "__main__":
    unittest.main()
