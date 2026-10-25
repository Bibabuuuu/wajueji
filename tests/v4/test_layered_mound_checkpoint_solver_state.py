import json
from pathlib import Path
import unittest

import newton
import numpy as np
import warp as wp

import layered_mound_checkpoint as checkpoint
from run_layered_mound_three_rock_adaptive_press import MultiRockFixture


ROOT = Path(__file__).resolve().parents[2]


class LayeredMoundCheckpointSolverStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_fresh_fixture_with_solver_state_matches_uninterrupted_next_step(self):
        config = json.loads((
            ROOT / "configs/v4/layered-mound-thirtythree-rock-three-layer-v1.json"
        ).read_text(encoding="utf-8"))
        manifest = json.loads((
            ROOT / "results/layered-mound-physical-preflight-v1/scene-manifest.json"
        ).read_text(encoding="utf-8"))
        record = manifest["rocks"][0]
        spacing = float(config["mpm"]["particle_spacing_m"])
        initial_q = np.asarray([
            (x * spacing, y * spacing, 0.016 + z * spacing)
            for z in range(2) for y in range(3) for x in range(3)
        ], dtype=np.float32)
        dt = float(config["probe"]["output_dt_s"])
        origins = [np.asarray(record["position_world_m"], dtype=np.float64)]

        source = MultiRockFixture(config, ROOT, [record], initial_q, "cpu")
        for _ in range(4):
            source.step(dt, origins)
        physical = {
            "particle_q": source.state.particle_q.numpy().copy(),
            "particle_qd": source.state.particle_qd.numpy().copy(),
            "particle_f": source.state.particle_f.numpy().copy(),
            "particle_flags": source.model.particle_flags.numpy().copy(),
            **{
                name: getattr(source.state.mpm, name).numpy().copy()
                for name in checkpoint.MPM_STATE_FIELDS
            },
            "body_q": source.state.body_q.numpy().copy(),
            "body_qd": source.state.body_qd.numpy().copy(),
            "body_f": source.state.body_f.numpy().copy(),
        }
        solver = checkpoint._solver_continuation_arrays(source)

        source.step(dt, origins)
        expected = {
            "particle_q": source.state.particle_q.numpy().copy(),
            "particle_qd": source.state.particle_qd.numpy().copy(),
            "particle_stress": source.state.mpm.particle_stress.numpy().copy(),
            "particle_elastic_strain":
                source.state.mpm.particle_elastic_strain.numpy().copy(),
        }

        restored = MultiRockFixture(
            config, ROOT, [record], solver["grid_reference_particle_q"], "cpu")
        restored.state.particle_q.assign(physical["particle_q"])
        restored.state.particle_qd.assign(physical["particle_qd"])
        restored.state.particle_f.assign(physical["particle_f"])
        restored.model.particle_flags.assign(physical["particle_flags"])
        for name in checkpoint.MPM_STATE_FIELDS:
            getattr(restored.state.mpm, name).assign(physical[name])
        restored.state.body_q.assign(physical["body_q"])
        restored.state.body_qd.assign(physical["body_qd"])
        restored.state.body_f.assign(physical["body_f"])
        restored.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        report = checkpoint.restore_solver_continuation(restored, solver)
        restored.step(dt, origins)

        self.assertTrue(all(report["grid_readback_exact"].values()))
        self.assertTrue(all(report["solver_field_readback_exact"].values()))
        actual = {
            "particle_q": restored.state.particle_q.numpy(),
            "particle_qd": restored.state.particle_qd.numpy(),
            "particle_stress": restored.state.mpm.particle_stress.numpy(),
            "particle_elastic_strain":
                restored.state.mpm.particle_elastic_strain.numpy(),
        }
        for name in expected:
            self.assertTrue(np.array_equal(actual[name], expected[name]), name)


if __name__ == "__main__":
    unittest.main()
