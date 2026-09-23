from types import SimpleNamespace
import unittest
import numpy as np
import newton
import warp as wp
from newton.solvers import SolverImplicitMPM
from moving_bucket_sweep_candidate import MovingBucketSweep


def particle_state(q, v):
    return SimpleNamespace(
        particle_q=wp.array(q, dtype=wp.vec3, device="cpu"),
        particle_qd=wp.array(v, dtype=wp.vec3, device="cpu"),
        mpm=SimpleNamespace(particle_qd_grad=wp.zeros(len(q), dtype=wp.mat33, device="cpu")))


class MovingSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        builder = newton.ModelBuilder()
        builder.add_shape_box(-1, hx=.5, hy=.5, hz=.018)
        model = builder.finalize(device="cpu")
        sand = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand)
        sand.add_particle(pos=(0, 0, 1), vel=(0, 0, 0), mass=1, radius=.01)
        cls.solver = SolverImplicitMPM(sand.finalize(device="cpu"), config=SolverImplicitMPM.Config())
        cls.solver.setup_collider(model=model)
        cls.mesh = cls.solver._mpm_model._collider_meshes[0]

    def correct(self, start, target, velocity, old=None, new=None, linear=(0,0,0), angular=(0,0,0)):
        state = particle_state([target], [velocity])
        guard = MovingBucketSweep(self.mesh, state, 2.0, 0.45)
        info = guard.apply(state, wp.array([start], dtype=wp.vec3, device="cpu"),
                           old or wp.transform_identity(), new or wp.transform_identity(), linear, angular)
        return state.particle_q.numpy()[0], state.particle_qd.numpy()[0], info

    def test_inside_to_outside_wall_and_impulse_balance(self):
        q,v,info = self.correct((0,0,-.05),(0,0,.05),(0,0,1))
        self.assertLess(q[2], -.018)
        np.testing.assert_allclose(v, 0, atol=1e-7)
        np.testing.assert_allclose(np.asarray(info['reaction_impulse_ns']) + 2*(v-[0,0,1]), 0, atol=1e-6)

    def test_shallow_initial_penetration_uses_old_side(self):
        q,v,info = self.correct((0,0,-.005),(0,0,.05),(0,0,1))
        self.assertLess(q[2], -.018)
        self.assertEqual(info['initial_overlap_recoveries'], 1)

    def test_exterior_particle_cannot_enter_through_wall(self):
        q,v,info = self.correct((0,0,.05),(0,0,-.05),(0,0,-1))
        self.assertGreater(q[2], .018)

    def test_miss_is_bitwise_unchanged(self):
        q,v,info = self.correct((2,0,-.05),(2,0,.05),(.13,.27,1))
        np.testing.assert_array_equal(q, np.array([2,0,.05],np.float32))
        np.testing.assert_array_equal(v, np.array([.13,.27,1],np.float32))
        self.assertEqual(info['corrected_particles'], 0)

    def test_translating_wall_pushes_stationary_particle(self):
        pose=wp.transform(wp.vec3(0,0,.04),wp.quat_identity())
        q,v,info=self.correct((0,0,.05),(0,0,.05),(0,0,0),new=pose,linear=(0,0,2.4))
        self.assertGreater(q[2], .058)
        self.assertAlmostEqual(float(v[2]),2.4,places=5)

    def test_rotating_frame_preserves_entering_face(self):
        old=wp.transform(wp.vec3(1,2,3),wp.quat_from_axis_angle(wp.vec3(0,1,0),.3))
        new=wp.transform(wp.vec3(1,2,3),wp.quat_from_axis_angle(wp.vec3(0,1,0),.31))
        start=wp.transform_point(old,wp.vec3(0,0,-.05))
        target=wp.transform_point(new,wp.vec3(0,0,.05))
        q,v,info=self.correct(start,target,(0,0,1),old=old,new=new,angular=(0,.6,0))
        local=wp.transform_point(wp.transform_inverse(new),wp.vec3(*q))
        self.assertLess(local[2],-.01799)


if __name__ == '__main__':
    unittest.main()
