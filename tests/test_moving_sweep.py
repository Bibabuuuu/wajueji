import unittest
from types import SimpleNamespace
import numpy as np
try:
    import warp as wp
    from terrain.moving_sweep import MovingSweep
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None
    MovingSweep = None


@unittest.skipUnless(wp is not None, "Warp runtime required")
class MovingSweepTests(unittest.TestCase):
    def state(self,q,v):
        return SimpleNamespace(particle_q=wp.array(q,dtype=wp.vec3,device='cpu'),particle_qd=wp.array(v,dtype=wp.vec3,device='cpu'),
             mpm=SimpleNamespace(particle_qd_grad=wp.zeros(len(q),dtype=wp.mat33,device='cpu')))

    def test_translating_wall_pushes_stationary_particle(self):
        # Plane normal points +X, translating toward +X.
        s=self.state([[.05,0,0]],[[0,0,0]])
        sweep=MovingSweep([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]],[[0,1,2],[0,2,3]],s)
        sweep.begin(s)
        hits=sweep.finish(s,wp.transform_identity(),wp.transform([.1,0,0],wp.quat_identity()),[1,0,0],[0,0,0])
        self.assertEqual(hits,1);self.assertGreaterEqual(s.particle_q.numpy()[0,0],.1)
        np.testing.assert_allclose(s.particle_qd.numpy(),[[1,0,0]],atol=1e-6)

    def test_static_surface_sliding_and_untouched_state(self):
        s=self.state([[.1,0,0],[.2,0,0]],[[-2,1,0],[0,1,0]])
        sweep=MovingSweep([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]],[[0,1,2],[0,2,3]],s);sweep.begin(s)
        s.particle_q.assign(np.array([[-.1,.1,0],[.2,.1,0]],np.float32));saved=s.particle_q.numpy()[1].copy()
        self.assertEqual(sweep.finish(s,wp.transform_identity(),wp.transform_identity(),[0,0,0],[0,0,0],mu=0.),1)
        self.assertGreater(s.particle_q.numpy()[0,0],0);np.testing.assert_allclose(s.particle_qd.numpy()[0],[0,1,0])
        np.testing.assert_array_equal(s.particle_q.numpy()[1],saved)

    def test_rotating_surface_relative_normal_velocity(self):
        s=self.state([[.05,.8,0]],[[0,0,0]])
        sweep=MovingSweep([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]],[[0,1,2],[0,2,3]],s);sweep.begin(s)
        pose=wp.transform([0,0,0],wp.quat_from_axis_angle(wp.vec3(0,0,1),-.1))
        self.assertEqual(sweep.finish(s,wp.transform_identity(),pose,[0,0,0],[0,0,-1],mu=0),1)
        q=s.particle_q.numpy()[0];v=s.particle_qd.numpy()[0]
        normal=np.array([np.cos(.1),-np.sin(.1),0.])
        self.assertGreater(q@normal,0.)
        self.assertGreaterEqual((v-np.cross([0,0,-1],q))@normal,-1e-6)
