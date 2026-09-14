import unittest
import numpy as np
from perception.core import Camera, look_at
try:
    from perception.virtual_depth import VirtualDepth
except ImportError:
    VirtualDepth = None


@unittest.skipIf(VirtualDepth is None, 'Warp runtime required')
class VirtualDepthTests(unittest.TestCase):
    def setUp(self):
        self.camera=Camera(9,9,10,10,4,4,.1,10)

    def test_analytic_sphere_first_surface(self):
        sensor=VirtualDepth(self.camera,radius_m=.5,ground=False)
        depth,_=sensor.capture(np.array([[0,0,3],[0,0,5]],np.float32),np.eye(4))
        self.assertAlmostEqual(float(depth[4,4]),2.5,places=5)
        self.assertTrue(np.isnan(depth[0,0]))
        # Off-axis pixel analytic quadratic, independent of raster bounds.
        ray=np.array([.1,0,1]);c=np.array([0,0,3]);a=ray@ray;b=ray@c
        expected=(b-np.sqrt(b*b-a*(c@c-.25)))/a
        self.assertAlmostEqual(float(depth[4,5]),expected,places=4)

    def test_mesh_occludes_particles_and_robot_prediction(self):
        vertices=np.array([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],np.float32)
        faces=np.array([[0,1,2],[0,2,3]],np.int32)
        sensor=VirtualDepth(self.camera,[(vertices,faces)],robot_count=1,radius_m=.5,ground=False)
        depth,robot=sensor.capture(np.array([[0,0,3]],np.float32),np.eye(4),[np.eye(4)])
        np.testing.assert_allclose(depth,2,atol=1e-5);np.testing.assert_allclose(robot,depth)
        # Material in front of the known robot is visible and is not robot depth.
        depth,robot=sensor.capture(np.array([[0,0,1]],np.float32),np.eye(4),[np.eye(4)])
        self.assertAlmostEqual(float(depth[4,4]),.5,places=5)
        self.assertAlmostEqual(float(robot[4,4]),2.,places=5)

    def test_ground_camera_transform_and_input_readonly(self):
        sensor=VirtualDepth(self.camera)
        particles=np.empty((0,3),np.float32);pose=look_at([0,0,5],[0,0,0]);before=pose.copy()
        depth,_=sensor.capture(particles,pose)
        np.testing.assert_allclose(depth,5,atol=1e-5);np.testing.assert_array_equal(pose,before)

    def test_rotated_mesh_and_camera_extrinsics(self):
        vertices=np.array([[-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],np.float32)
        sensor=VirtualDepth(self.camera,[(vertices,np.array([[0,1,2],[0,2,3]]))],ground=False)
        mesh_pose=np.array([[0,0,1,3],[0,1,0,4],[-1,0,0,5],[0,0,0,1]],float)
        camera_pose=look_at([3,4,5],[4,4,5])
        depth,_=sensor.capture(np.empty((0,3),np.float32),camera_pose,[mesh_pose])
        np.testing.assert_allclose(depth,2,atol=1e-5)


if __name__=='__main__':unittest.main()
