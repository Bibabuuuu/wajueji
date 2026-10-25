import unittest
import numpy as np
from contact_bc_ground_guard import BucketGroundGuard


class Assembly:
    def assemble(self,q,include_meshes=True):
        transform=np.eye(4);transform[2,3]=q[0]
        return [dict(body='bucket',body_local_vertices=[[0.,0.,-1.],[1.,0.,0.]])],dict(bucket=transform)


class GroundGuardTest(unittest.TestCase):
    def test_real_lowest_vertex_not_bucket_origin(self):
        guard=BucketGroundGuard(Assembly(),[2.])
        self.assertTrue(guard.measure([1.02])['safe'])
        self.assertFalse(guard.measure([1.004])['safe'])
        self.assertFalse(guard.measure([.9])['safe'])
        self.assertAlmostEqual(guard.measure([.9])['minimum_bucket_z_m'],-.1)

    def test_ground_offset_and_nonfinite_fail_closed(self):
        guard=BucketGroundGuard(Assembly(),[2.],ground_z_m=.2)
        self.assertFalse(guard.measure([1.1])['safe'])
        with self.assertRaisesRegex(ValueError,'nonfinite'):guard.measure([np.nan])
        with self.assertRaisesRegex(ValueError,'parameters'):BucketGroundGuard(Assembly(),[2.],margin_m=-.01)

    def test_missing_bucket_rejected(self):
        class Empty:
            def assemble(self,q):return [],{}
        with self.assertRaisesRegex(ValueError,'missing_geometry'):BucketGroundGuard(Empty(),[2.])


if __name__=='__main__':unittest.main()
