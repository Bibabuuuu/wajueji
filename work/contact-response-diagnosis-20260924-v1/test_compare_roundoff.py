import unittest
from compare_roundoff import compare


class ComparisonTests(unittest.TestCase):
    def test_records_angular_impulse_tail_without_mutation(self):
        a={'reaction_angular_impulse_nms':[0.,0.,-0.007785992075072017]}
        b={'reaction_angular_impulse_nms':[0.,0.,-0.0077859920750720174]}
        before=repr(a);d=compare(a,b)
        self.assertEqual(len(d),1);self.assertEqual(repr(a),before)
        self.assertEqual(d[0]['error'],8.673617379884035e-19)

    def test_rejects_real_motion_difference(self):
        with self.assertRaises(ValueError):compare({'mechanism_qpos':[.1]},{'mechanism_qpos':[.100000001]})

    def test_rejects_guard_count_and_schema_differences(self):
        for a,b in [(True,False),(1,2),({'x':1},{}),(1.,1)]:
            with self.assertRaises(ValueError):compare(a,b)

    def test_nonfinite_never_passes(self):
        for x in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):compare(x,x)


if __name__=='__main__':unittest.main()
