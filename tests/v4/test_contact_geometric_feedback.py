import unittest
import numpy as np
from contact_geometric_feedback import bounded_velocity, FeedbackConfig, ContactGeometryFeedback


class FeedbackTest(unittest.TestCase):
    def test_ground_constraint_preserves_independent_curl(self):
        action=bounded_velocity(np.eye(4),[0.,0.,-1.,.7],[0.,0.,1.,0.],0.)
        self.assertGreaterEqual(action[2],-1e-7)
        self.assertAlmostEqual(action[3],.7,places=4)

    def test_coupled_axes_can_cancel_downward_motion(self):
        jac=np.eye(4);jac[2,3]=-1.
        action=bounded_velocity(jac,[0.,0.,0.,.5],[0.,0.,1.,-1.],0.)
        self.assertGreater(action[2],.49)
        self.assertGreater(action[3],.49)
        self.assertGreaterEqual((jac@action)[2],-1e-7)

    def test_normalized_axis_envelope(self):
        action=bounded_velocity(np.eye(4),[20.,-20.,.5,0.],[0.,0.,1.,0.],-.2)
        self.assertLessEqual(np.max(abs(action)),1.)
        self.assertAlmostEqual(action[0],1.)
        self.assertAlmostEqual(action[1],-1.)

    def test_approach_preserves_pitch_when_translation_saturates(self):
        jac=np.eye(4);jac[3]=[0.,0.,1.,1.]
        action=bounded_velocity(jac,[0.,0.,2.,0.],[0.,0.,1.,0.],-10.,exact_pitch=True)
        self.assertAlmostEqual((jac@action)[3],0.,places=7)
        self.assertLessEqual((jac@action)[2],1.)

    def test_impossible_ground_recovery_rejected(self):
        with self.assertRaisesRegex(ValueError,'infeasible'):
            bounded_velocity(np.eye(4),np.zeros(4),[0.,0.,1.,0.],2.)

    def test_nonfinite_problem_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid'):
            bounded_velocity(np.eye(4),[0.,0.,np.nan,0.],np.ones(4),0.)
        with self.assertRaises(ValueError):FeedbackConfig(ground_clearance_m=.001)
        with self.assertRaises(ValueError):FeedbackConfig(cut_drop_m=np.nan)

    def test_phase_skips_and_reordered_calls_rejected_before_queries(self):
        controller=ContactGeometryFeedback.__new__(ContactGeometryFeedback)
        controller.last_frame=0;controller.last_phase='approach_contact'
        with self.assertRaisesRegex(ValueError,'noncontiguous'):
            controller.command(frame=2,qpos=None,phase='approach_contact',bc_action=None,projector=None)
        with self.assertRaisesRegex(ValueError,'phase_order'):
            controller.command(frame=1,qpos=None,phase='curl_load',bc_action=None,projector=None)


if __name__=='__main__':unittest.main()
