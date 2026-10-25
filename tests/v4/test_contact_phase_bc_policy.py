import unittest
import numpy as np
from tests.v4.test_contact_bc_policy import data
from contact_bc_policy import encode_episode as old_encode
from contact_phase_bc_policy import encode_episode,fit_normalization,normalize
from contact_phase_data import PHASES


def example():
    a=data();n=len(a['frame']);a['phase_id']=np.minimum(np.arange(n)//3,4).astype(np.int8)
    a['phase_one_hot']=np.eye(5,dtype=np.float32)[a['phase_id']]
    a['latest_allowed_phase_evidence_frame']=a['frame']-1
    return a


class PhasePolicyTest(unittest.TestCase):
    def test_adds_only_current_phase_without_normalizing_categories(self):
        a=example();old,_,_=old_encode(a);x,spec,_=encode_episode(a)
        np.testing.assert_array_equal(x[:,:-5],old)
        np.testing.assert_array_equal(x[:,-5:],a['phase_one_hot'])
        self.assertEqual(spec['phase_names'],list(PHASES))
        mean,scale,frozen=fit_normalization(x,spec['continuous_indices'])
        np.testing.assert_array_equal(normalize(x,mean,scale,frozen)[:,-5:],a['phase_one_hot'])

    def test_rejects_same_step_future_phase_evidence(self):
        a=example();a['latest_allowed_phase_evidence_frame'][3]=a['frame'][3]
        with self.assertRaisesRegex(ValueError,'noncausal'):encode_episode(a)

    def test_rejects_unknown_fractional_or_inconsistent_labels(self):
        for kind in ('unknown','fractional','mismatch'):
            a=example()
            if kind=='unknown':a['phase_id'][0]=-1
            elif kind=='fractional':a['phase_id']=a['phase_id'].astype(float)+.1
            else:a['phase_one_hot'][0]=0
            with self.subTest(kind=kind),self.assertRaises(ValueError):encode_episode(a)

    def test_future_phase_cannot_change_earlier_features(self):
        a=example();before,_,_=encode_episode(a)
        a['phase_id'][9:]=4;a['phase_one_hot'][9:]=np.eye(5,dtype=np.float32)[4]
        after,_,_=encode_episode(a)
        np.testing.assert_array_equal(before[:9],after[:9])

if __name__=='__main__':unittest.main()
