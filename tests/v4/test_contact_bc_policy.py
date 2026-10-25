import unittest
import numpy as np
from contact_bc_policy import encode_episode,fit_normalization,normalize,build_actor
from contact_bc_data import STATE_FIELDS


def data():
    n=12
    a={k:np.tile(np.arange(size,dtype=np.float32),(n,1)) for k,size in STATE_FIELDS.items()}
    a.update(frame=np.arange(1,n+1),sequence_id=np.ones(n,dtype=int),terrain_height_m=np.ones((n,2,2)),
        terrain_known=np.ones((n,2,2),bool),terrain_origin_xy_m=np.zeros((n,2)),terrain_resolution_m=np.full(n,.1),
        terrain_age_frames=np.arange(n)%3,previous_applied_action_si=np.zeros((n,4)),previous_drive_load=np.zeros((n,4)),
        previous_drive_load_valid=np.ones(n,bool),goal_world_m=np.ones((n,3)))
    return a


class PolicyTest(unittest.TestCase):
    def test_history_is_causal_and_cannot_cross_gap(self):
        a=data();a['sequence_id'][6:]=2;a['frame'][6:]+=1
        x,spec,_=encode_episode(a);valid=spec['feature_names'].index('history_minus_3_valid[0]')
        self.assertEqual(x[5,valid],1);self.assertEqual(x[6,valid],0);self.assertEqual(x[9,valid],1)
        a['mechanism_qpos'][10:]=999
        changed,_,_=encode_episode(a);np.testing.assert_array_equal(x[:10],changed[:10])

    def test_invalid_goal_cannot_leak_retargeting(self):
        a=data();x,_,_=encode_episode(a,goal_valid=False);a['goal_world_m'][:]=999
        y,_,_=encode_episode(a,goal_valid=False);np.testing.assert_array_equal(x,y)

    def test_normalization_fits_development_only(self):
        train=np.array([[1.,0.],[3.,1.]],dtype=np.float32)
        mean,scale,frozen=fit_normalization(train,[0]);np.testing.assert_array_equal(mean,[2.,0.])
        heldout=np.array([[100.,1.]],dtype=np.float32);out=normalize(heldout,mean,scale,frozen)
        np.testing.assert_array_equal(mean,[2.,0.]);self.assertEqual(out[0,1],1.)

    def test_actor_can_output_all_four_axes_and_reload(self):
        import torch
        torch.manual_seed(1);actor=build_actor(torch,5)
        actor[-2].bias.data.fill_(.5)
        x=torch.zeros((2,5));y=actor(x)
        self.assertEqual(tuple(y.shape),(2,4));self.assertTrue(torch.isfinite(y).all());self.assertTrue((y.abs()<=1).all())
        self.assertTrue((y[:,0].abs()>0).all())
        other=build_actor(torch,5);other.load_state_dict(actor.state_dict());torch.testing.assert_close(y,other(x),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
