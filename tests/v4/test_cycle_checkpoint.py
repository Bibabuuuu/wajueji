"""Reject corrupted continuation envelopes before native launch or physics."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import unittest
from layered_47_cycle_checkpoint import validate_controller, validate_sweep, restore_sweep


def controller():
    return dict(controller_target=[0]*4,controller_target_velocity=[0]*4,
        bucket_integral_effort=[0,0,0,120000],bucket_active_curl_effort=[0,0,0,60000],
        bucket_linear_velocity_world_m_s=[1,2,3],bucket_angular_velocity_world_rad_s=[4,5,6],
        native_time_s=2,native_last_time_s=2-1/4800,native_timestep_s=1/4800,
        native_state_kind='full',native_warmstart=[0]*300,native_qacc=[0]*300,native_ctrl=[0]*12)


def sweep():
    return dict(total_generated_reaction_ns=[1,2,3],total_routed_reaction_ns=[.5,1,1.5],
        pending_velocity_reaction=dict(reaction_impulse_ns=[.5,1,1.5],
            reaction_angular_impulse_nms=[4,5,6],point_world_m=[6,0,2]),resume_qualified=False)


class CycleCheckpointTest(unittest.TestCase):
    def test_invalid_controller(self):
        for key,value in [
            ('bucket_active_curl_effort',[0,0,0,60001]),('bucket_integral_effort',[1,0,0,0]),
            ('native_last_time_s',0),('native_time_s',float('nan')),
            ('native_qacc',[0]*299),('native_ctrl',[float('nan')]*12),
            ('bucket_linear_velocity_world_m_s',[1,2]),('native_state_kind','unknown')]:
            with self.subTest(key=key):
                data=controller();data[key]=value
                with self.assertRaises((ValueError,KeyError)):validate_controller(data,47)

    def test_full_controller_and_pending_impulse_are_preserved(self):
        validate_controller(controller(),47)
        data=sweep();original=deepcopy(data)
        runtime=SimpleNamespace(old_pose=None)
        restore_sweep(runtime,data)
        self.assertEqual(runtime.pending,original['pending_velocity_reaction'])
        self.assertTrue(np.array_equal(runtime.generated-runtime.routed,runtime.pending['reaction_impulse_ns']))
        runtime.pending['point_world_m'][0]=10
        self.assertEqual(data,original)

    def test_dropped_pending_impulse_is_rejected(self):
        data=sweep();data['pending_velocity_reaction']=None
        with self.assertRaisesRegex(ValueError,'balance'):validate_sweep(data)

    def test_mid_frame_restore_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'mid_frame'):
            restore_sweep(SimpleNamespace(old_pose='in_progress'),sweep())
