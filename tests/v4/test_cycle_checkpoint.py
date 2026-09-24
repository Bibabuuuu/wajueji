"""Reject corrupted continuation envelopes before native launch or physics."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import layered_47_cycle_checkpoint as checkpoint
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
    def test_fault_final_and_budget_saves_cannot_seed_next_scoop(self):
        from run_layered_47_sparse_manual_contact_episode import cycle_snapshot_stage
        self.assertEqual(cycle_snapshot_stage('final',fault_frozen=True),'failure')
        self.assertEqual(cycle_snapshot_stage('failure',fault_frozen=False),'failure')
        self.assertEqual(cycle_snapshot_stage('budget',fault_frozen=False),'hold')
        self.assertEqual(cycle_snapshot_stage('final',fault_frozen=False,zero_probe_frames=3),'hold')
        self.assertEqual(cycle_snapshot_stage('final',fault_frozen=False),'loaded_manual_endpoint')

    def test_continuation_metadata_and_physics_identity_fail_closed(self):
        # Exercise the envelope independently of the already separately tested
        # 256441-particle array schema, without allocating a physical scene.
        with TemporaryDirectory() as directory:
            path=Path(directory)/'state.npz'
            np.savez(path,placeholder=np.zeros(1))
            baseline=dict(schema=checkpoint.SCHEMA,
                checkpoint_sha256=checkpoint.checkpoint_identity(path)['sha256'],
                stage='prepoint',cycle_index=2,truck_enabled=True,
                controller_state=controller(),moving_bucket_sweep_state=sweep(),
                runtime_bindings={name:checkpoint.checkpoint_identity(
                    Path(checkpoint.__file__).with_name(name))['sha256']
                    for name in checkpoint.PHYSICS_FILES})
            sidecar=Path(str(path)+'.json')
            with patch.object(checkpoint.sparse,'_validate'):
                for index in (2,3):
                    sidecar.write_text(json.dumps(dict(baseline,cycle_index=index)))
                    _,loaded=checkpoint.load(path)
                    self.assertEqual(loaded['cycle_index'],index)
                    self.assertEqual(loaded['source_metadata_identity']['sha256'],
                                     checkpoint.checkpoint_identity(sidecar)['sha256'])
                for key,value in [('stage','failure'),('stage','unknown'),('cycle_index',4),
                                  ('truck_enabled',0),('checkpoint_sha256','changed'),
                                  ('runtime_bindings',{})]:
                    with self.subTest(key=key,value=value):
                        sidecar.write_text(json.dumps(dict(baseline,**{key:value})))
                        with self.assertRaises(ValueError):checkpoint.load(path)

    def test_invalid_controller(self):
        for key,value in [
            ('bucket_active_curl_effort',[0,0,0,60001]),('bucket_integral_effort',[1,0,0,0]),
            ('native_last_time_s',0),('native_time_s',float('nan')),
            ('native_qacc',[0]*299),('native_ctrl',[float('nan')]*12),
            ('bucket_linear_velocity_world_m_s',[1,2]),('native_state_kind','unknown')]:
            with self.subTest(key=key):
                data=controller();data[key]=value
                with self.assertRaises((ValueError,KeyError)):validate_controller(data,47)

    def test_passive_history_policy_requires_matching_implementation(self):
        policy={'material_history_policy':'zero_hardening_frozen_jp_v1'}
        name='soil_passive_jp_history.py'
        bindings={name:checkpoint.checkpoint_identity(Path(checkpoint.__file__).with_name(name))['sha256']}
        checkpoint.validate_material_history_policy(policy,bindings)
        checkpoint.validate_material_history_policy({}, {})
        for bad_policy,bad_bindings in [
                (policy,{}),(policy,{name:'wrong'}),
                ({'material_history_policy':'unknown'},bindings)]:
            with self.assertRaises(ValueError):
                checkpoint.validate_material_history_policy(bad_policy,bad_bindings)

    def test_passive_history_policy_is_saved_for_continuation(self):
        name='soil_passive_jp_history.py'
        bindings={name:checkpoint.checkpoint_identity(Path(checkpoint.__file__).with_name(name))['sha256']}
        parent=dict(source_identity={'sha256':'source'},lineage_root='root',
                    material_history_policy='zero_hardening_frozen_jp_v1')
        with TemporaryDirectory() as directory:
            path=Path(directory)/'point.npz'
            with patch.object(checkpoint.sparse,'write_checkpoint_atomic',return_value={'sha256':'result'}):
                checkpoint.save(path,{'simulation_time_s':np.array([1.])},controller(),sweep(),
                    parent=parent,stage='prepoint',cycle_index=2,truck_enabled=True,
                    solver_profile={},runtime_bindings=bindings)
            saved=json.loads(Path(str(path)+'.json').read_text())
            self.assertEqual(saved['material_history_policy'],parent['material_history_policy'])
            self.assertEqual(saved['runtime_bindings'],bindings)

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
