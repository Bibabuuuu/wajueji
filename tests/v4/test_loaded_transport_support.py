"""CPU-only policy binding and checkpoint metadata regressions."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

import layered_47_cycle_checkpoint as checkpoint
from layered_47_loaded_transport_support import validate_window,validate_activation_clock,POLICY


def policy():
    names=('layered_47_loaded_transport_support.py','loaded_transport_support.h')
    return dict(policy=POLICY,clock='native_time_s',start_s=1.,end_s=5.,ramp_s=.05,
                bindings={n:checkpoint.checkpoint_identity(Path(checkpoint.__file__).with_name(n))['sha256'] for n in names})


class LoadedTransportSupportTest(unittest.TestCase):
    def test_wrong_accumulated_clock_is_rejected_before_native_entry(self):
        validate_activation_clock(191.0666667,191.1166667)
        for start in [239.7166667,190.,192.,float('nan')]:
            with self.subTest(start=start),self.assertRaises(ValueError):validate_activation_clock(191.0666667,start)

    def test_finite_positive_window_required(self):
        validate_window(1,2)
        for start,end in [(0,float('nan')),(0,float('inf')),(1,1),(2,1),(1,1.05)]:
            with self.subTest(start=start,end=end),self.assertRaises(ValueError):validate_window(start,end)

    def test_legacy_metadata_remains_valid(self):
        checkpoint.validate_loaded_transport_support({}, {})

    def test_valid_policy_and_each_corrupt_binding(self):
        p=policy();checkpoint.validate_loaded_transport_support({'loaded_transport_support':p},p['bindings'])
        for name in p['bindings']:
            q=deepcopy(p);q['bindings'][name]='old'
            with self.assertRaises(ValueError):checkpoint.validate_loaded_transport_support({'loaded_transport_support':q},p['bindings'])
            b=dict(p['bindings']);b[name]='old'
            with self.assertRaises(ValueError):checkpoint.validate_loaded_transport_support({'loaded_transport_support':p},b)

    def test_invalid_policy_rejected_even_if_feedback_has_expired(self):
        p=policy()
        for k,v in [('policy','unknown'),('ramp_s',1),('start_s',-1),('end_s',float('nan')),('end_s',1.05)]:
            q=deepcopy(p);q[k]=v
            with self.subTest(field=k),self.assertRaises(ValueError):checkpoint.validate_loaded_transport_support({'loaded_transport_support':q},p['bindings'])

    def test_save_preserves_policy_without_mutating_parent(self):
        p=policy();parent=dict(source_identity={'sha256':'source'},lineage_root='root',loaded_transport_support=p)
        saved_parent=deepcopy(parent)
        with TemporaryDirectory() as directory:
            target=Path(directory)/'state.npz'
            def save_arrays(path,arrays):
                np.savez(path,**arrays)
                return {'sha256':checkpoint.checkpoint_identity(path)['sha256']}
            with patch.object(checkpoint,'validate_controller'),patch.object(checkpoint,'validate_sweep',return_value={}),patch.object(checkpoint.sparse,'write_checkpoint_atomic',side_effect=save_arrays):
                checkpoint.save(target,{'simulation_time_s':np.array([3.])},{'native_state_kind':'full'}, {},parent=parent,stage='transport',cycle_index=1,truck_enabled=True,solver_profile={},runtime_bindings=p['bindings'])
            actual=json.loads(Path(str(target)+'.json').read_text())
            self.assertEqual(actual['loaded_transport_support'],p)
            self.assertFalse(actual['independent_restore_qualified'])
            self.assertEqual(parent,saved_parent)


if __name__=='__main__':unittest.main()
