"""Training boundaries must never silently become continuous-cycle evidence."""
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import layered_47_cycle_checkpoint as checkpoint
from test_cycle_checkpoint import controller, sweep


def reset():
    return dict(mode='fast_contact_collection_v1', continuous_physics=False,
                entries=[dict(soil_indices=[1,3], rock_indices=[2], soil_count=2,
                              rock_count=1, translation_m=[-5,-3,-2],
                              ownership='withdrawn_for_training_only_not_truck_delivery')])


class TrainingResetTest(unittest.TestCase):
    def test_legacy_diagnostic_cannot_erase_training_boundary(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'diagnostic.npz'
            np.savez(path,placeholder=np.zeros(1))
            Path(str(path)+'.json').write_text(json.dumps(dict(
                schema='layered_47_manual_contact_diagnostic_snapshot_v1',
                sha256=checkpoint.checkpoint_identity(path)['sha256'],runtime_state='running')))
            (path.parent/'report.json').write_text(json.dumps({'training_reset':reset()}))
            with patch.object(checkpoint.sparse,'_validate'), self.assertRaisesRegex(ValueError,'full_cycle_checkpoint'):
                checkpoint.load(path,allow_training_reset=True)

    def test_invalid_inventory_and_continuity_rejected(self):
        checkpoint.validate_training_reset({'training_reset':reset()})
        for change in ('duplicate', 'range', 'count', 'nan', 'continuous', 'empty', 'ownership'):
            value=reset();entry=value['entries'][0]
            if change=='duplicate': value['entries'].append(deepcopy(entry))
            if change=='range': entry['rock_indices']=[47]
            if change=='count': entry['soil_count']=3
            if change=='nan': entry['translation_m'][0]=float('nan')
            if change=='continuous': value['continuous_physics']=True
            if change=='empty': value['entries']=[]
            if change=='ownership': entry['ownership']='delivered'
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint.validate_training_reset({'training_reset':value})

    def test_opt_in_and_save_propagation(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'point.npz'
            np.savez(path,simulation_time_s=np.array([2.]))
            bindings={name:checkpoint.checkpoint_identity(Path(checkpoint.__file__).with_name(name))['sha256']
                      for name in checkpoint.PHYSICS_FILES}
            parent=dict(source_identity={'sha256':'source'},lineage_root='root',training_reset=reset())
            with patch.object(checkpoint.sparse,'write_checkpoint_atomic',return_value=checkpoint.checkpoint_identity(path)):
                checkpoint.save(path,{'simulation_time_s':np.array([2.])},controller(),sweep(),
                                parent=parent,stage='prepoint',cycle_index=2,truck_enabled=True,
                                solver_profile={},runtime_bindings=bindings)
            with self.assertRaisesRegex(ValueError,'not_continuous'):
                checkpoint.load(path)
            with patch.object(checkpoint.sparse,'_validate'):
                _, metadata=checkpoint.load(path,allow_training_reset=True)
            self.assertEqual(metadata['training_reset'],reset())
            self.assertFalse(metadata['training_eligible'])


if __name__=='__main__': unittest.main()
