import copy
import unittest
from manual_residual_sac_env import environment_contract, require_environment_contract


class ContractTests(unittest.TestCase):
    def test_current_strength_contract_is_accepted(self):
        require_environment_contract({'environment_contract':environment_contract()})

    def test_same_dimensions_do_not_make_legacy_action_compatible(self):
        receipt={'environment_contract':environment_contract('signed_residual_v2')}
        self.assertEqual(receipt['environment_contract']['observation_size'],221)
        with self.assertRaises(ValueError):require_environment_contract(receipt)

    def test_score_handoff_or_source_change_is_rejected(self):
        for key in ['terminal_score_version','action_handoff_version','source_sha256']:
            contract=copy.deepcopy(environment_contract());contract[key]='old-version'
            with self.assertRaises(ValueError):require_environment_contract({'environment_contract':contract})
        contract=environment_contract();contract['code']['manual_residual_sac_env.py']='old-code'
        with self.assertRaises(ValueError):require_environment_contract({'environment_contract':contract})

    def test_legacy_or_wrong_dimensions_cannot_pass_by_omitting_metadata(self):
        with self.assertRaises(ValueError):require_environment_contract({})
        contract=environment_contract();contract['observation_size']=215
        with self.assertRaises(ValueError):require_environment_contract({'environment_contract':contract})


if __name__=='__main__':unittest.main()
