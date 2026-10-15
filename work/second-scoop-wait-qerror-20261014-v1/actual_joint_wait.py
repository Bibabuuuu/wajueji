"""CPU candidate only: actual-joint wait error, unchanged frozen observation.

Inherited .step/.reset reject physical use. The legacy alignment report remains
the observation source; a separate copy supplies the command error to the
unchanged command_at law. This action mapping requires a new replay contract.
"""
import numpy as np
from yaw_transfer import YawTransferEnv

VERSION = 'wait_actual_joint_error_strength_v2_cpu'
AXES = np.array([7, 9, 12, 15])


def actual_joint_error(reference_q, cursor, actual_q):
    reference_q = np.asarray(reference_q, dtype=float)
    actual_q = np.asarray(actual_q, dtype=float)
    if (reference_q.ndim != 2 or reference_q.shape[1] != 19
            or len(reference_q) < 2 or actual_q.shape != (19,)
            or not np.isfinite(cursor) or not 0 <= cursor <= len(reference_q)-1
            or not np.isfinite(actual_q).all()):
        raise ValueError('actual_joint_wait_state_contract')
    first = int(cursor)
    last = min(first+1, len(reference_q)-1)
    pair = reference_q[[first, last]][:, AXES]
    if not np.isfinite(pair).all():
        raise ValueError('actual_joint_wait_reference_contract')
    fraction = cursor-first
    target = pair[0]*(1-fraction)+pair[1]*fraction
    return target-actual_q[AXES]


def command_alignment(legacy_alignment, reference_q, cursor, actual_q):
    """Do not mutate report/observations, progress, trust or corridor gates."""
    result = dict(legacy_alignment)
    error = actual_joint_error(reference_q, cursor, actual_q)
    result.update(action_mapping_version=VERSION,
                  actual_joint_error_si=error.tolist(),
                  legacy_wait_target_error_si=list(legacy_alignment['wait_target_error_si']))
    if legacy_alignment['reference_limit']-cursor <= 1e-12:
        # Existing command_at consumes this field, still using its unchanged
        # 0..100% strength, 10% axis speed and remaining-error/60Hz caps.
        result['wait_target_error_si'] = error.tolist()
        result['wait_error_source'] = 'interpolated_reference_q_minus_actual_q'
    else:
        result['wait_error_source'] = 'legacy_unused_while_following'
    return result


class ActualJointWaitEnv(YawTransferEnv):
    """No physical admission. Original model input encoder is inherited."""
    def alignment_state(self, stage):
        legacy = super().alignment_state(stage)
        return command_alignment(legacy, self.reference_q, self.cursor,
                                 self.runtime.bridge.state['mechanism_qpos'])

    def attach(self, *args, **kwargs):
        obs = super().attach(*args, **kwargs)
        self.report.update(action_mapping_version=VERSION,
                           observation_contract='unchanged_221_legacy_target_error',
                           replay_compatible_with_old_mapping=False,
                           physical_admission=False)
        return obs
