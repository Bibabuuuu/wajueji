"""Native 60 Hz history BC conditioned on the causal five-stage observer.

This is a distinct model contract. Original unconditioned actors remain intact.
The phase describes information available before the action, never a future
human annotation or the phase observed at the end of that same physical step.
"""
import hashlib
import json
import numpy as np
from contact_bc_policy import encode_episode as encode_unconditioned
from contact_bc_policy import fit_normalization, normalize, build_actor
from contact_phase_data import PHASES

SCHEMA = 'contact_bc_native60_causal_phase_history_mlp_v1'


def encode_episode(arrays, *, goal_valid=True):
    frames = np.asarray(arrays['frame'])
    ids = np.asarray(arrays['phase_id'])
    one_hot = np.asarray(arrays['phase_one_hot'])
    evidence = np.asarray(arrays['latest_allowed_phase_evidence_frame'])
    n = len(frames)
    if (ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(PHASES))):
        raise ValueError('invalid_causal_phase_ids')
    if evidence.shape != (n,) or not np.array_equal(evidence, frames - 1):
        raise ValueError('noncausal_phase_evidence')
    expected = np.eye(len(PHASES), dtype=np.float32)[ids]
    if one_hot.shape != expected.shape or not np.array_equal(one_hot, expected):
        raise ValueError('phase_id_one_hot_mismatch')
    x, original, _ = encode_unconditioned(arrays, goal_valid=goal_valid)
    spec = dict(original, schema=SCHEMA,
                feature_names=original['feature_names'] + [f'phase[{name}]' for name in PHASES],
                phase_routing='external_causal_five_stage_latched_before_action',
                phase_names=list(PHASES), phase_evidence='frame_end_at_most_action_frame_minus_one')
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return np.concatenate((x, expected), axis=1), spec, digest
