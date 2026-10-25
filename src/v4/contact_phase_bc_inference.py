"""Opt-in causal-phase inference, separate from the frozen unconditioned worker."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np
from contact_bc_inference import InferenceWorker as BaseWorker, LIMITS, sha
from contact_phase_bc_policy import SCHEMA, encode_episode, normalize, build_actor
from contact_phase_data import PHASES


def with_phase(row, phase, evidence_frame):
    if phase not in PHASES or type(evidence_frame) is not int or evidence_frame != int(row['frame']) - 1:
        raise ValueError('noncausal_live_phase')
    index = PHASES.index(phase)
    return dict(row, phase_id=np.int8(index), phase_one_hot=np.eye(5, dtype=np.float32)[index],
                latest_allowed_phase_evidence_frame=np.int64(evidence_frame))


class OnlinePhaseFeatures:
    def __init__(self, *, goal_valid=True):
        self.goal_valid = goal_valid
        self.rows = deque(maxlen=10)
        self.last_frame = self.last_sequence = None

    def update(self, row):
        frame, sequence = int(row['frame']), int(row['sequence_id'])
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError('nonmonotonic_live_frame')
        candidate = deque(self.rows, maxlen=10)
        if self.last_frame is not None and (frame != self.last_frame + 1 or sequence != self.last_sequence):
            candidate.clear()
        candidate.append(row)
        arrays = {key: np.stack([r[key] for r in candidate]) for key in row}
        x, _, digest = encode_episode(arrays, goal_valid=self.goal_valid)
        self.rows, self.last_frame, self.last_sequence = candidate, frame, sequence
        return x[-1], digest


class PhaseInferenceWorker(BaseWorker):
    # Transport/output validation stays identical; only the explicit worker entry changes.
    def __init__(self, python, checkpoint, expected_sha, *, stderr_path, timeout=15):
        self.timeout = timeout
        self.proc = None
        self.request_id = 0
        self.err = open(stderr_path, 'w')
        self.selector = selectors.DefaultSelector()
        try:
            self.proc = subprocess.Popen(
                [str(python), str(Path(__file__).resolve()), '--checkpoint', str(checkpoint), '--sha256', expected_sha],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1)
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            self.ready = self._receive()
            if (self.ready.get('status') != 'ready' or self.ready.get('checkpoint_sha256') != expected_sha
                    or self.ready.get('schema') != SCHEMA):
                raise ValueError('phase_inference_worker_not_ready')
        except BaseException:
            self.close()
            raise


def worker(checkpoint, expected_sha):
    import torch
    if sha(checkpoint) != expected_sha:
        raise ValueError('policy_checkpoint_sha256')
    torch.set_num_threads(2)
    c = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if c['schema'] != SCHEMA or c['action_hz'] != 60 or c['action_limits_si'] != LIMITS.tolist():
        raise ValueError('phase_policy_contract')
    spec = c['feature_spec']
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (digest != c['feature_spec_sha256'] or spec['phase_names'] != list(PHASES)
            or spec['feature_names'][-5:] != [f'phase[{p}]' for p in PHASES]):
        raise ValueError('phase_policy_feature_spec')
    n = len(spec['feature_names'])
    actor = build_actor(torch, n)
    actor.load_state_dict(c['actor_state_dict'])
    actor.eval()
    mean, scale, frozen = (c[k].numpy() for k in ('normalization_mean', 'normalization_scale', 'normalization_frozen'))
    if (any(v.shape != (n,) for v in (mean, scale, frozen)) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or np.any(mean[-5:]) or np.any(scale[-5:] != 1) or np.any(frozen[-5:])):
        raise ValueError('phase_policy_normalization')
    print(json.dumps(dict(status='ready', schema=SCHEMA, checkpoint_sha256=expected_sha,
                          feature_spec_sha256=digest, feature_count=n, device='cpu', action_hz=60)), flush=True)
    for line in sys.stdin:
        try:
            packet = json.loads(line)
            x = np.asarray(packet['features'], dtype=np.float32)
            if (x.shape != (n,) or not np.isfinite(x).all() or packet['feature_spec_sha256'] != digest
                    or not np.any(np.all(np.eye(5, dtype=np.float32) == x[-5:], axis=1))):
                raise ValueError('phase_policy_features')
            start = time.monotonic()
            with torch.inference_mode():
                action = actor(torch.from_numpy(normalize(x[None], mean, scale, frozen))).numpy()[0]
            if not np.isfinite(action).all():
                raise ValueError('policy_nonfinite')
            print(json.dumps(dict(id=packet['id'], action_normalized=action.tolist(),
                                  inference_seconds=time.monotonic()-start), allow_nan=False), flush=True)
        except BaseException as error:
            print(json.dumps(dict(error=repr(error))), flush=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    worker(args.checkpoint, args.sha256)
