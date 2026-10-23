"""Bounded fixed-start, same-process D152 runtime reconstruction check.

This exercises only saved, already executed SI commands. It does not create a
policy API, qualify arbitrary mid-episode reset, or produce a training reward.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
sys.path[:0] = [str(ROOT), str(ROOT/'src/v4'),
    str(ROOT/'work/contact-multisource-lift-20260924-v1'),
    str(ROOT/'work/contact-full-response-20260924-v1'),
    str(ROOT/'work/contact-preview-lift-20260924-v1')]

SOURCE = ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
SOURCE_SHA = '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'
ACTIONS = ROOT/'runs/roadshow-measurements-20261011/rl-env-reset-gate-20261012-v1/repeat-a/physics/actions.ndjson'
OUTPUT = ROOT/'runs/roadshow-measurements-20261011/rl-env-inprocess-reset-gate-20261012-v2'
FRAMES = 6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(arr):
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.tobytes(order='C'))
    return digest.hexdigest()


def snapshot_digest(arrays):
    return {key: array_digest(value) for key,value in sorted(arrays.items())}


def episode(label, commands):
    from candidate_multisource import MultisourceRuntime
    start = time.monotonic()
    with MultisourceRuntime(SOURCE, OUTPUT/label) as runtime:
        if runtime.metadata['stage'] != 'prepoint':
            raise ValueError('not_prepoint')
        initial = runtime.capture()
        initial_exact = {key: bool(np.array_equal(value, runtime.arrays[key]))
                         for key,value in initial.items()}
        if len(initial_exact) != 24 or not all(initial_exact.values()):
            raise ValueError('zero_step_source_restore_mismatch')
        initial_digest = snapshot_digest(initial)
        states = []
        for frame, command in enumerate(commands, 1):
            row = runtime.step(command)
            states.append(dict(frame=frame, arrays=snapshot_digest(runtime.capture()),
                               hard_guard=row['hard_guard'],
                               hard_guard_checks=row['hard_guard_checks'],
                               after_state=row['after_state']))
        bindings = runtime.bindings
        restored = runtime.restored
        count = runtime.frames
    gc.collect()
    return dict(label=label, wall_seconds=time.monotonic()-start,
                frames=count, initial_arrays=initial_digest,
                zero_step_exact=initial_exact,
                states=states, runtime_bindings=bindings,
                native_restore_errors=restored['cycle_errors'])


def main():
    if OUTPUT.exists():
        raise ValueError('immutable_output_already_exists')
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('source_sha_mismatch')
    original = [json.loads(line) for line in ACTIONS.open()][:FRAMES]
    if len(original) != FRAMES or [x['frame'] for x in original] != list(range(1,FRAMES+1)):
        raise ValueError('command_prefix_missing')
    if any(x['source'] != 'geometric_feedback_without_policy' for x in original):
        raise ValueError('command_source_not_geometry')
    commands = [np.asarray(x['applied_command_si'], dtype=float) for x in original]
    OUTPUT.mkdir(parents=True)
    report = dict(schema='roadshow_rl_inprocess_reset_gate_v1',
                  status='running', training_environment_qualified=False,
                  training_started=False, source_sha256=SOURCE_SHA,
                  action_log_sha256=sha(ACTIONS),
                  script_sha256=sha(__file__),
                  command_count=FRAMES,
                  reset_kind='destroy_and_reconstruct_runtime_in_one_python_process',
                  episodes=[])
    try:
        for label in ('episode-a','episode-b'):
            report['episodes'].append(episode(label, commands))
            print('EPISODE_DONE', label, flush=True)
        first,second = report['episodes']
        checks = {
            'same_initial_24_arrays': first['initial_arrays']==second['initial_arrays'] and len(first['initial_arrays'])==24,
            'same_every_poststep_24_arrays': all(a['arrays']==b['arrays'] and len(a['arrays'])==24
                                                 for a,b in zip(first['states'],second['states'])),
            'same_poststep_compact_state': all(a['after_state']==b['after_state']
                                               for a,b in zip(first['states'],second['states'])),
            'all_hard_guards': all(x['hard_guard'] for e in report['episodes'] for x in e['states']),
            'native_restore_exact': all(max(e['native_restore_errors'].values()) <= 1e-12
                                        for e in report['episodes']),
            'same_bindings': first['runtime_bindings']==second['runtime_bindings'],
            'frozen_source_unchanged': sha(SOURCE)==SOURCE_SHA,
        }
        report['checks']=checks
        report['status']='fixed_start_inprocess_reconstruction_passed' if all(checks.values()) else 'fixed_start_inprocess_reconstruction_failed'
    except BaseException as error:
        report.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        (OUTPUT/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not all(report['checks'].values()):
        raise ValueError('inprocess_reset_checks_failed')


if __name__ == '__main__':
    main()
