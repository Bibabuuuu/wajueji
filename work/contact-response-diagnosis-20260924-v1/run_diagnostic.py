"""One bounded baseline reproduction with additive observers, no tuning."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from observe import instrument
import layered_47_cycle_bridge as cycle_bridge
import layered_47_cycle_runtime as runtime
import run_contact_geometric_probe as probe

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
OUT = ROOT/'runs/contact-response-diagnosis-20260924-v1/contact1200'
BASE = ROOT/'runs/contact-geometric-probe-20260924-v9-handoff/contact1200'
baseline = [json.loads(x) for x in (BASE/'physics/frames.ndjson').read_text().splitlines()]
original_instrument = cycle_bridge.instrument
OriginalRuntime = runtime.CycleRuntime


def write(path, value):
    with path.open('a') as file:
        file.write(json.dumps(value, allow_nan=False, default=lambda x: x.tolist())+'\n')


def observed_instrument(source):
    original = original_instrument(source)
    frozen = (BASE/'physics/cycle/capacity-bridge.cc').read_text()
    if original.replace(str(OUT),str(BASE)) != frozen:
        raise ValueError('observer_base_generated_source_changed')
    result, stripped = instrument(original, OUT/'native-substeps.tsv')
    assert stripped == original
    (OUT/'observer-manifest.json').write_text(json.dumps(dict(
        original_generated_sha256=hashlib.sha256(original.encode()).hexdigest(),
        observed_generated_sha256=hashlib.sha256(result.encode()).hexdigest(),
        inserted_observations_only=True, window=[940,972], native_hz=4800,
        force_timing='MuJoCo force diagnostics for just-integrated substep; q/v before and after both logged',
        observer_sha256=probe.sha(W/'observe.py')), indent=2)+'\n')
    return result


class ObservedRuntime(OriginalRuntime):
    def __enter__(self):
        super().__enter__()
        original = self.sweep.step_mpm

        def step_mpm(step, dt, origins):
            frame = self.frames+1
            record = 940 <= frame <= 972
            pending = copy.deepcopy(self.sweep.pending) if record else None
            raw = {}

            def capture_raw(*args, **kwargs):
                measured, particles = step(*args, **kwargs)
                if record:
                    raw.update(copy.deepcopy(measured))
                return measured, particles

            measured, particles = original(capture_raw, dt, origins)
            if record:
                write(OUT/'wrench-split.ndjson', dict(frame=frame, dt=dt, origins=origins,
                    raw_mpm=raw, applied=measured, pending_from_previous_frame=pending,
                    current_sweep=copy.deepcopy(self.sweep.current)))
            return measured, particles

        self.sweep.step_mpm = step_mpm
        return self

    def step(self, command):
        # Same closed-loop policy/ground protection runs in the original probe.
        expected = baseline[self.frames]
        if list(command) != expected['applied_command_si']:
            raise ValueError('observer_command_mismatch:'+str(self.frames+1))
        result = super().step(command)
        for field in ('after_state','moving_bucket_sweep','hencky_material_guard','hard_guard_checks'):
            if result[field] != expected[field]:
                raise ValueError('observer_physics_mismatch:'+str(self.frames)+':'+field)
        if 940 <= self.frames <= 972:
            write(OUT/'bridge-frames.ndjson',dict(frame=self.frames,state=self.bridge.state))
        if self.frames == len(baseline):
            with np.load(BASE/'physics/failure.npz') as expected_arrays:
                exact = {k:bool(np.array_equal(v,expected_arrays[k])) for k,v in self.capture().items()}
            (OUT/'endpoint-array-check.json').write_text(json.dumps(exact,indent=2)+'\n')
            if not all(exact.values()):raise ValueError('observer_endpoint_arrays_mismatch')
        return result


if __name__ == '__main__':
    # Prevent alternate flags from silently enlarging this single diagnostic.
    if len(sys.argv)!=1:raise ValueError('diagnostic_no_extra_arguments')
    if OUT.exists():raise FileExistsError(OUT)
    sys.argv = [str(Path(probe.__file__)), '--source', str(ROOT/'runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz'),
        '--training-reset-checkpoint','--policy',str(ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'),
        '--policy-sha256','b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095',
        '--training-python','/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',
        '--phase-contract',str(W/'protocol.json'),'--output',str(OUT),'--frames','1200',
        '--contact-window','--prefix-review',str(ROOT/'work/contact-geometric-feedback-20260924-v9/startup-review.json')]
    cycle_bridge.instrument = observed_instrument
    runtime.CycleRuntime = ObservedRuntime
    try:
        probe.main()
    finally:
        cycle_bridge.instrument = original_instrument
        runtime.CycleRuntime = OriginalRuntime
