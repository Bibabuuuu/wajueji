"""Read-only CPU predicates on recorded poses; no native entry or stepping."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib, json, signal, sys, time
import numpy as np

R = Path('/home/qunwang/wajueji')
W = Path(__file__).resolve().parent
ORIGINAL = R / 'work/edge-first-two-scoop-20261015-v1'
sys.path.insert(0, str(ORIGINAL))
import run_prepoint_continuation as m
old_minimum = sys.modules['minimum_guard']
old_phase = sys.modules['rise_hold_guard']
for name in ['minimum_guard', 'rise_hold_guard']:
    del sys.modules[name]
sys.path.insert(0, str(W))
new_minimum = importlib.import_module('minimum_guard')
new_phase = importlib.import_module('rise_hold_guard')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU180secondlimit')))
    signal.alarm(180)
    out = W / 'cpu-v1'
    out.mkdir(exist_ok=False)
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        assert value, name
    def outcome(fn):
        try:
            return 'pass', fn()
        except ValueError as exc:
            return 'reject', str(exc)
    D = m.OUT
    frozen = [D/'report.json', D/'read-only-audit.json', D/'display/states.json',
              D/'first-returned.npz', D/'first-returned.control.json',
              D/'scoop-2-plan/report.json', D/'scoop-2-plan/approach.npz',
              D/'audit-derived/scoop-2-terrain.npz', ORIGINAL/'closure.json',
              Path(old_minimum.__file__), Path(old_phase.__file__),
              ORIGINAL/'run_prepoint_continuation.py', m.MODEL]
    frozen_hashes = {str(p): sha(p) for p in frozen}
    check('isolated_candidate_loaded', Path(new_minimum.__file__).parent == W and Path(new_phase.__file__).parent == W)
    check('old_runner_remains_bound_to_old_guard', m.RiseHoldGuard is old_phase.RiseHoldGuard and new_phase.RiseHoldGuard is not old_phase.RiseHoldGuard)
    check('exact_59um_units', new_minimum.MAX_DOWNWARD_STEP_M == 59e-6 and new_minimum.MAX_DOWNWARD_STEP_M*1000 == .059)
    old_text = Path(old_minimum.__file__).read_text()
    expected = old_text.replace("VERSION='return_minimum_height_guard_v1_cpu'", "VERSION='return_minimum_height_guard_59um_v2_cpu'\nMAX_DOWNWARD_STEP_M=59e-6  # User-approved per-frame tolerance; 0.059 mm.").replace('if after-before<-1e-6:', 'if after-before<-MAX_DOWNWARD_STEP_M:')
    check('only_minimum_height_threshold_changed', (W/'minimum_guard.py').read_text() == expected)
    check('phase_switch_order_and_rules_unchanged', (W/'rise_hold_guard.py').read_text() == Path(old_phase.__file__).read_text().replace("VERSION='return_rise_then_high_hold_v1_cpu'", "VERSION='return_rise_then_high_hold_59um_v2_cpu'"))
    base = np.array([[0.,0.,0.], [0.,0.,1.]])
    def predicate(current, previous=base, gap=.3, inherited=.3):
        return outcome(lambda: new_minimum.minimum_rise(current, previous, gap, inherited))
    for label, drop in [('zero', 0.), ('0p9um', .9e-6), ('1um', 1e-6), ('recorded_1p066um', 1.065902862e-6), ('58p999um', 58.999e-6), ('exact59um', 59e-6)]:
        check(label+'_allowed', predicate(base-[0,0,drop])[0] == 'pass')
    for label, drop in [('above59um', np.nextafter(59e-6, np.inf)), ('60um', 60e-6)]:
        check(label+'_rejected', predicate(base-[0,0,drop]) == ('reject', 'return_rise_minimum_bucket_height_descended'))
    check('rising_lowest_with_descending_top_passes', predicate(base+[[0,0,.01],[0,0,-.01]])[0] == 'pass')
    check('gap_boundary_unchanged', predicate(base, gap=.3-1e-8)[0] == 'pass' and predicate(base, gap=np.nextafter(.3-1e-8, -np.inf)) == ('reject', 'return_rise_reduced_inherited_gap'))
    check('59um_does_not_waive_inherited_gap', predicate(base-[0,0,20e-6], gap=.299) == ('reject', 'return_rise_reduced_inherited_gap'))
    for label, points in [('nan', np.full((2,3),np.nan)), ('infinite',np.full((2,3),np.inf)), ('empty',np.empty((0,3))), ('shape',np.zeros((2,2)))]:
        check(label+'_rejected', predicate(points) == ('reject', 'minimum_rise_input_contract'))
    report = m.read(D/'report.json')
    check('original_run_stays_failed', report['status'] == 'failed' and report['failure'] == "ValueError('return_rise_minimum_bucket_height_descended')")
    closure = m.read(ORIGINAL/'closure.json')
    check('original_budget_stays_closed', closure['status'] == 'closed' and closure['no_more_physics_or_training'])
    states = m.read(D/'display/states.json')
    begin = m.read(D/'first-returned.control.json')['frame']
    steps = m.read(D/'scoop-2-plan/report.json')['segments'][0]['frames']
    check('cached_rise_scope_exact', len(states) == report['frames']+1 and report['frames']-begin == steps == 2151)
    runtime = m.CpuRuntime(D/'first-returned.npz', out/'runtime', enable_truck=True, freeze_passive_jp=True, bucket_motion_consistency=True)
    runtime.bridge = SimpleNamespace(state=states[begin])
    with np.load(D/'audit-derived/scoop-2-terrain.npz') as z:
        terrain = {k:z[k] for k in z.files}
    with np.load(D/'scoop-2-plan/approach.npz') as z:
        goal = z['qpos'][steps].copy()
    old = old_phase.RiseHoldGuard(runtime, terrain, goal, steps)
    new = new_phase.RiseHoldGuard(runtime, terrain, goal, steps)
    check('candidate_phase_uses_59um_class', type(new.rising) is new_minimum.MinimumHeightClearance)
    differences = []
    minimum_step = float('inf')
    for i in range(begin+1, len(states)):
        q, prev = states[i]['mechanism_qpos'], states[i-1]['mechanism_qpos']
        current = new.check(q, prev, physical=True)
        minimum_step = min(minimum_step, current['minimum_height_step_m'])
        prior = outcome(lambda: old.check(q, prev, physical=True))
        if prior[0] == 'reject':
            differences.append(dict(frame=i, old_error=prior[1], new_minimum_height_step_m=current['minimum_height_step_m']))
        else:
            assert prior[1] == current, ('non_threshold_result_changed', i)
    check('all2151_cached_rise_frames_pass_59um', new.mode == 'rise' and minimum_step >= -59e-6)
    check('only_former_final_failure_changes', len(differences) == 1 and differences[0]['frame'] == report['frames'] and differences[0]['old_error'] == 'return_rise_minimum_bucket_height_descended')
    check('early_transition_still_rejected', outcome(lambda: new.enter_hold(states[-1]['mechanism_qpos'], completed_nominal_steps=steps-1)) == ('reject', 'rise_path_not_completed') and new.mode == 'rise')
    admission = new.enter_hold(states[-1]['mechanism_qpos'], completed_nominal_steps=steps)
    check('completed_endpoint_hold_admission_passes', new.mode == 'hold' and admission['bucket_observed_terrain_m'] >= .32 and admission['endpoint_axis_error'] <= .015)
    bad = np.asarray(states[-1]['mechanism_qpos']).copy()
    bad[runtime.chain.axes[2]] = goal[runtime.chain.axes[2]]+.016
    check('hold_endpoint_drift_still_rejected', outcome(lambda: new.check(bad, physical=True)) == ('reject', 'original_endpoint_guard_during_hold'))
    a = old_minimum.MinimumHeightClearance(runtime, terrain, rise=False)
    b = new_minimum.MinimumHeightClearance(runtime, terrain, rise=False)
    samples = sorted({begin, begin+1, begin+steps//2, len(states)-1})
    check('nonrise_static_32cm_and_other_rules_identical', all(outcome(lambda:a.check(states[i]['mechanism_qpos'], physical=True)) == outcome(lambda:b.check(states[i]['mechanism_qpos'], physical=True)) for i in samples))
    check('no_native_entry_or_step_interface_added', not hasattr(new, 'step') and not hasattr(new, '__enter__') and not hasattr(new, 'reset'))
    check('all_frozen_inputs_and_old_code_unchanged', all(sha(p) == h for p,h in frozen_hashes.items()))
    result = dict(status='cpu_candidate_passed', checks=checks, check_count=len(checks),
        user_authorized_tolerance_um=59, tolerance_m=new_minimum.MAX_DOWNWARD_STEP_M,
        cached_rise_frames=steps, minimum_cached_height_step_m=minimum_step,
        changed_predicates=differences, hold_admission=admission,
        frozen_inputs_sha256=frozen_hashes,
        code_sha256={p.name:sha(p) for p in [W/'minimum_guard.py',W/'rise_hold_guard.py',Path(__file__)]},
        new_physics_steps=0, native_initializations=0, policy_predictions=0, training_updates=0,
        physical_admission=False, old_run_reclassified=False,
        limitation='Per-frame 59um downward tolerance only. Cache replay is not a new physical run, does not validate subsequent hold/approach/second dig, and does not reopen any budget.',
        wall_seconds=time.monotonic()-start)
    (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    signal.alarm(0)
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','frozen_inputs_sha256','code_sha256']}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
