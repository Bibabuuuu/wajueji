"""Portable D-107 integration: unchanged 270-frame protocol, new bridge only."""
import json
from pathlib import Path
import traceback

from layered_47_optimized_bridge import optimized_bridge, parser
import run_layered_47_sparse_motion as motion


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = dict(status='optimized_motion_failed', checks={},
                  scope=dict(training_eligible=False, formal_configuration_changed=False))
    try:
        with optimized_bridge(args.mechanism_header_dir, output.with_suffix('.build')) as provenance:
            report = motion.run(args)
            report['optimization'] = provenance.copy()
            applied = report.get('restore', {}).get('runtime_build', {}).get('bridge', {}).get('optimized_inertia')
            report.setdefault('checks', {})['optimized_header_applied'] = bool(
                applied and applied['candidate_header_sha256'] == provenance['candidate_header_sha256'])
            report['status'] = ('optimized_motion_passed' if report['status']=='sparse_motion_passed'
                and report['checks']['optimized_header_applied'] else 'optimized_motion_failed')
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='optimized_motion_failed', failure=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
    report.update(schema='layered_47_optimized_motion_v1',
        integration_runner_sha256=motion.probe.bench.sha256(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    motion.probe.bench._write_json(output, report)
    return report


if __name__ == '__main__':
    args = parser().parse_args()
    report = run(args)
    code = 0 if report['status']=='optimized_motion_passed' else 1
    print('COPY_RESULT_BEGIN')
    print(json.dumps({k:report.get(k) for k in ('status','checks','timing','execution','failure',
        'rejection','optimization','source_checkpoint','diagnostic_snapshots')} | dict(
        runner_exit_code=code, report=str(Path(args.output).resolve()), report_exists=True,
        next_gate='review_optimized_motion_before_manual_practice_no_training'), indent=2))
    print('COPY_RESULT_END')
    raise SystemExit(code)
