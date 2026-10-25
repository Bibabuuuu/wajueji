#!/usr/bin/env python3
"""Read-only paired-run checks; write only a new comparison report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    report=json.loads(path.read_text())
    rows=[json.loads(line) for line in (path.parent/'physics.ndjson').read_text().splitlines()]
    return report,rows


def compare(baseline_path,candidate_path):
    base,br=read_run(baseline_path)
    candidate,cr=read_run(candidate_path)
    for report,rows in ((base,br),(candidate,cr)):
        if (report['status']!='practice_completed' or report['mode']!='combined_smoke'
                or len(rows)!=270 or report['frames']!=270
                or not report['checks'] or not all(report['checks'].values())
                or not all(row['motion_guard'] and row['strict'] for row in rows)):
            raise ValueError('comparison_requires_complete_qualified_windows')
    if base['protocol']['rigid_internal_hz']!=4800 or candidate['protocol']['rigid_internal_hz'] not in (2400,3600):
        raise ValueError('comparison_frequency_pair')
    for key in ('source_checkpoint','input_bindings','runner_sha256','device','acquisition_domain','scope'):
        if base[key]!=candidate[key]:
            raise ValueError('comparison_binding_mismatch:'+key)
    ignored={'rigid_internal_hz','rigid_substeps_per_mpm_frame','rigid_substeps_per_bridge_half','rigid_frequency_candidate_only'}
    if {k:v for k,v in base['protocol'].items() if k not in ignored}!={k:v for k,v in candidate['protocol'].items() if k not in ignored}:
        raise ValueError('comparison_protocol_mismatch')
    if [(r['phase'],r['command_si']) for r in br]!=[(r['phase'],r['command_si']) for r in cr]:
        raise ValueError('comparison_actions_mismatch')
    axis=np.max(np.abs(np.array([r['axis_positions'] for r in br])-np.array([r['axis_positions'] for r in cr])),axis=0)
    clearance=max(abs(b['bucket_vertical_clearance_m']-c['bucket_vertical_clearance_m']) for b,c in zip(br,cr,strict=True))
    max_rock=max_particle=0.
    snapshots=[]
    if len(base['diagnostic_snapshots'])!=5 or len(candidate['diagnostic_snapshots'])!=5:
        raise ValueError('comparison_requires_five_diagnostic_snapshots')
    for b,c in zip(base['diagnostic_snapshots'],candidate['diagnostic_snapshots'],strict=True):
        for info in (b,c):
            path=Path(info['path'])
            if sha(path)!=info['sha256'] or sha(Path(str(path)+'.json'))!=info['metadata_sha256']:
                raise ValueError('comparison_snapshot_hash')
        with np.load(b['path'],allow_pickle=False) as ba,np.load(c['path'],allow_pickle=False) as ca:
            if not np.array_equal(ba['simulation_time_s'],ca['simulation_time_s']):
                raise ValueError('comparison_snapshot_time')
            rock=float(np.linalg.norm(ba['rigid_qpos'][:,:3]-ca['rigid_qpos'][:,:3],axis=1).max())
            particle=float(np.linalg.norm(ba['particle_q']-ca['particle_q'],axis=1).max())
            max_rock=max(max_rock,rock);max_particle=max(max_particle,particle)
            snapshots.append(dict(baseline=b['sha256'],candidate=c['sha256'],max_rock_position_difference_m=rock,max_particle_position_difference_m=particle))
    checks=dict(same_input_actions_and_protocol=True,all_source_runs_qualified=True,
        axis_position_difference_within_limits=bool(np.all(axis<=np.array([.002,.001,.001,.001]))),
        clearance_difference_within_2mm=clearance<=.002,rock_position_difference_within_1mm=max_rock<=.001)
    def timing(report,rows):
        result=dict(total_wall_seconds=report['timing']['total_wall_seconds'],simulation_seconds=len(rows)/60)
        for key in ('wall_seconds','projection_wall_seconds','mpm_wall_seconds','mujoco_wall_seconds','assignment_wall_seconds'):
            result[key]=sum(r[key] for r in rows)
        result['physics_plus_prediction_seconds']=result['wall_seconds']+result['projection_wall_seconds']
        result['wall_per_sim_second']=result['physics_plus_prediction_seconds']/result['simulation_seconds']
        steady=rows[15:]
        result['excluding_first_025s_wall_per_sim_second']=sum(r['wall_seconds']+r['projection_wall_seconds'] for r in steady)/(len(steady)/60)
        return result
    bt,ct=timing(base,br),timing(candidate,cr)
    return dict(status='rigid_frequency_comparison_passed' if all(checks.values()) else 'rigid_frequency_comparison_rejected',
        baseline_report=str(baseline_path),baseline_report_sha256=sha(baseline_path),
        candidate_report=str(candidate_path),candidate_report_sha256=sha(candidate_path),
        candidate_rigid_hz=candidate['protocol']['rigid_internal_hz'],checks=checks,
        max_axis_position_difference_si=axis.tolist(),max_clearance_difference_m=clearance,
        max_snapshot_rock_position_difference_m=max_rock,max_snapshot_particle_position_difference_m=max_particle,
        snapshots=snapshots,baseline_timing=bt,candidate_timing=ct,
        physics_plus_prediction_speedup=bt['physics_plus_prediction_seconds']/ct['physics_plus_prediction_seconds'],
        time_reduction_fraction=1-ct['physics_plus_prediction_seconds']/bt['physics_plus_prediction_seconds'],
        scope=dict(noncontact_short_test_only=True,default_changed=False,training_eligible=False))


def main():
    parser=argparse.ArgumentParser()
    for name in ('baseline','candidate','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.baseline.resolve(),args.candidate.resolve())
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
    return 0 if all(result['checks'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
