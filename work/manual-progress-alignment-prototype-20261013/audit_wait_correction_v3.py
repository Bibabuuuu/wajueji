"""Read-only audit of the finished alignment gate, including accepted projection."""
import json
from pathlib import Path
import sys
ROOT=Path('/home/qunwang/wajueji')
sys.path.insert(0,str(ROOT/'src/v4'))
import numpy as np
from manual_trajectory_residual import compare_arrays
import hashlib


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    out=ROOT/'runs/manual-progress-alignment-20261013-v3'
    report=json.loads((out/'physical-review.json').read_text())
    assert report['status']=='passed'
    old=ROOT/'runs/manual-residual-sac-20261013/retiming-v2-gate/baseline/episode-0001'
    checks={}
    old_commands=[json.loads(x)['applied_si'] for x in (old/'actions.ndjson').read_text().splitlines()]
    episodes={}
    for result in report['episodes']:
        folder=Path(result['path']);label=result['label']
        rows=[json.loads(x) for x in (folder/'physics/frames.ndjson').read_text().splitlines()]
        actions=[json.loads(x) for x in (folder/'actions.ndjson').read_text().splitlines()]
        alignment=[json.loads(x) for x in (folder/'alignment.ndjson').read_text().splitlines()]
        checks[label+'_physical_row_count']=len(rows)==result['physical_frames']
        checks[label+'_all_original_hard_checks']=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in rows)
        checks[label+'_all_alignment_trusted']=all(x['trusted'] for x in alignment)
        checks[label+'_bounded_reference_lead']=all(x['reference_progress']-x['actual_progress']<=6+1e-9 for x in alignment)
        checks[label+'_actual_phase_monotonic']=all(a['actual_progress']<=b['actual_progress'] for a,b in zip(alignment,alignment[1:]))
        waiting=[x for x in actions if x['pace']==0 and x['hold_frame'] is None]
        checks[label+'_waiting_freezes_baseline']=all(not np.any(x['baseline_si']) for x in waiting)
        checks[label+'_waiting_correction_direction']=all(np.all(np.array(x['request_si'])*np.array(x['alignment']['wait_target_error_si'])>=-1e-15) for x in waiting)
        checks[label+'_waiting_correction_cap']=all(np.all(abs(np.array(x['request_si']))<=np.array([.12,.0225,.0225,.0675])*.1+1e-15) for x in waiting)
        checks[label+'_waiting_correction_no_target_overshoot']=all(np.all(abs(np.array(x['request_si']))/60<=abs(np.array(x['alignment']['wait_target_error_si']))+1e-15) for x in waiting)
        if label!='zero_full':
            correction=[x for x in waiting if x['frame']>1020 and np.any(x['request_si'])]
            checks['directed_correction_really_executed']=len(correction)>0
            checks['actual_progress_recovered']=result['last_alignment']['actual_progress']>result['correction_start_actual_progress']+10
            checks['reference_progress_recovered']=result['cursor']>result['correction_start_cursor']+10
            checks['stalled_point_full_arrays_exact']=all(result['old_stalled_point_arrays_exact'].values())
        checks[label+'_physics_source_bindings']=all(sha(ROOT/'src/v4'/p)==expected for p,expected in result['runtime_bindings'].items())
        for p,expected in report['source_code'].items():
            checks['code_'+p]=sha(ROOT/'src/v4'/p)==expected
        prefix=1170 if label=='zero_full' else 900
        checks[label+'_projection_reproduced']=all(np.array_equal(x['applied_si'],old_commands[i]) for i,x in enumerate(actions[:prefix]))
        loaded=[x for x in rows if x['frame']>900]
        episodes[label]=dict(status=result['status'],frames=result['physical_frames'],cursor=result['cursor'],
            wait_frames=result['alignment_wait_frames'],max_lead_frames=result['maximum_reference_lead_frames'],
            max_position_error_m=result['maximum_alignment_error_m'],
            bucket_contact_frames=sum(x['bucket_mpm_contact_samples']>0 for x in loaded),
            contact_sample_range=[min(x['bucket_mpm_contact_samples'] for x in loaded),max(x['bucket_mpm_contact_samples'] for x in loaded)],
            force_checks=result['checks'])
        if label!='zero_full':
            checks[label+'_really_loaded']=sum(x['bucket_mpm_contact_samples']>0 for x in loaded)>0 and result['checks']['loaded_force_frames']>0
    zero=out/'zero_full/episode-0001/endpoint.npz'
    with np.load(zero,allow_pickle=False) as current,np.load(old/'endpoint.npz',allow_pickle=False) as previous:
        exact,errors=compare_arrays(current,previous)
    checks['zero_endpoint_arrays_exact']=all(exact.values())
    current=json.loads(zero.with_suffix('.npz.json').read_text())
    previous=json.loads((old/'endpoint.npz.json').read_text())
    checks['zero_native_controller_exact']=current['controller_state']==previous['controller_state']
    checks['zero_retained_fill_unchanged']=report['episodes'][0]['retained_fill_fraction']==1.4538132234025687
    checks['source_unchanged']=all(x['source_unchanged'] for x in report['episodes'])
    audit=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
               endpoint_exact=exact,endpoint_errors=errors,episodes=episodes,
               scope='frozen-scene zero reproduction and two short loaded probes only',
               no_training=True,report_sha256=sha(out/'physical-review.json'))
    path=out/'independent-audit.json'
    assert not path.exists()
    path.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit))
    assert all(checks.values())
    receipt=dict(status='passed',training_admission=True,scope='new replay only; frozen scene; no learned improvement',
                 code={n:sha(ROOT/'src/v4'/n) for n in ['manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']},
                 audit=str(path),audit_sha256=sha(path),baseline_retained_fill_fraction=1.4538132234025687,
                 observation_size=221,waiting_strategy='freeze_phase_allow_directed_residual_v2')
    receipt_path=out/'training-admission.json';assert not receipt_path.exists()
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
