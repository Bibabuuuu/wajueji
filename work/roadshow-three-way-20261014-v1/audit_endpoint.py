"""Read-only audit of CPU candidate outputs; no native runtime or replay."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
out=WORK/'transport-endpoint-preflight-v2'
report=json.loads((out/'report.json').read_text());checks={}
checks['cpu_report_passed']=report['status']=='endpoint_nominal_cpu_passed'
checks['no_physical_or_training']=report['new_physical_frames']==report['training_updates']==0
checks['not_claimed_native_restore']=report['native_restore_tested'] is False
checks['not_claimed_actual_transport']=report['physical_transport_tested'] is False
for name in ['better_manual','frozen_residual']:
    case=report['sources'][name];plan=case['plan']
    checks[name+'_all_original_cpu_checks']=all(case['checks'].values())
    checks[name+'_all_stages']= [x['name'] for x in plan['stages']]==['CLEAR_PILE','ALIGN','DUMP']
    checks[name+'_source_sha']=hashlib.sha256(Path(plan['source']).read_bytes()).hexdigest()==json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources'][name]['sha256']
    with np.load(plan['source']) as data:cylinder=float(data['mechanism_qpos'][15])
    with np.load(out/(name+'-plan/plan.npz')) as archive:
        checks[name+'_finite_paths']=all(np.isfinite(archive[k]).all() for k in archive.files)
        checks[name+'_held_cylinder']=all(np.max(abs(archive[stage+'_qpos'][:,15]-cylinder))<1e-8 for stage in ['CLEAR_PILE','ALIGN'])
        checks[name+'_loaded_cylinder_velocity_zero']=all(np.max(abs(archive[stage+'_velocity'][:,3]))<1e-12 for stage in ['CLEAR_PILE','ALIGN'])
        checks[name+'_nominal_command_count']=case['nominal_commands']==sum(len(archive[s+'_velocity'])+30+(180 if s=='DUMP' else 0) for s in ['CLEAR_PILE','ALIGN','DUMP'])
    checks[name+'_original_waypoint_xy']=plan['old_waypoint_xy_unchanged'] is True
    checks[name+'_explicit_five_cm_lift']=plan['lift_height_m']==3.35 and plan['endpoint_solver_calls'][0]['original_target_m'][2]==3.3
    checks[name+'_closure_precision']=plan['endpoint_solver_calls'][0]['closure_m']<1e-8
    checks[name+'_unchanged_thresholds']=plan['mechanical_bounds_unchanged'] and plan['clearance_and_closure_thresholds_unchanged']
    checks[name+'_four_cm_truck_clearance']=all(s['minimum_clearance']['bucket_truck_m']>=.04 for s in plan['stages'])
model=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
checks['frozen_model_unchanged']=hashlib.sha256(model.read_bytes()).hexdigest()=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),
    new_physical_frames=0,training_updates=0,scope='CPU path artifact audit only; no actual loaded tracking or net delivered volume result')
(out/'read-only-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert all(checks.values())
