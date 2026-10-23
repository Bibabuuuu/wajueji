"""Independent saved-log audit; no physical runtime or GPU initialization."""
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/home/qunwang/wajueji')
OUT=ROOT/'runs'/sys.argv[1]
OLD=ROOT/'runs/motion-physical-short-20261015-v2/loaded'
report=json.loads((OUT/'report.json').read_text())
checks={}
rows=[]
with (OUT/'substeps.csv').open() as f:
    reader=csv.DictReader(f);names=reader.fieldnames
    rows=np.array([[float(r[k]) for k in names] for r in reader])
d={k:rows[:,i] for i,k in enumerate(names)}
bucket=d['axis']==3
b={k:v[bucket] for k,v in d.items()}
checks['finite_substeps']=bool(np.all(np.isfinite(rows)))
checks['row_count']=len(rows)==report['physics_steps']*80*4
checks['native_timestep']=bool(np.max(abs(d['time_after']-d['time_before']-1/4800))<1e-10)
checks['zero_active_bucket_commands']=bool(np.all(b['command']==0))
checks['dissipative_feedback']=bool(np.max(b['damping_effort']*b['v_before'])<=1e-9)
checks['damping_effort_cap']=bool(np.max(abs(b['damping_effort']))<=120000)
checks['integral_effort_cap']=bool(np.max(abs(b['integral']))<=120000)
checks['hold_gain_unchanged']=bool(np.all(b['hold_kp']==3e6))
checks['drive_gain_unchanged']=bool(np.all(b['drive_kv']==1e5))
ramp=.05
s=np.clip((b['time_before']-report['activation_clock'])/ramp,0,1);blend=s*s*(3-2*s)
expected_damping=blend*np.maximum(0,2*np.sqrt(b['mass_bucket']*b['hold_kp'])-b['drive_kv'])
checks['inertia_damping_formula']=bool(np.max(abs(expected_damping-b['extra_damping']))<1e-6)
checks['damping_force_formula']=bool(np.max(abs(np.clip(-expected_damping*b['v_before'],-120000,120000)-b['damping_effort']))<1e-6)
extra=np.where(bucket,d['damping_effort'],0)
force_residual=d['hold_kp']*(d['target']-d['q_before'])+d['ff']+d['integral']+extra-d['hold_effort']
checks['actuator_force_equation']=bool(np.max(abs(force_residual))<1e-6)
checks['drive_force_equation']=bool(np.max(abs(d['drive_kv']*(d['target_velocity']-d['v_before'])-d['drive_effort']))<1e-6)
checks['feedforward_components']=bool(np.max(abs(d['ff']-d['gravity_ff']-d['accel_ff']-d['velocity_ff']))<1e-6)
active=blend[1:]>0
previous=b['integral'][:-1]
expected=previous+blend[1:]*np.clip((np.clip(-b['external'][1:],-120000,120000)-previous)/.25,-30000,30000)/4800
checks['loaded_support_recurrence']=bool(np.max(abs(b['integral'][1:][active]-expected[active]))<1e-6)
checks['loaded_support_slew']=bool(np.max(abs(np.diff(b['integral'])[active]))<=30000/4800+1e-6)
checks['prefix_exact']=all(x['state_exact'] and x['particles_exact'] for x in report['prefix_checks']) and len(report['prefix_checks'])==3
states=json.loads((OUT/'states.json').read_text())
checks['state_count']=len(states)==report['physics_steps']+1
old_states=json.loads((OLD/'display/states.json').read_text())
checks['native_targets_unchanged']=max(np.max(abs(np.asarray(a['controller_target'])-np.asarray(baseline['controller_target']))) for a,baseline in zip(states,old_states,strict=True))<1e-12
commands=[json.loads(x)['command'] for x in (OLD/'dispatch.ndjson').read_text().splitlines()]
guards=0;physical_max_error=0
for line in (OUT/'physics/frames.ndjson').open():
    row=json.loads(line)
    assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']
    assert row['after_state']==states[guards+1]
    assert row['applied_command_si']==commands[guards]
    guards+=1
checks['all_physics_guards']=guards==report['physics_steps']
checks['no_inference_or_training']=report['inference_calls']==report['training_updates']==0
checks['one_native_instance']=report['native_initializations']==1
values=dict(max_abs_force_equation_error_N=float(np.max(abs(force_residual))),max_added_damping_effort_N=float(np.max(abs(b['damping_effort']))),max_support_rate_N_s=float(np.max(abs(np.diff(b['integral'])[active]))*4800),bucket_speed_rms_m_s=float(np.sqrt(np.mean(b['v_before']**2))),peak_bucket_speed_m_s=float(np.max(abs(b['v_before']))),native_substeps=len(b['time_before']),csv_rows=len(rows))
errors=json.loads((OUT/'pitch-errors.json').read_text())
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
sys.path[:0]=[str(kin),str(ROOT)]
assembly=importlib.import_module('scene_assembly').Assembler(kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
importlib.import_module('fast_kinematics').install(assembly)
from planning.cycle_geometry import tool_pose,MOUTH
actions={x['frame']:x for x in map(json.loads,(OLD/'actions.ndjson').read_text().splitlines())}
pitch_recompute=[];axis_errors=[]
for x in errors:
    q=np.asarray(states[x['frame']-1]['mechanism_qpos']);desired=np.asarray(actions[x['frame']]['desired'])
    actual=tool_pose(assembly,q,MOUTH)[1];target=tool_pose(assembly,desired,MOUTH)[1]
    pitch_recompute.append(abs(float(np.rad2deg(np.arctan2(np.sin(actual-target),np.cos(actual-target))))-x['pitch_error_deg']))
    axis_errors.append(float(np.max(abs(q[[7,9,12,15]]-desired[[7,9,12,15]]))))
checks['pitch_recomputed_from_states']=max(pitch_recompute)<1e-10
checks['tracking25mm']=max(axis_errors)<=.025
values['max_axis_reference_error_m_or_rad']=max(axis_errors)
if report['physics_steps']>=767:
    q=np.asarray(states[767]['mechanism_qpos']);desired=np.asarray(actions[767]['desired'])
    values['lift_endpoint_error_m_or_rad']=float(np.max(abs(q[[7,9,12,15]]-desired[[7,9,12,15]])))
    checks['lift_endpoint15mm']=values['lift_endpoint_error_m_or_rad']<=.015
for phase in ['CLEAR_PILE','ALIGN']:
    e=[x for x in errors if x['phase']==phase]
    if e:
        a=np.array([x['pitch_error_deg'] for x in e]);base=np.array([x['old_pitch_error_deg'] for x in e])
        values[phase]=dict(frames=len(e),peak_deg=float(np.max(abs(a))),rms_deg=float(np.sqrt(np.mean(a*a))),baseline_peak_deg=float(np.max(abs(base))),baseline_rms_deg=float(np.sqrt(np.mean(base*base))),last_deg=float(a[-1]))
        if phase=='CLEAR_PILE':
            checks['full_lift_peak_reduction_40pct']=bool(np.max(abs(a))<=.6*np.max(abs(base)))
            checks['full_lift_rms_reduction_40pct']=bool(np.mean(a*a)<=.36*np.mean(base*base))
            if len(a)>=700:
                checks['late_lift_error_below_one_degree']=bool(np.max(abs(a[120:]))<=1)
                values['late_lift_peak_deg']=float(np.max(abs(a[120:])))
checks['complete_887']=report['status']=='loaded_887_frame_damping_completed' and report['physics_steps']==887
checks={k:bool(v) for k,v in checks.items()}
audit=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,metrics=values,scope='Saved native force equations and physical safety, same frozen commands; closed-loop command recomputation and whole swing not established by this audit.',physics_steps_added=0)
(OUT/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
