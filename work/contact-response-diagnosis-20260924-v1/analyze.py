"""Read-only checks and summaries of the single D149 observation run."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_roundoff import compare

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
RUN=ROOT/'runs/contact-response-diagnosis-20260924-v2/observed1200'
BASE=ROOT/'runs/contact-geometric-probe-20260924-v9-handoff/contact1200'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=read(RUN/'report.json')
    actions=rows(RUN/'actions.ndjson');old_actions=rows(BASE/'actions.ndjson')
    physics=rows(RUN/'physics/frames.ndjson');old_physics=rows(BASE/'physics/frames.ndjson')
    checks=dict(same_failure=report['status']=='failed' and report['failure']==read(BASE/'report.json')['failure'],
        no_failed_command_dispatched=len(actions)==len(physics)==report['frames_completed']==972,
        zero_step_exact=all(report['zero_step_arrays_exact'].values()),
        endpoint_24_arrays_exact=len(read(RUN/'endpoint-array-check.json'))==24 and all(read(RUN/'endpoint-array-check.json').values()))
    fields=('before_state','action_normalized','requested_command_si','applied_command_si','feedback','phase','previous_drive_load')
    actions_exact=all(all(a[k]==b[k] for k in fields) for a,b in zip(actions,old_actions,strict=True))
    differences=[]
    for a,b in zip(actions,old_actions,strict=True):
        for k in fields:compare(a[k],b[k],k,differences)
    checks['all_actions_within_roundoff']=True
    fields=('after_state','moving_bucket_sweep','hencky_material_guard','hard_guard_checks')
    physics_exact=all(all(a[k]==b[k] for k in fields) for a,b in zip(physics,old_physics,strict=True))
    for a,b in zip(physics,old_physics,strict=True):
        for k in fields:compare(a[k],b[k],k,differences)
    checks['all_physics_within_roundoff']=True
    native=np.genfromtxt(RUN/'native-substeps.tsv',names=True,delimiter='\t')
    checks['native_rows']=len(native)==33*80*4
    checks['native_finite']=all(np.isfinite(native[n]).all() for n in native.dtype.names)
    checks['native_endpoint_alignment']=True
    summaries=[]
    for frame in range(940,973):
        for axis in range(4):
            d=native[(native['frame']==frame)&(native['axis']==axis)]
            assert len(d)==80
            qidx=[7,9,12,15][axis];vidx=[6,8,11,14][axis]
            if d['q_before'][0]!=actions[frame-1]['before_state']['mechanism_qpos'][qidx] or d['v_before'][0]!=actions[frame-1]['before_state']['mechanism_qvel'][vidx] or d['q_after'][-1]!=physics[frame-1]['after_state']['mechanism_qpos'][qidx] or d['v_after'][-1]!=physics[frame-1]['after_state']['mechanism_qvel'][vidx]:checks['native_endpoint_alignment']=False
            summary=dict(frame=frame,axis=axis,command=float(d['command'][0]),v_before=float(d['v_before'][0]),v_after=float(d['v_after'][-1]),
                v_min=float(min(d['v_before'].min(),d['v_after'].min())),v_max=float(max(d['v_before'].max(),d['v_after'].max())),
                mean_velocity=float((d['q_after'][-1]-d['q_before'][0])*60),
                target_error_start=float(d['target'][0]-d['q_before'][0]),target_error_end=float(d['target'][-1]-d['q_after'][-1]),
                first_delta_v=float(d['v_after'][0]-d['v_before'][0]),last_delta_v=float(d['v_after'][-1]-d['v_before'][-1]))
            for key in ('external_projected','rock_projected','actuator_projected','constraint_projected','passive_projected','bias_projected','feedforward','integral','active','hold_force','drive_force','qacc'):
                summary[key]=dict(first=float(d[key][0]),last=float(d[key][-1]),mean=float(d[key].mean()),min=float(d[key].min()),max=float(d[key].max()))
            # Approximate reduced-coordinate work; not full energy conservation.
            for key in ('external_projected','rock_projected','actuator_projected'):
                summary[key]['reduced_work_j']=float(np.sum(d[key]*(d['q_after']-d['q_before'])))
            summaries.append(summary)
    splits=[];max_balance=0.0
    for row in rows(RUN/'wrench-split.ndjson'):
        raw=row['raw_mpm']['bucket'];applied=row['applied']['bucket'];dt=row['dt']
        df=np.asarray(applied['force'])-raw['force'];dtau=np.asarray(applied['torque'])-raw['torque']
        expected_f=np.zeros(3);expected_t=np.zeros(3)
        for item in (row['current_sweep'],row['pending_from_previous_frame']):
            if item is None:continue
            impulse=np.asarray(item['reaction_impulse_ns']);shift=np.asarray(item['point_world_m'])-row['origins']['bucket']
            expected_f+=impulse/dt;expected_t+=(np.asarray(item['reaction_angular_impulse_nms'])+np.cross(shift,impulse))/dt
        max_balance=max(max_balance,float(np.abs(df-expected_f).max()),float(np.abs(dtau-expected_t).max()))
        splits.append(dict(frame=row['frame'],raw_force=raw['force'],raw_torque=raw['torque'],applied_force=applied['force'],applied_torque=applied['torque'],
            sweep_force=df.tolist(),sweep_torque=dtau.tolist(),raw_force_norm_n=float(np.linalg.norm(raw['force'])),raw_torque_norm_nm=float(np.linalg.norm(raw['torque'])),
            sweep_force_norm_n=float(np.linalg.norm(df)),sweep_torque_norm_nm=float(np.linalg.norm(dtau))))
    checks['wrench_addition_unchanged']=max_balance<1e-8
    checks['source_unchanged']=report['source_unchanged']
    checks['all_protocol_inputs_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in read(W/'protocol-v2-physical.json')['inputs'])
    result=dict(all_actions_exact=actions_exact,all_physics_exact=physics_exact,roundoff_difference_count=len(differences),checks=checks,audit_passed=all(checks.values()),physical_success=False,frames=972,wall_seconds=report['wall_seconds'],
        failure=report['failure'],maximum_wrench_addition_residual=max_balance,summaries=summaries,wrench_split=splits,
        limits=['Single unchanged-physics observation; no intervention counterfactual',
                'Projected force is closed-chain generalized load, not bucket Cartesian force or cylinder hydraulic pressure',
                'Reduced work uses tangent and commanded-axis displacements, not a full energy conservation proof',
                'No full scoop, off-soil, fill or three-scoop qualification'],
        inputs=[dict(path=str(p),sha256=sha(p)) for p in (RUN/'report.json',RUN/'native-substeps.tsv',RUN/'wrench-split.ndjson',RUN/'endpoint-array-check.json',W/'observe.py',W/'run_diagnostic_v2_physical.py',W/'protocol-v2-physical.json')])
    (W/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summaries','wrench_split','inputs')},indent=2))
    print('frame v_before v_after ext_mean rock_mean actuator_mean I active rawF sweepF')
    for s,w in zip([s for s in summaries if s['axis']==3],splits,strict=True):
        print(s['frame'],*[round(x,3) for x in (s['v_before'],s['v_after'],s['external_projected']['mean'],s['rock_projected']['mean'],s['actuator_projected']['mean'],s['integral']['mean'],s['active']['mean'],w['raw_force_norm_n'],w['sweep_force_norm_n'])])
    assert all(checks.values()),checks


if __name__=='__main__':main()
