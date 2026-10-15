"""Read existing state/command logs only; no simulator or policy imports."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
R=Path('/home/qunwang/wajueji');O=R/'work/motion-review-20261015-v1';O.mkdir(exist_ok=False)
A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
axes=[7,9,12,15];names=['swing','boom','stick','bucket'];out=[];evidence={};shake={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for root in [A,B]:
 states=read(root/'display/states.json');report=read(root/'report.json');geometry={}
 for line in (root/'geometry-actions.ndjson').open():
  x=json.loads(line);geometry.setdefault(x['phase'],[]).append(x)
 for p in report['phases']:
  name=p['name'];lo=p['start_frame'];hi=p['end_frame']
  if root==A and lo>=10504:continue
  q=np.array([x['mechanism_qpos'] for x in states[lo:hi+1]])[:,axes]
  origin=np.array([x['bucket_origin_world_m'] for x in states[lo:hi+1]])
  records=geometry.get(name,[])
  summary=dict(run=root.name,phase=name,frames=hi-lo,seconds=(hi-lo)/60,axis_start=q[0].tolist(),axis_end=q[-1].tolist(),axis_total_travel=np.sum(abs(np.diff(q,axis=0)),axis=0).tolist(),bucket_origin_start=origin[0].tolist(),bucket_origin_end=origin[-1].tolist())
  if records:
   qs=np.array([x['before_state']['mechanism_qpos'] for x in records])[:,axes];des=np.array([x['desired'] for x in records])[:,axes];cmd=np.array([x['command'] for x in records]);vel=np.array([x['velocity'] for x in records]);target=np.array([x['before_state']['controller_target'] for x in records]);pitch=np.unwrap(np.array([x['detail']['pitch_diagnostic_rad'] for x in records]));omg=np.array([x['before_state']['bucket_angular_velocity_world_rad_s'] for x in records]);lin=np.array([x['before_state']['bucket_linear_velocity_world_m_s'] for x in records])
   summary.update(nominal_peak_axis_error_si=np.max(abs(qs-des),axis=0).tolist(),command_max_abs_si=np.max(abs(cmd),axis=0).tolist(),actual_bucket_max_angular_speed_rad_s=float(np.max(np.linalg.norm(omg,axis=1))),actual_bucket_max_linear_speed_m_s=float(np.max(np.linalg.norm(lin,axis=1))),actual_pitch_range_deg=np.rad2deg([pitch.min(),pitch.max()]).tolist(),actual_pitch_travel_deg=float(np.rad2deg(np.sum(abs(np.diff(pitch))))),pitch_net_deg=float(np.rad2deg(pitch[-1]-pitch[0])))
   if name in ['CLEAR_PILE','ALIGN']:
    # Time-local linear-trend removal: quantifies wiggle, not tracking error.
    residual=[]
    for j in range(len(pitch)-60):
     window=pitch[j:j+61];residual.append(float(np.rad2deg(np.ptp(window-np.linspace(window[0],window[-1],61)))))
    summary['largest_1s_detrended_pitch_range_deg']=max(residual,default=0.)
    summary['extra_pitch_travel_over_net_deg']=summary['actual_pitch_travel_deg']-abs(summary['pitch_net_deg'])
    shake[name]=dict(summary=summary,first_command=records[0]['command'],first_reference_velocity=records[0]['velocity'],first_actual_velocity=[records[0]['before_state']['mechanism_qvel'][k] for k in [6,8,11,14]],first_actual_target_deflection_si=(target[0]-qs[0]).tolist())
    for j,x in enumerate(records):
     row=dict(phase=name,frame=x['global_frame'],workflow_s=(x['global_frame']-1)/60,pitch_deg=float(np.rad2deg(pitch[j])),bucket_angular_speed_rad_s=float(np.linalg.norm(omg[j])),bucket_linear_speed_m_s=float(np.linalg.norm(lin[j])))
     for k,key in enumerate(names):row.update({key+'_actual':float(qs[j,k]),key+'_reference':float(des[j,k]),key+'_command':float(cmd[j,k]),key+'_reference_velocity':float(vel[j,k])})
     out.append(row)
  evidence.setdefault(root.name,[]).append(summary)
 for file in ['report.json','geometry-actions.ndjson','display/states.json']:
  evidence.setdefault('source_sha256',{})[str(root/file)]=sha(root/file)
with (O/'loaded-transfer.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
result=dict(status='read_only_motion_review',phases=evidence,loaded_transfer=shake,physics_steps=0,inference_calls=0,training_updates=0,interpretation='Detrended pitch range and excess travel diagnose recorded wiggle only; no causal attribution or proposed-path physical validation.')
(O/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(loaded_transfer=shake,phase_seconds=[(x['phase'],round(x['seconds'],2)) for key in [A.name,B.name] for x in evidence[key]]),ensure_ascii=False,indent=2))
