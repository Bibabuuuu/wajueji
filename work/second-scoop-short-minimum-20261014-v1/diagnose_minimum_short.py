"""Measure the actual stop with saved states only; never initializes physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-minimum-20261014-v1';sys.path.insert(0,str(W))
import run_minimum_short as m
import numpy as np
from types import SimpleNamespace
from planning.cycle_geometry import tool_pose,TIP

def main():
 D=m.OUT;p=m.read(D/'report.json');states=m.read(D/'display/states.json');frames=[json.loads(s) for s in (D/'physics/frames.ndjson').read_text().splitlines()]
 r=m.CpuRuntime(D/'zero.npz',W/('diagnose-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 r.bridge=SimpleNamespace(state=states[3]);static=m.FixedRouteClearance(r)
 with np.load(D/'live-terrain.npz') as f:t={k:f[k] for k in f.files}
 checker=m.LiveClearance(r,t,rise=True)
 with np.load(D/'live-plan.npz') as f:qplan=f['qpos'].copy()
 rise_frames=m.read(D/'live-plan.json')['segments'][0]['frames'];nominal_end=3+rise_frames
 points=[];gaps=[];tips=[];axes=[];planned=[]
 for i,s in enumerate(states):
  q=np.array(s['mechanism_qpos']);pts,gap,_=checker.measure(q);points.append(pts);gaps.append(gap);tips.append(tool_pose(r.assembly,q,TIP)[0]);axes.append(q[r.chain.axes])
  if i>=3:planned.append(tool_pose(r.assembly,qplan[min(i-3,rise_frames)],TIP)[0])
 points=np.asarray(points);changes=np.diff(points[:,:,2],axis=0);last=changes[-1];idx=int(last.argmin());tips=np.asarray(tips)
 lastq=np.asarray(states[-1]['mechanism_qpos']);desired=qplan[min(len(states)-1-3,rise_frames)];prevq=np.asarray(states[-2]['mechanism_qpos'])
 static_last=static.check(lastq,physical=True)
 result=dict(nominal_rise_end_global_frame=nominal_end,stop_frames_into_tracking_hold=p['frames']-nominal_end,
   maximum_lowest_height_drop_from_rise_peak_m=float(points[3:,:,2].min(axis=1).max()-points[-1,:,2].min()),
   status='diagnosed_CPU_only',stop_frame=p['frames'],simulated_seconds=p['frames']/60,
   observed_failure=p['failure'],minimum_vertex_vertical_step_m=float(last.min()),minimum_vertex_index=idx,
   maximum_vertex_vertical_step_m=float(last.max()),mean_vertex_vertical_step_m=float(last.mean()),
   descending_sample_previous_and_stop_z_m=[float(points[-2,idx,2]),float(points[-1,idx,2])],
   bucket_minimum_previous_and_stop_z_m=[float(points[-2,:,2].min()),float(points[-1,:,2].min())],
   bucket_minimum_z_last_step_m=float(points[-1,:,2].min()-points[-2,:,2].min()),
   bucket_minimum_z_smallest_step_during_rise_m=float(np.diff(points[3:,:,2].min(axis=1)).min()),
   original_no_descent_threshold_m=-1e-6,tip_last_vertical_step_m=float(tips[-1,2]-tips[-2,2]),
   tip_vertical_rise_from_restore_m=float(tips[-1,2]-tips[0,2]),
   bucket_origin_last_vertical_step_m=float(states[-1]['bucket_origin_world_m'][2]-states[-2]['bucket_origin_world_m'][2]),
   inherited_observed_stock_gap_m=float(gaps[3]),stop_observed_stock_gap_m=float(gaps[-1]),previous_observed_stock_gap_m=float(gaps[-2]),minimum_observed_stock_gap_m=float(min(gaps)),
   last_nominal_vertex_vertical_step_m=float(np.min(checker.measure(qplan[min(len(states)-1-3,rise_frames)])[0][:,2]-checker.measure(qplan[min(len(states)-2-3,rise_frames)])[0][:,2])),
   stop_axis_error_si=(lastq[r.chain.axes]-desired[r.chain.axes]).tolist(),last_axis_displacement_si=(lastq[r.chain.axes]-prevq[r.chain.axes]).tolist(),
   stop_static_clearance=static_last,all_native_hard_guards=all(x['hard_guard'] for x in frames),
   all_material_and_sweep_guards=all(x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in frames),
   source_bucket_contact_samples=187,initial_and_stop_bucket_contact_samples=[frames[0]['bucket_mpm_contact_samples'],frames[-1]['bucket_mpm_contact_samples']],
   physics_steps_added_by_diagnosis=0,model_policy_calls=0,training_updates=0,
   finding='Nominal rise completed; the retained reference controller issued tiny endpoint corrections during tracking hold. Lowest bucket height then fell beyond1um/frame, so the admitted guard correctly stopped the run. No original hard/material/sweep guard failed. No threshold changed or physical retry.',
   boundary='Geometric source/stop gap and aggregate contact count do not establish force-based adhesion/ground ownership. No second approach or residual policy execution occurred.')
 m.write(D/'stop-diagnosis.json',result)
 np.savez(D/'stop-trace.npz',tip=tips,observed_stock_gap=np.asarray(gaps),minimum_vertex_step=np.min(changes,axis=1),axis_position=np.asarray(axes),nominal_tip=np.asarray(planned))
 print(m.encode(result),flush=True)
if __name__=='__main__':main()
