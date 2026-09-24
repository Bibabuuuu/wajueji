"""CPU-only alignment of saved motion, commands, targets and contact loads."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/contact-second-finish-unload-20260925-v2'))
from controller import CarryFeedback,LIMITS
from layered_excavation_approach_planner import _load_kinematics
from contact_bc_ground_guard import BucketGroundGuard
run=ROOT/'runs/contact-second-finish-unload-20260925-v2/physical'
def rows(name):return [json.loads(x) for x in (run/name).read_text().splitlines()]
frames=rows('physics/frames.ndjson');acts=rows('actions.ndjson');halves=rows('physics/native-halves.ndjson');responses=rows('physics/response.ndjson')
states=[acts[0]['before_state']]+[x['after_state'] for x in frames]
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
q=np.array([s['mechanism_qpos'] for s in states]);axes=c.axes
fb=CarryFeedback(a,c,BucketGroundGuard(a,q[0]),q[0],np.array(acts[0]['command']),frame_end=2725,initial_soil_top_m=1.346285343170166)
center=np.array([fb.body_heights(x).mean() for x in q]);origin=np.array([s['bucket_origin_world_m'][2] for s in states]);peak=int(np.argmax(center));origin_peak=int(np.argmax(origin));command=np.array([x['command'] for x in acts]);targets=np.array([s['controller_target'] for s in states]);pos=q[:,axes];speed=np.diff(pos,axis=0)*60
forces=np.array([x['applied_bucket_mpm_force_world_n'] for x in halves]).reshape(-1,2,3).mean(axis=1)
loads=np.array([x['equivalent_drive_load'] for x in halves]).reshape(-1,2,4).mean(axis=1)
summary=dict(physics_steps_added=0,frames=len(frames),qpos_axes=list(map(int,axes)),center_peak_frame=peak,origin_peak_frame=origin_peak,
 center_peak_to_end_mm=float((center[peak]-center[-1])*1000),origin_peak_to_end_mm=float((origin[origin_peak]-origin[-1])*1000),
 center_initial_to_end_mm=float((center[-1]-center[0])*1000),origin_initial_to_end_mm=float((origin[-1]-origin[0])*1000),
 decline_duration_s=(len(frames)-peak)/60,maximum_single_frame_center_drop_mm=float(-min(np.diff(center))*1000),
 all_hard_guards_passed=all(x['hard_guard'] for x in frames),all_command_projection_none=all(x['detail']['projection_reason'] is None for x in acts),
 all_desired_vertical_velocity_m_s=sorted(set(x['detail']['desired_task_velocity'][2] for x in acts)),
 all_phase_finish_curl=all(x['stage']=='FINISH_CURL' for x in acts),samples=[])
for i in sorted(set([1,60,120,180,240,peak,origin_peak,280,300,320,343])):
 before=q[i-1];jac,ground,body=fb.linearize(before,include_body=True)
 st=states[i];co=np.array(responses[i-1]['coefficients']['stiffness']);damp=np.array(responses[i-1]['coefficients']['drive_damping'])
 summary['samples'].append(dict(frame=i,center_z_m=float(center[i]),origin_z_m=float(origin[i]),
 command=command[i-1].tolist(),actual_axis_speed=speed[i-1].tolist(),axis_position=pos[i].tolist(),
 target_minus_actual_rot_mrad_cyl_mm=((targets[i]-pos[i])*1000).tolist(),
 predicted_center_velocity_m_s=float(np.mean(body@(command[i-1]/LIMITS))),
 native_target_velocity=st['controller_target_velocity'],
 center_velocity_using_native_target_m_s=float(np.mean(body@(np.array(st['controller_target_velocity'])/LIMITS))),
 center_velocity_axis_contributions_actual_m_s=(body.mean(axis=0)/LIMITS*speed[i-1]).tolist(),
 actual_center_velocity_m_s=float((center[i]-center[i-1])*60),
 free_twist_z_m_s=responses[i-1]['free_twist'][2],
 approximate_stiffness_error_force=(co*(targets[i]-pos[i])).tolist(),
 mpm_force_n=forces[i-1].tolist(),equivalent_drive_load=loads[i-1].tolist(),
 bucket_integral_effort=st['bucket_integral_effort'],rock_bucket_substeps=frames[i-1]['rock_bucket_contact_substeps']))
np.savez(W/'series.npz',center_z_m=center,origin_z_m=origin,command=command,axis_position=pos,controller_target=targets,actual_axis_speed=speed,mpm_force_n=forces,equivalent_load=loads)
summary['first_bucket_target_upper_frame']=next((x['frame'] for x in frames if x['after_state']['controller_target'][3]>=.852-1e-9),None)
summary['first_bucket_native_zero_velocity_frame']=next((x['frame'] for x in frames if x['frame']>=250 and x['after_state']['controller_target_velocity'][3]==0),None)
summary['inputs']={str(run/p):hashlib.sha256((run/p).read_bytes()).hexdigest() for p in ['physics/frames.ndjson','actions.ndjson','physics/native-halves.ndjson','physics/response.ndjson']}
(W/'report.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
