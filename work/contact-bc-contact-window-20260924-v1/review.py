"""Read-only behavioral review of the bounded contact rollout."""
from pathlib import Path
import json
import numpy as np
from contact_bc_inference import sha
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import tool_pose,TIP
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RUN=ROOT/'runs/contact-bc-contact-window-20260924-v1'
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
report=json.loads((RUN/'report.json').read_text());actions=rows(RUN/'actions.ndjson');ph=rows(RUN/'physics/frames.ndjson')
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,chain=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
initial=actions[0]['before_state'];goal=report['selected_goal'];entry=np.array(goal['entry_xy_m']);tangent=np.array([-entry[1],entry[0]])/np.linalg.norm(entry)
def pose(state):
 tip,pitch=tool_pose(assembly,state['mechanism_qpos'],TIP)
 return dict(tip_world_m=tip.tolist(),bucket_pitch_deg=float(np.rad2deg(pitch)),lateral_error_to_target_m=float((tip[:2]-entry)@tangent),
  cylinder_axis_position_m=np.array(state['mechanism_qpos'])[chain.axes].tolist(),
  controller_axis_error_si=(np.array(state['controller_target'])-np.array(state['mechanism_qpos'])[chain.axes]).tolist())
series=[dict(frame=0,**pose(initial))]+[dict(frame=p['frame'],**pose(p['after_state'])) for p in ph]
bins=[]
for start in range(0,len(ph),60):
 block=ph[start:start+60];a=actions[start:start+len(block)];cmd=np.array([x['action_normalized'] for x in a]);applied=np.array([x['applied_command_si'] for x in a]);z=[series[start]['tip_world_m'][2],series[start+len(block)]['tip_world_m'][2]]
 bins.append(dict(first_frame=start+1,last_frame=start+len(block),mean_action_normalized=cmd.mean(axis=0).tolist(),
   requested_curl_positive_frames=int(np.sum(cmd[:,3]>.05)),requested_boom_lift_frames=int(np.sum(cmd[:,1]>.05)),
   applied_nonzero_frames=int(np.sum(np.any(np.abs(applied)>1e-8,axis=1))),
   projection_reason_frames=sum(x['command_projection_reason'] is not None for x in a),
   soil_contact_max=max(p['bucket_mpm_contact_samples'] for p in block),
   rock_contact_max=max(p['rock_bucket_contact_substeps'] for p in block),
   tip_vertical_delta_m=z[1]-z[0],end_pose=series[start+len(block)]))
cache=json.loads((RUN/'display/reconstruction.json').read_text());states=json.loads((RUN/'display/states.json').read_text())
checks=dict(prefix_120_exact=report.get('prefix_frames_verified')==120 and all(report.get('prefix_final_arrays_exact',{}).values()),
 source_unchanged=report['source_unchanged'],frames_sequential=[p['frame'] for p in ph]==list(range(1,len(ph)+1)),
 commands_match=all(a['applied_command_si']==p['applied_command_si'] for a,p in zip(actions,ph)),
 before_after_chain=all(actions[i]['before_state']==ph[i-1]['after_state'] for i in range(1,len(ph))),
 cache_states_match=len(states)==len(ph)+1 and states[0]==initial and all(states[i+1]==p['after_state'] for i,p in enumerate(ph)),
 all_hard_safe=all(p['hard_guard'] for p in ph),all_material_safe=all(p['hencky_material_guard']['safe'] for p in ph),
 all_sweep_safe=all(p['moving_bucket_sweep']['safe'] for p in ph))
failures=[dict(frame=p['frame'],failed=[k for k,v in p['hard_guard_checks'].items() if not v]) for p in ph if not p['hard_guard']]
result=dict(status=report['status'],failure=report.get('failure'),frames=len(ph),simulation_seconds=len(ph)/60,
 checks=checks,hard_failures=failures,first_pose=series[0],last_pose=series[-1],per_second=bins,
 max_absolute_lateral_error_m=max(abs(p['lateral_error_to_target_m']) for p in series),
 tip_height_min_m=min(p['tip_world_m'][2] for p in series),tip_height_max_m=max(p['tip_world_m'][2] for p in series),
 contact_frames=sum(p['bucket_mpm_contact_samples']>0 for p in ph),
 inference_median_seconds=float(np.median([a['timing']['roundtrip_seconds'] for a in actions])),
 physics_wall_seconds=sum(p['wall_seconds'] for p in ph),run_wall_seconds=report['wall_seconds'],
 cache=cache,full_excavation_qualified=False,fill_estimated=False,
 identities=[dict(path=str(p),sha256=sha(p)) for p in (RUN/'report.json',RUN/'actions.ndjson',RUN/'physics/frames.ndjson')])
(OUT/'review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
