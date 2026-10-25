"""Audit late-curl dispatch and measured body motion from the new physical run."""
import json,sys
from pathlib import Path
import numpy as np
from itertools import product
from contact_bc_ground_guard import BucketGroundGuard
from contact_bc_inference import sha
from layered_excavation_approach_planner import _load_kinematics
W=Path(__file__).resolve().parent;ROOT=W.parents[1];run=ROOT/sys.argv[1]
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
actions=rows(run/'actions.ndjson');physical=rows(run/'physics/frames.ndjson');report=read(run/'report.json')
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
guard=BucketGroundGuard(a,np.asarray(actions[0]['before_state']['mechanism_qpos']))
corners=np.asarray(list(product(*zip(guard.vertices.min(axis=0),guard.vertices.max(axis=0)))))
def heights(q):
 pose=a.assemble(np.asarray(q),include_meshes=False)[1]['bucket'];return corners@pose[2,:3]+pose[2,3]
active=[(e,p) for e,p in zip(actions,physical) if e['feedback']['no_down_latched']]
checks=dict(active=bool(active),all_valid_physical=len(actions)==len(physical),no_boom_down=all(e['applied_command_si'][1]>=-1e-10 for e,p in active),predicted_body_no_down=all(e['feedback']['no_down_minimum_velocity_m_s']>=-1e-7 and e['feedback']['no_down_preview_minimum_delta_m']>=-1e-6 for e,p in active))
old_run=ROOT/'runs/contact-geometric-probe-20260924-v6/scoop2400'
old_actions=rows(old_run/'actions.ndjson');old_physical=rows(old_run/'physics/frames.ndjson')
prefix=next((e['frame']-1 for e in actions if e['feedback']['no_down_latched']),len(actions))
checks['pre_trigger_commands_unchanged']=all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(actions[:prefix],old_actions[:prefix]))
checks['pre_trigger_actual_states_unchanged']=all(x['after_state']==y['after_state'] for x,y in zip(physical[:prefix],old_physical[:prefix]))
checks['latch_not_released']=not active or all(e['feedback']['no_down_latched'] for e in actions[active[0][0]['frame']-1:])
result=dict(pre_trigger_frames=prefix,run=str(run),status=report['status'],failure=report.get('failure'),frames=len(physical),final_phase=report.get('final_phase'),events=report['events'],checks=checks,command_guard_passed=all(checks.values()),physics_steps_added=0,source=report['source'],policy=report['policy'],outputs=[dict(path=str(run/x),sha256=sha(run/x)) for x in ('report.json','actions.ndjson','physics/frames.ndjson')])
if active:
 first=active[0][0];h0=heights(first['before_state']['mechanism_qpos']);delta=[];from_trigger=[]
 root=np.asarray(first['feedback']['soil_plan']['root_local_m']);pose0=a.assemble(np.asarray(first['before_state']['mechanism_qpos']),include_meshes=False)[1]['bucket']
 cad0=guard.vertices@pose0[2,:3]+pose0[2,3];root0=float(root@pose0[2,:3]+pose0[2,3]);cad_changes=[];root_changes=[]
 for e,p in active:
  pose=a.assemble(np.asarray(p['after_state']['mechanism_qpos']),include_meshes=False)[1]['bucket'];cad_changes.append(float(np.min(guard.vertices@pose[2,:3]+pose[2,3]-cad0)));root_changes.append(float(root@pose[2,:3]+pose[2,3]-root0))
  after=heights(p['after_state']['mechanism_qpos']);delta.append(after-heights(e['before_state']['mechanism_qpos']));from_trigger.append(after-h0)
 old_root_changes=[]
 for previous in old_physical[first['frame']-1:len(physical)]:
  op=a.assemble(np.asarray(previous['after_state']['mechanism_qpos']),include_meshes=False)[1]['bucket'];old_root_changes.append(float(root@op[2,:3]+op[2,3]-root0))
 result['v6_minimum_root_change_from_same_trigger_m']=min(old_root_changes)
 result['comparison_last_frame']=min(len(physical),len(old_physical))
 delta=np.asarray(delta);from_trigger=np.asarray(from_trigger)
 result.update(minimum_actual_cad_vertex_change_from_trigger_m=min(cad_changes),minimum_root_change_from_trigger_m=min(root_changes),worst_cad_drop_frame=active[int(np.argmin(cad_changes))][0]['frame'],worst_root_drop_frame=active[int(np.argmin(root_changes))][0]['frame'],trigger_frame=first['frame'],trigger_fraction=first['feedback']['closure_fraction'],active_frames=len(active),zero_command_frames=int(sum(np.max(abs(np.asarray(e['applied_command_si'])))<1e-8 for e,p in active)),curl_command_frames=sum(e['applied_command_si'][3]>.001 for e,p in active),minimum_actual_corner_step_m=float(delta.min()),minimum_actual_corner_change_from_trigger_m=float(from_trigger.min()),frames_with_corner_step_below_minus_1mm=int(np.sum(np.min(delta,axis=1)<-.001)),final_corner_changes_m=from_trigger[-1].tolist(),maximum_bucket_cylinder_m=max(p['after_state']['mechanism_qpos'][15] for e,p in active),minimum_boom_cylinder_step_m=min(p['after_state']['mechanism_qpos'][9]-e['before_state']['mechanism_qpos'][9] for e,p in active))
result['measured_whole_body_no_down_passed']=bool(active and result['minimum_actual_cad_vertex_change_from_trigger_m']>=-1e-6)
result['scope']='Command constraint and measured rigid body response only. Sampled box-corner motion is not continuous collision, soil preservation, end-stop braking or fill qualification.'
(W/(run.name+'-late-curl-review.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
