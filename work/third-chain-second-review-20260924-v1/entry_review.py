"""Read-only marker and actual entry-reference comparison; no label rewriting."""
import json
from pathlib import Path
import numpy as np
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP,tool_pose
ROOT=Path('/home/qunwang/wajueji');HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-114908-1452582'
r=json.loads((RUN/'report.json').read_text());rows=[json.loads(s) for s in (RUN/'physics.ndjson').read_text().splitlines()]
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
xy=np.array(r['selected_cut']['entry_xy_m']);radial=xy/np.linalg.norm(xy);lateral=np.array([-radial[1],radial[0]])
first=next(i for i,row in enumerate(rows) if row['bucket_mpm_contact_samples']>0)
sustained=next(i for i in range(len(rows)-4) if all(row['bucket_mpm_contact_samples']>=10 for row in rows[i:i+5]))
marks=[('first_nonzero_sample',first),('first_five_frames_at_least_ten_samples',sustained)]
marks += [(e['label'],e['next_frame']-1) for e in r['phase_events'] if e['next_frame']<=len(rows)]
summary=[]
for label,i in marks:
 row=rows[i];p,pitch=tool_pose(assembly,row['before_state']['mechanism_qpos'],TIP)
 summary.append(dict(label=label,frame=row['frame'],center_reference_world_m=p.tolist(),
  lateral_from_planned_center_m=float((p[:2]-xy)@lateral),
  bucket_contact_samples=row['bucket_mpm_contact_samples'],pitch_rad=pitch))
result=dict(schema='operator_entry_adjustment_review_v1',run=str(RUN),planned_target=r['selected_cut'],
 marker_semantics='Fixed target for bucket width-center reference; visible points are a vertical column 0.05..0.65 m above target.',
 local_reference_m=TIP.tolist(),reference_is_exact_outer_tooth=False,
 user_annotation='User reports the scoop is complete but fill seems lower. No manual retargeting assertion.',
 observations=summary,automatic_actual_target_relabelled=False,
 training_note='Keep original selected target and operator adjustment separately; review task conditioning before training.',
 scope='Reference poses from recorded kinematics; contact-count heuristic is not an exact tooth-surface first intersection.')
(HERE/'entry-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
