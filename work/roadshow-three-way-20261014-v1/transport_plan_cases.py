"""Independent CPU reachability checks for both saved sources; fixed old route."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path.insert(0,str(WORK))
from transport_cpu_candidate_v3 import CpuTransportRuntime,encode
import plan
from planning.cycle_geometry import MOUTH,tool_pose,solve_pose

out=WORK/'transport-fixed-route-reachability-v1';out.mkdir(exist_ok=False)
started=time.monotonic();sources=json.loads((WORK/'offline-boundary-v2/transport-source-preflight.json').read_text())['sources']
report=dict(status='checking',sources={},new_physical_frames=0,training_updates=0,parameters_changed=False)
for name in ['better_manual','frozen_residual']:
    source=sources[name];r=CpuTransportRuntime(source['source'],out/name,source['sha256'])
    q=r.arrays['mechanism_qpos'].copy();mouth,pitch=tool_pose(r.assembly,q,MOUTH)
    radial=mouth[:2]/np.linalg.norm(mouth[:2]);seed=plan.project_passive(r.chain,q)
    item=dict(source=source['source'],sha256=source['sha256'],actual_mouth=mouth.tolist(),
        actual_pitch_rad=float(pitch),actual_cylinder_m=float(q[15]),nominal_source_projection_max_m=float(np.max(abs(seed-q))),
        target_mouth=[*(3.9*radial).tolist(),3.3],target_pitch_rad=float(pitch))
    try:
        target=solve_pose(r.assembly,r.chain,seed,np.r_[3.9*radial,3.3],pitch,MOUTH)
        item.update(status='first_lift_pose_passed',goal=target.tolist())
    except ValueError as e:item.update(status='first_lift_pose_rejected',error=str(e))
    item['physical_state_unchanged']=bool(np.array_equal(q,r.arrays['mechanism_qpos']) and r.bridge is None)
    report['sources'][name]=item
    print(encode(item),flush=True)
report.update(status='diagnosis_complete',elapsed_seconds=time.monotonic()-started,
    next_gate='Review source-specific feasible CLEAR_PILE waypoint/retention constraints before any route implementation or physics; do not relax original solve threshold',
    full_path_validated=False,native_restore_tested=False)
(out/'report.json').write_text(encode(report,ensure_ascii=False,indent=2)+'\n')
