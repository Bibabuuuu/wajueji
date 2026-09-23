import json,sys
from pathlib import Path
import numpy as np
from moving_bucket_sweep_candidate import convex_planes
from layered_excavation_approach_planner import _load_kinematics
w=Path(__file__).resolve().parent;run=w/sys.argv[1];r=json.loads((run/'report.json').read_text());p=Path(r.get('terminal',r.get('recovery'))['path']);a=np.load(p)
proxy=json.loads(Path('/home/qunwang/wajueji/reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh');pose=assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket'];local=(a['particle_q']-pose[:3,3])@pose[:3,:3]
o,ns,ds,primary=convex_planes(None,proxy);ns=np.array(ns);ds=np.array(ds);inside=np.zeros(len(local),bool);details=[]
for i,name in enumerate([b['name'] for b in proxy['shell_boxes']+proxy['tooth_boxes']+proxy['side_plate_meshes']]):
 depth=np.min(ds[o[i]:o[i+1]]-local@ns[o[i]:o[i+1]].T,axis=1)
 mask=depth>1e-4;inside|=mask;details.append({'piece':name,'particles_deeper_than_0_1mm':int(mask.sum()),'maximum_inside_depth_m':float(max(0,depth.max()))})
rows=[json.loads(l) for l in (run/'trace.ndjson').read_text().splitlines()]
result={'frames':r['frames'],'status':r['status'],'particle_centers_inside_bucket_solid_deeper_than_0_1mm':int(inside.sum()),'pieces':details,'all_hard_safe':all(all(x['hard'].values()) for x in rows),'retained_initial_cavity':rows[-1]['original_cavity_particles_still_in_cavity'],'maximum_additional_current_correction_m':max(x['mpm_sweep']['maximum_position_correction_m'] for x in rows),'maximum_additional_motion_correction_m':max(x['actual_motion_sweep_pending_next_frame']['maximum_position_correction_m'] for x in rows),'physics_frames_wall_seconds':sum(x['wall_seconds'] for x in rows),'max_particle_speed_m_s':max(x['particle_max_speed_m_s'] for x in rows),'min_particle_z_m':min(x['particle_minimum_z_m'] for x in rows),'maximum_routed_bucket_force_n':max(np.linalg.norm(x['bucket_force']) for x in rows),'maximum_incremental_impulse_balance_error_ns':max(np.max(np.abs(x['incremental_impulse_balance_error_ns'])) for x in rows)}
(run/'terminal-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='pieces'},indent=2))
