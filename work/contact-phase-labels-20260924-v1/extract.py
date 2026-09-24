"""Geometry and contact audit only: no labels fitted, no physics advanced."""
from pathlib import Path
import argparse,json,hashlib,numpy as np
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
a,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
contract=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json')
poly=np.array(json.loads(contract.read_text())['bucket']['polygon_xz_m']);local=np.c_[poly[:,0],np.zeros(len(poly)),poly[:,1]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',nargs='+',default=['010900-1088617','100857-1387193','110932-1426831']);parser.add_argument('--summary',default='feature-summary.json');args=parser.parse_args()
summary=[]
for suffix in args.runs:
 run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/('manual-contact-20260924-'+suffix)
 rows=readrows(run/'physics.ndjson');states=[rows[0]['before_state']]+[r['after_state'] for r in rows];trace=[]
 for i,s in enumerate(states):
  pose=a.assemble(np.asarray(s['mechanism_qpos']),include_meshes=False)[1]['bucket'];world=local@pose[:3,:3].T+pose[:3,3]
  radial=np.array([pose[1,1],-pose[0,1]]);radial/=np.linalg.norm(radial);chord=world[-1]-world[0]
  angle=float(np.rad2deg(np.arctan2(chord[:2]@radial,-chord[2])))
  tip=pose[:3,:3]@TIP+pose[:3,3]
  r=None if i==0 else rows[i-1]
  trace.append(dict(frame=i,time_s=i/60,available_from_decision_frame=i+1,mouth_chord_from_down_deg=angle,
   tip_z_m=float(tip[2]),origin_z_m=float(pose[2,3]),q_bucket=float(s['mechanism_qpos'][15]),q_boom=float(s['mechanism_qpos'][9]),
   origin_vertical_velocity_m_s=s['bucket_linear_velocity_world_m_s'][2],
   soil_samples=None if r is None else r['bucket_mpm_contact_samples'],rock_substeps=None if r is None else r['rock_bucket_contact_substeps'],
   soil_force_n=None if r is None else r['bucket_mpm_force_norm_n'],
   command=None if r is None else (np.asarray(r['applied_command_si'])/np.array([.12,.0225,.0225,.0675])).tolist(),
   section_rz=np.round(np.c_[world[:,:2]@radial,world[:,2]],5).tolist()))
 annotations=readrows(run/'annotations.ndjson')
 result=dict(run=run.name,calibration_status='unreviewed_features_only',geometry_scope='Recorded rigid pose and CAD cavity section; no reconstructed particle video',
  angle_reference='CAD cavity upper rim to lower cutting rim chord; signed relative to world downward vertical in instantaneous bucket radial plane. User visual meaning of vertical still unconfirmed.',
  section_source=dict(path=str(contract),sha256=sha(contract)),inputs=[dict(path=str(run/n),sha256=sha(run/n)) for n in ('physics.ndjson','annotations.ndjson')],annotations=annotations,trace=trace)
 (OUT/(suffix+'-features.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
 nz=[r['frame'] for r in trace if r['soil_samples'] is not None and r['soil_samples']>0];closest=min(trace,key=lambda r:abs(r['mouth_chord_from_down_deg']))
 summary.append(dict(run=run.name,frames=len(rows),first_nonzero_soil_sample_frame=nz[0] if nz else None,closest_vertical=dict(frame=closest['frame'],angle=closest['mouth_chord_from_down_deg']),keys=[dict(frame=e.get('next_frame'),label=e.get('label')) for e in annotations if e.get('kind')=='phase'],samples=[{k:r[k] for k in ('frame','mouth_chord_from_down_deg','tip_z_m','command','soil_samples')} for r in trace[::120]]))
(OUT/args.summary).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
