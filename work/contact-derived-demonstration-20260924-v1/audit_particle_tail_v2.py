"""Read-only audit of large per-particle endpoint differences, without relabeling."""
import hashlib,json
from pathlib import Path
import numpy as np
from contact_phase_geometry import ground_connected_mask
from layered_47_cycle_geometry import bucket_mask
from layered_excavation_approach_planner import _load_kinematics
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-derived-demonstration-v1/third-scoop-observe-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((RUN/'report.json').read_text());a=Path(r['original_endpoint']['path']);b=Path(r['final_checkpoint']['path'])
 assert sha(a)==r['original_endpoint']['sha256'] and sha(b)==r['final_checkpoint']['sha256']
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
 with np.load(a,allow_pickle=False) as old,np.load(b,allow_pickle=False) as new:
  q0=old['particle_q'].copy();q1=new['particle_q'].copy()
  p0=assembly.assemble(old['mechanism_qpos'],include_meshes=False)[1]['bucket'];p1=assembly.assemble(new['mechanism_qpos'],include_meshes=False)[1]['bucket']
 error=np.linalg.norm(q1.astype(float)-q0.astype(float),axis=1);c0=bucket_mask(q0,p0);c1=bucket_mask(q1,p1);g0=ground_connected_mask(q0);g1=ground_connected_mask(q1)
 counts=[]
 for threshold in (.001,.01,.1):
  mask=error>threshold
  counts.append(dict(threshold_m=threshold,count=int(mask.sum()),fraction=float(mask.mean()),original_cavity=int((mask&c0).sum()),derived_cavity=int((mask&c1).sum()),original_ground_connected=int((mask&g0).sum()),derived_ground_connected=int((mask&g1).sum())))
 worst=int(error.argmax());result=dict(schema='derived_endpoint_particle_tail_v2',inputs=[dict(path=str(p),sha256=sha(p)) for p in (a,b,Path(__file__),ROOT/'src/v4/contact_phase_geometry.py')],
 particle_count=len(error),quantiles_m={str(x):float(np.quantile(error,x)) for x in (.5,.9,.99,.999,1.)},thresholds=counts,
 worst=dict(index=worst,error_m=float(error[worst]),original_world_m=q0[worst].tolist(),derived_world_m=q1[worst].tolist(),original_cavity=bool(c0[worst]),derived_cavity=bool(c1[worst]),original_ground_connected=bool(g0[worst]),derived_ground_connected=bool(g1[worst])),
 cavity_membership_changed=int((c0!=c1).sum()),ground_connectivity_membership_changed=int((g0!=g1).sum()),training_eligible=False,
 interpretation='Per-particle trajectory identity is not established. Cavity and connectivity are endpoint geometric diagnostics, not path or material-failure explanations. Actual replay observations must remain paired with actual replay actions.')
 (W/'particle-tail-v2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
