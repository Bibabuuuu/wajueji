"""Read-only exact broad-phase comparison against frozen 145f53e observer."""
import hashlib, json, subprocess, time, types
from pathlib import Path
import numpy as np
from contact_phase_geometry import SolidBucketGap
from layered_excavation_approach_planner import _load_kinematics
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'work/contact-phase-labels-20260924-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=subprocess.check_output(['git','show','145f53e:src/v4/contact_phase_geometry.py'],cwd=ROOT,text=True)
baseline=types.ModuleType('frozen_geometry');exec(compile(source,'frozen_145f53e_geometry','exec'),baseline.__dict__)
prior=json.loads((OLD/'phase-completion-replay-v1.json').read_text())
proxy_path=Path(next(i['path'] for i in prior['geometry_inputs'] if i['path'].endswith('proxy-manifest.json')))
proxy=json.loads(proxy_path.read_text());a=baseline.SolidBucketGap(proxy);b=SolidBucketGap(proxy)
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
results=[]
for e in prior['episodes']:
 p=Path(e['endpoint']['snapshot']['path']);assert sha(p)==e['endpoint']['snapshot']['sha256']
 with np.load(p,allow_pickle=False) as d:q=d['particle_q'];pose=assembly.assemble(d['mechanism_qpos'],include_meshes=False)[1]['bucket']
 measurements={'baseline':[],'candidate':[]};outputs={}
 for repeat in range(6):
  for name,observer in (('baseline',a),('candidate',b)) if repeat%2==0 else (('candidate',b),('baseline',a)):
   start=time.perf_counter();value=observer.observe(q,pose);elapsed=time.perf_counter()-start
   outputs[name]=value
   if repeat:measurements[name].append(elapsed)
 assert outputs['baseline']==outputs['candidate']
 # Also compare altered rigid poses without advancing particles or physics.
 perturb=[]
 for dz in (-.1,-.01,.01,.1):
  shifted=pose.copy();shifted[2,3]+=dz
  x=a.minimum_gap(q,shifted);y=b.minimum_gap(q,shifted)
  assert abs(x-y)<1e-12
  perturb.append(dict(pose_only_z_shift_m=dz,baseline_m=x,candidate_m=y))
 assert sha(p)==e['endpoint']['snapshot']['sha256']
 item=dict(run=e['run'],source=e['endpoint']['snapshot'],observation=outputs['candidate'],
           timings_s=measurements,median_s={k:float(np.median(v)) for k,v in measurements.items()},pose_only_comparisons=perturb)
 results.append(item);print(e['run'],item['median_s'],flush=True)
report=dict(schema='phase_gap_broadphase_profile_v1',physics_advanced=False,
 baseline_git_commit='145f53e',baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
 candidate_source_sha256=sha(ROOT/'src/v4/contact_phase_geometry.py'),script_sha256=sha(Path(__file__)),
 prior_report_sha256=sha(OLD/'phase-completion-replay-v1.json'),episodes=results,
 limits='Single-process CPU timings, not end-to-end 60Hz; pose perturbations are geometry tests, not physical trials')
(OUT/'observer-profile-v1.json').write_text(json.dumps(report,indent=2)+'\n')
