"""CPU-only screen for a shorter already-guarded approach, not live admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,time
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
D=m.OUT;O=W/'cpu-short-approach';O.mkdir(exist_ok=False);source=D/'first-returned.npz';start=time.monotonic();r=m.CpuRuntime(source,O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
r.bridge=SimpleNamespace(state={'mechanism_qpos':r.arrays['mechanism_qpos'].copy()})
spec=m.importlib.util.spec_from_file_location('short_stock',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=m.importlib.util.module_from_spec(spec);spec.loader.exec_module(observer);terrain,obs=observer.stationary_terrain(r,r.arrays)
entry=m.read(D/'report.json')['scoops'][0]['actual_entry_mouth_xy_m'];ranked,_=m.rank_adjacent(terrain,entry,inward_sign=-1);selected=ranked[0];reference=m.ManualResidualEnv(O/'reference',None,collect=False,waiting_action_mode='error_strength_v1').reference_q.copy();angle=np.arctan2(selected['entry_xy_m'][1],selected['entry_xy_m'][0])-reference[0,7];reference[:,7]+=angle;goal=reference[0].copy();goal[9]+=.001;goal=m.project(r,goal);initial=m.project(r,r.arrays['mechanism_qpos']);checker=m.MinimumHeightClearance(r,terrain)
failures=[];accepted=None
with np.load(R/'work/continuous-single-scoop-20261014-v1/cpu-v2/approach/plan.npz') as z:legacy=z['APPROACH_OUTWARD_qpos'][-1].copy()
legacy[:7]=initial[:7];legacy[7]=goal[7];legacy=m.project(r,legacy)
for name,goals in [('direct',[goal]),('legacy_outward_then_goal',[legacy,goal])]:
 try:
  parts=[];point=initial
  for target in goals:parts.append(m.segment(r,point,target,checker));point=target
  accepted=dict(name=name,frames=sum(x['steps'] for x in parts),minimum_gap=min(x['minimum_clearance']['bucket_observed_terrain_m'] for x in parts))
  np.savez(O/'candidate.npz',**{'q%d'%i:x['qpos'] for i,x in enumerate(parts)},**{'v%d'%i:x['velocity'] for i,x in enumerate(parts)})
  break
 except ValueError as e:failures.append(dict(name=name,error=str(e)))
m.write(O/'report.json',dict(status='cpu_short_approach_passed' if accepted else 'cpu_short_approach_rejected',accepted=accepted,rejected=failures,selected=selected,source_sha256=m.sha(source),physical_steps=0,wall_seconds=time.monotonic()-start));print(accepted,failures,flush=True)
