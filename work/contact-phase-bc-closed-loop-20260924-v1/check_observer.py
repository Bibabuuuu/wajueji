"""Replay saved observations through the live adapter, without advancing physics."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
from contact_phase_live import LivePhaseObserver
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig
from layered_excavation_approach_planner import _load_kinematics

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
B=ROOT/'work/contact-derived-batch-20260924-v1';OLD=ROOT/'work/contact-phase-labels-20260924-v1'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def ident(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
 contact=OLD/'phase-candidate-v1.json';completion=OLD/'phase-completion-v1.json'
 cavity=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json')
 proxy=ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json'
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
 assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
 pose=lambda state:assembly.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
 results=[]
 for e in read(B/'manifest.json')['episodes']:
  if e['split']!='validation_runs':continue
  rp=Path(e['report']['path']);assert sha(rp)==e['report']['sha256'];r=read(rp);run=rp.parent
  trace=rows(run/'phase-trace.ndjson');physics=rows(run/'frames.ndjson');decisions=rows(run/'decisions.ndjson')
  observer=LivePhaseObserver(read(proxy),read(cavity)['bucket']['polygon_xz_m'],ContactPhaseConfig(**read(contact)['parameters']),CompletionConfig(**read(completion)['parameters']),pose(decisions[0]['before_state']))
  geometry=observer.geometry
  class RecordedGap:
   def observe(self,q,p):return self.current
  recorded=RecordedGap();observer.geometry=recorded
  for n,(stage,physical,decision) in enumerate(zip(trace,physics,decisions,strict=True),1):
   assert observer.router.phase_for_decision(n)==decision['phase']
   recorded.current=stage['observation']
   out=observer.observe_end(frame=n,state=physical['after_state'],pose=pose(physical['after_state']),soil_force_n=stage['bucket_mpm_force_norm_n'],rock_contact_substeps=physical['rock_bucket_contact_substeps'],particles=lambda:None)
   assert out['phase']==stage['phase'] and out['observation']==stage['observation']
  # Recompute geometry from the real saved confirmation particle arrays.
  for snapshot in r['confirmation_snapshots']:
   assert sha(snapshot['path'])==snapshot['sha256']
   with np.load(snapshot['path'],allow_pickle=False) as z:
    actual=geometry.observe(z['particle_q'],pose({'mechanism_qpos':z['mechanism_qpos']}))
   assert actual==trace[snapshot['frame']-1]['observation']
  results.append(dict(run=e['run'],physical_frames=len(trace),phase_results_exact=True,confirmation_snapshots=len(r['confirmation_snapshots']),report=ident(rp)))
 result=dict(passed=True,physics_steps=0,episodes=results,frames=sum(e['physical_frames'] for e in results),
  scope='All recorded signals through live causal adapter; stored per-frame distances used for routing, five real confirmation geometries per episode recomputed.',
  inputs=[ident(p) for p in (B/'manifest.json',contact,completion,cavity,proxy)],code=[ident(p) for p in (Path(__file__),ROOT/'src/v4/contact_phase_live.py',ROOT/'src/v4/contact_phase_geometry.py',ROOT/'src/v4/contact_phase_candidate.py',ROOT/'src/v4/contact_phase_completion.py')])
 (W/'observer-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
