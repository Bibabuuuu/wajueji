"""Read-only endpoint mixed fill estimate; never grants formal volume qualification."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/contact-preview-lift-20260924-v1';RUN=ROOT/'runs/contact-preview-lift-20260924-v1/continuation'
from live_bucket_fill import MixedFillEstimator,load_config
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices

def main():
 report=json.loads((RUN/'report.json').read_text());scene=json.loads((RUN/'display/scene.json').read_text());states=json.loads((RUN/'display/states.json').read_text());state=states[-1];source=RUN/('final.npz' if report['status']=='bounded_continuation_completed' else 'recovery.npz')
 records=scene['records'];vertices=_cache_rock_vertices(Path(scene['root']),records);config=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json');estimator=MixedFillEstimator(config,[r['volume_m3'] for r in records],vertices)
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh');pose=a.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
 with np.load(source) as f:soil=f['particle_q'].copy()
 rocks=[r['qpos'] for r in state['rocks']]
 particles=np.load(RUN/'display/particles.npy',mmap_mode='r');measurements=[]
 for local in (0,60,120,240,360,420,421,422,423,424):
  st=states[local];pose=a.assemble(st['mechanism_qpos'],include_meshes=False)[1]['bucket']
  measured=estimator.measure_snapshot(dict(frame=2400+local,pose=pose,soil=particles[local],rocks=[r['qpos'] for r in st['rocks']]))
  measurements.append(measured)
 result=measurements[-1]
 result.update(checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),final_phase=report['final_phase'],formal_fill_qualification=False,physics_steps_added=0,sampled_history=measurements[:-1])
 (W/'fill-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
