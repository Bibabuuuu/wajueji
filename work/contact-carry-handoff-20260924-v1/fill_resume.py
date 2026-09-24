"""Read-only working fill at the new handoff and endpoint; no physics."""
import json
from pathlib import Path
import numpy as np
from manual_key4_review import ROOT, W, sha
from live_bucket_fill import MixedFillEstimator, load_config
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices

RUN=ROOT/'runs/contact-carry-handoff-20260924-v1/first-resume'
def main():
    read=lambda p:json.loads(p.read_text())
    report=read(RUN/'report.json');scene=read(RUN/'display/scene.json');states=read(RUN/'display/states.json')
    records=scene['records'];vertices=_cache_rock_vertices(Path(scene['root']),records)
    config=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json')
    estimator=MixedFillEstimator(config,[r['volume_m3'] for r in records],vertices)
    k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
    assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
    particles=np.load(RUN/'display/particles.npy',mmap_mode='r')
    n=report['frames_completed'];samples={0,*range(max(0,n-4),n+1)}
    if report['handoff']:samples.add(report['handoff']['frame_end']-2327)
    measurements=[]
    for i in sorted(samples):
        st=states[i];pose=assembly.assemble(st['mechanism_qpos'],include_meshes=False)[1]['bucket']
        result=estimator.measure_snapshot(dict(frame=i,pose=pose,soil=particles[i],rocks=[r['qpos'] for r in st['rocks']]))
        result['global_frame']=2327+i;measurements.append(result)
    endpoint=RUN/('recovery.npz' if report['status']=='failed' else 'final.npz')
    result=dict(physics_steps_added=0,formal_fill_qualification=False,checkpoint_sha256=sha(endpoint),
                measurements=measurements,limitation='Working bucket cavity plus heap occupancy only; no stable unloaded volume or causal spill estimate.')
    (W/'resume-fill.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
