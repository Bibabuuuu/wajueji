"""Five final cached geometry observations; no policy or physics advance."""
import json,hashlib
from pathlib import Path
import numpy as np
from contact_phase_geometry import SolidBucketGap
from layered_excavation_approach_planner import _load_kinematics
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/contact-first-lift-review-20260924-v1';RUN=ROOT/'runs/contact-multisource-lift-20260924-v1/first-candidate'
def main():
    report=json.loads((RUN/'report.json').read_text());n=report['frames_completed'];cache=RUN/'display';states=json.loads((cache/'states.json').read_text());positions=np.load(cache/'particles.npy',mmap_mode='r')
    assert len(states)==n+1
    endpoint=RUN/('final.npz' if report['status']=='bounded_probe_completed' else 'recovery.npz')
    with np.load(endpoint) as final:
        assert np.array_equal(final['particle_q'],positions[n]);assert np.array_equal(final['mechanism_qpos'],states[n]['mechanism_qpos'])
    proxy=ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json';geometry=SolidBucketGap(json.loads(proxy.read_text()));k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh');measurements=[]
    for f in range(max(0,n-4),n+1):
        pose=a.assemble(states[f]['mechanism_qpos'],include_meshes=False)[1]['bucket'];measurements.append(dict(frame=f,**geometry.observe(positions[f],pose)))
    gaps=[x['remaining_soil_gap_m'] for x in measurements]
    result=dict(status='read_only_final_five_frames',physics_steps_added=0,final_phase=report['final_phase'],five_frame_gap_3mm_passed=len(gaps)==5 and min(gaps)>=.003,measurements=measurements,
        source_checkpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),proxy_sha256=hashlib.sha256(proxy.read_bytes()).hexdigest(),limitations=['Does not rewrite causal phase transitions or qualify transport.','Particle centers to open solid proxy; no new rock-clearance proof.'])
    (W/'final-geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
