"""Verify the three cached physical segments and write a no-physics replay."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
SOURCES=[('contact-next-scoop-20260925-v1/candidate','Dig / curl'),('contact-second-finish-unload-20260925-v4/physical','Finish curl / handoff'),('contact-second-fixed-route-20260925-v1/physical','Fixed lift / swing / dump')]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    total=0;previous_state=None;previous_particles=None;items=[]
    for relative,label in SOURCES:
        d=ROOT/'runs'/relative/'display';meta=json.loads((d/'reconstruction.json').read_text())
        for name,h in meta['cache_sha256'].items():
            if digest(d/name)!=h:raise ValueError('changed_cache:'+str(d/name))
        states=json.loads((d/'states.json').read_text());particles=np.load(d/'particles.npy',mmap_mode='r')
        if len(states)!=meta['frames']+1:raise ValueError('cache_frame_count')
        if previous_state is not None and (states[0]!=previous_state or not np.array_equal(particles[0],previous_particles)):raise ValueError('cache_boundary_changed:'+str(d))
        previous_state=states[-1];previous_particles=particles[len(states)-1].copy();total+=meta['frames']
        items.append(dict(path=str(d),label=label))
    result=json.loads((ROOT/'runs/contact-second-fixed-route-20260925-v1/physical/report.json').read_text())
    manifest=dict(label='SECOND SCOOP | COMPLETE CONTINUOUS CACHE',summary=result['status']+'; development result; formal fill not qualified',sources=items)
    (W/'second-replay.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (W/'replay-review.json').write_text(json.dumps(dict(status='cache_sha_and_boundaries_passed',frames=total,physical_steps_added=0),indent=2)+'\n')
    print('REPLAY_READY',total,flush=True)
if __name__=='__main__':main()
