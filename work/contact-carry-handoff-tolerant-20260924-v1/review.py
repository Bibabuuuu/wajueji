"""Replay the new gate on recorded actual motion; zero physical steps."""
import hashlib,json
from pathlib import Path
from gate import restore_with_tolerance

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=ROOT/'work/contact-carry-handoff-20260924-v1/cache-review.json'
    run=ROOT/'runs/contact-carry-handoff-20260924-v1/first-resume'
    physical=rows(run/'physics/frames.ndjson');previous=rows(run/'handoff.ndjson')
    gate=restore_with_tolerance(read(old)['episodes'][0]['gate_snapshot'])
    ready=[]
    for row,prior in zip(physical,previous,strict=True):
        st=row['after_state']
        result=gate.observe_end(frame=2327+row['frame'],closure_fraction=(st['mechanism_qpos'][15]-.37)/.482,
            origin_z_m=st['bucket_origin_world_m'][2],
            minimum_bucket_z_m=prior['bucket_above_initial_soil_m']+prior['initial_soil_top_m'],contact_established=True)
        if result['ready']:ready.append(result)
    assert ready[0]['frame_end']==2678 and result['ready']
    snapshot=gate.snapshot();assert restore_with_tolerance(snapshot).snapshot()==snapshot
    report=dict(status='cached_tolerant_gate_passed',physics_steps_added=0,actual_handoff_executed=False,
        first_ready=ready[0],final=result,ready_frame_count=len(ready),gate_snapshot=snapshot,
        inputs={str(p):sha(p) for p in (old,run/'physics/frames.ndjson',run/'handoff.ndjson',run/'final.npz',HERE/'gate.py',Path(__file__),ROOT/'src/v4/contact_transport_handoff.py')},
        limitation='Counterfactual gate evidence only. Original run is unchanged; no new physical handoff, transport or payload qualification.')
    (HERE/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','first_ready','final','ready_frame_count','physics_steps_added')}))
if __name__=='__main__':main()
