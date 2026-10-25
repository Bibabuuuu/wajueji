"""Evaluate D155 on frozen D154 first and D152+D153 third frame-end caches."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from contact_transport_handoff import CarryHandoffGate, CarryHandoffConfig
from contact_phase_geometry import ground_connected_mask
from manual_key4_review import sha, ROOT, W

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]

def review(name, runs):
    report=read(runs[0]/'report.json');source=Path(report['source']['path'])
    assert sha(source)==report['source']['sha256']
    with np.load(source) as data:
        points=data['particle_q'];initial_q=data['mechanism_qpos'].copy()
    mask=ground_connected_mask(points);top=float(points[mask,2].max())
    gate=CarryHandoffGate(initial_soil_top_m=top,source_sha256=sha(source))
    states=[];phases={0:'approach_contact'};heights={};inputs={str(source):sha(source)}
    for n,run in enumerate(runs):
        s=read(run/'display/states.json')
        if n:
            assert states[-1]==s[0]
            states+=s[1:]
        else:states=s
        for row in rows(run/'phase-trace.ndjson'):
            phases[row['frame']]=row['phase']['phase_for_next_decision']
        for row in rows(run/'ground-guard.ndjson'):
            if row['stage'] in ('after_step','after'):heights[row['frame']]=row['minimum_bucket_z_m']
            elif row['stage']=='restored':heights[0]=row['minimum_bucket_z_m']
        for path in (run/'display/states.json',run/'ground-guard.ndjson',run/'phase-trace.ndjson',run/'report.json'):
            inputs[str(path)]=sha(path)
    assert np.array_equal(initial_q,states[0]['mechanism_qpos'])
    with np.load(runs[-1]/'final.npz') as data:
        assert np.array_equal(data['mechanism_qpos'],states[-1]['mechanism_qpos'])
    inputs[str(runs[-1]/'final.npz')]=sha(runs[-1]/'final.npz')
    measurements=[]
    for frame,state in enumerate(states):
        assert frame in heights and frame in phases
        fraction=(state['mechanism_qpos'][15]-.37)/(.852-.37)
        measurements.append(gate.observe_end(frame=frame,closure_fraction=fraction,
            origin_z_m=state['bucket_origin_world_m'][2],minimum_bucket_z_m=heights[frame],
            contact_established=phases[frame] in ('curl_load','lift','off_soil')))
    ready=[r for r in measurements if r['ready']]
    final=measurements[-1]
    restored=CarryHandoffGate.restore(gate.snapshot())
    assert restored.snapshot()==gate.snapshot()
    result=dict(name=name,physics_steps_added=0,frames_reviewed=len(states)-1,
                first_ready=None if not ready else ready[0],ready_frame_count=len(ready),final=final,
                final_closure_fraction=(states[-1]['mechanism_qpos'][15]-.37)/(.852-.37),
                peak_closure_fraction=max((s['mechanism_qpos'][15]-.37)/(.852-.37) for s in states),
                gate_snapshot=gate.snapshot(),inputs=inputs)
    print(name,json.dumps({k:result[k] for k in ('first_ready','ready_frame_count','final','final_closure_fraction','peak_closure_fraction')}),flush=True)
    return result

def main():
    results=[review('first',[ROOT/'runs/contact-multisource-lift-20260924-v1/first-candidate']),
             review('third',[ROOT/'runs/contact-full-response-20260924-v1/candidate60',ROOT/'runs/contact-preview-lift-20260924-v1/continuation'])]
    result=dict(schema='D155_cached_handoff_review_v1',config=asdict(CarryHandoffConfig()),
                physics_steps_added=0,episodes=results,
                candidate_code={str(p):sha(p) for p in (Path(__file__),ROOT/'src/v4/contact_transport_handoff.py')},
                limitations=['Read-only counterfactual switch times, not newly executed controllers.',
                             'Gate readiness does not certify payload, no spill or unloading.'])
    (W/'cache-review.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
