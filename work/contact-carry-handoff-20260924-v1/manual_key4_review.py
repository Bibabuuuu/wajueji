"""Read original key-4 events and actual cylinder positions; zero physics."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
CATALOG=ROOT/'work/contact-collection-progress-20260924-v5/catalog.json'
MODEL=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance/scene/input/model/A.xml')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    # This model calls the bucket cylinder "arm piston rod". Its qpos address
    # is 15 in the saved 19-coordinate mechanism, axes [7,9,12,15].
    joint=next(j for j in ET.parse(MODEL).getroot().findall('.//joint')
               if j.get('name')=='arm piston rod')
    low,hard_upper=map(float,joint.get('range').split())
    assert (low,hard_upper)==(.37,.86)
    catalog=json.loads(CATALOG.read_text());episodes=[]
    for entry in catalog['episodes']:
        run=Path(entry['snapshot']['path']).parent
        paths=[run/'report.json',run/'physics.ndjson']
        bindings={str(p):sha(p) for p in paths}
        assert all(bindings[str(p)]==entry['source_bindings'][str(p)] for p in paths)
        report=json.loads(paths[0].read_text())
        upper=report['protocol']['bucket_internal_upper_target_m']
        assert upper==.852 and report['protocol']['hz']==60
        annotations=[json.loads(line) for line in (run/'annotations.ndjson').read_text().splitlines()]
        assert [r for r in annotations if r['kind']=='phase']==report['phase_events']
        bindings[str(run/'annotations.ndjson')]=sha(run/'annotations.ndjson')
        events=[e for e in report['phase_events'] if e['label']=='off_soil_confirm']
        states={};befores={}
        with paths[1].open() as f:
            for line in f:
                p=json.loads(line);frame=p['frame']
                states[frame]=(p['after_state']['mechanism_qpos'][15],
                               p['after_state']['bucket_origin_world_m'][2])
                befores[frame]=p['before_state']['mechanism_qpos'][15]
        assert sorted(states)==list(range(1,report['frames_completed']+1)) if 'frames_completed' in report else sorted(states)==list(range(1,len(states)+1))
        measurements=[]
        for number,event in enumerate(events,1):
            frame=event['next_frame']-1
            assert abs(event['sim_time_s']-frame/60)<1e-10
            actual,z=states[frame]
            if frame+1 in befores:assert actual==befores[frame+1]
            fraction=(actual-low)/(upper-low)
            start=frame
            while start>1 and (states[start-1][0]-low)/(upper-low)>=.95:start-=1
            hold=(frame-start)/60 if fraction>=.95 else 0.
            window=[states[k] for k in range(max(1,frame-180),frame+1)]
            measurements.append(dict(event_number=number,frame_end=frame,
                simulation_seconds=event['sim_time_s'],actual_bucket_cylinder_m=actual,
                effective_stroke_percent=100*fraction,
                physical_stroke_percent=100*(actual-low)/(hard_upper-low),
                below_internal_upper_mm=(upper-actual)*1000,
                consecutive_at_least_95_percent_seconds=hold,
                last_3s_minimum_effective_percent=100*min((q-low)/(upper-low) for q,_ in window),
                last_3s_origin_rise_m=z-window[0][1]))
        episodes.append(dict(run=entry['run'],scoop_index=entry['scoop_index'],
                             source_bindings=bindings,key4_events=measurements))
        print(entry['run'].split('-')[3],entry['scoop_index'],
              [(m['frame_end'],round(m['effective_stroke_percent'],4),
                round(m['physical_stroke_percent'],4),round(m['consecutive_at_least_95_percent_seconds'],3),
                round(m['last_3s_origin_rise_m'],4)) for m in measurements],flush=True)
    first=[e['key4_events'][0] for e in episodes if e['key4_events']]
    result=dict(schema='manual_key4_stroke_review_v1',physics_steps_added=0,
        inputs={str(CATALOG):sha(CATALOG),str(MODEL):sha(MODEL),str(Path(__file__)):sha(__file__)},
        event_semantics='key 4 occurs after next_frame-1, before next_frame command; use actual after_state qpos[15]',
        effective_range_m=[low,.852],physical_range_m=[low,hard_upper],
        calculation='100*(actual_position-lower)/(upper-lower); no clipping; not bucket angle or fill',
        episodes=episodes,first_key4_effective_percent_min=min(m['effective_stroke_percent'] for m in first),
        first_key4_effective_percent_max=max(m['effective_stroke_percent'] for m in first),
        first_key4_mean_effective_percent=sum(m['effective_stroke_percent'] for m in first)/len(first),
        limitations=['Original manual physics domain, not the new full-response interface.',
                     'User key 4 is an annotation, not proof of detachment, retention or successful unloading.',
                     'Net 3-second rise alone is not a sustained-rise test.',
                     '95 percent remains a proposed threshold, not validated by this read-only review.'])
    (W/'manual-key4-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FIRST_EVENT_RANGE',result['first_key4_effective_percent_min'],result['first_key4_effective_percent_max'],flush=True)

if __name__=='__main__':main()
