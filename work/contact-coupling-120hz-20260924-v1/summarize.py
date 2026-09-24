"""Read-only timing/response summary; never advances physics."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

def interval(data,key,start,end):
    a=[float(r[key]) for r in data if start<=r['frame']<=end]
    if not a:return None
    return dict(frames=len(a),simulated_seconds=len(a)/60,wall_seconds=sum(a),wall_seconds_per_simulated_second=sum(a)*60/len(a),frame_wall_quantiles=np.quantile(a,[0,.5,.95,1]).tolist())

def summarize(name):
    p=ROOT/'runs/contact-coupling-120hz-20260924-v1'/name
    r=read(p/'report.json');physical=rows(p/'physics/frames.ndjson');loops=rows(p/'frame-timing.ndjson');runtime=rows(p/'physics/timing.ndjson')
    result=dict(run=str(p),status=r['status'],failure=r.get('failure'),frames=r['frames_completed'],phase=r.get('final_phase'),
        startup_wall_seconds=r.get('startup_wall_seconds'),total_wall_seconds=r['wall_seconds'],zero_exact=all(r['zero_step_arrays_exact'].values()),gates=r.get('same_instance_gates'),
        all_hard=all(x['hard_guard'] for x in physical),source_unchanged=r['source_unchanged'],windows={})
    for start,end in ((1,120),(61,120),(121,432),(433,900),(901,972),(1,len(physical))):
        label=f'{start}-{end}'
        result['windows'][label]={name:interval(data,key,start,end) for name,data,key in [('physics',physical,'wall_seconds'),('mpm',physical,'mpm_wall_seconds'),('rigid',physical,'mujoco_wall_seconds'),('runtime_with_guards',runtime,'runtime_step_wall_seconds'),('full_loop',loops,'wall_seconds')]}
    velocity=np.array([r['after_state']['mechanism_qvel'] for r in physical])
    result['velocity']=dict(bucket_peak_absolute_m_s=float(abs(velocity[:,3]).max()),bucket_max_adjacent_change_m_s=float(abs(np.diff(velocity[:,3])).max()),by_window={})
    for lo,hi in ((901,972),(940,972),(1,len(physical))):
        a=velocity[lo-1:hi,3]
        if len(a)>1:result['velocity']['by_window'][f'{lo}-{hi}']=dict(peak_absolute_m_s=float(abs(a).max()),max_adjacent_change_m_s=float(abs(np.diff(a)).max()),sign_reversals=int(np.count_nonzero(a[1:]*a[:-1]<0)))
    result['outputs']=[dict(path=str(f),sha256=sha(f)) for f in (p/'report.json',p/'physics/frames.ndjson',p/'frame-timing.ndjson',p/'physics/timing.ndjson')]
    return result

if __name__=='__main__':
    results={name:summarize(name) for name in ('baseline60','candidate120') if (ROOT/'runs/contact-coupling-120hz-20260924-v1'/name/'report.json').exists()}
    old=ROOT/'runs/contact-geometric-probe-20260924-v9-handoff/contact1200'
    oldrows=rows(old/'physics/frames.ndjson');newrows=rows(ROOT/'runs/contact-coupling-120hz-20260924-v1/baseline60/physics/frames.ndjson')
    results['baseline_120_actual_states_equal_frozen60']=len(newrows)==120 and all(a['after_state']==b['after_state'] for a,b in zip(newrows,oldrows))
    results['historical60']=dict(run=str(old),windows={f'{lo}-{hi}':{key:interval(oldrows,key,lo,hi) for key in ('wall_seconds','mpm_wall_seconds','mujoco_wall_seconds')} for lo,hi in ((1,120),(433,900),(901,972),(1,972))})
    if 'candidate120' in results:
        results['matched_startup_ratios']={key:results['candidate120']['windows']['1-120'][key]['wall_seconds_per_simulated_second']/results['baseline60']['windows']['1-120'][key]['wall_seconds_per_simulated_second'] for key in results['baseline60']['windows']['1-120']}
    (W/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
