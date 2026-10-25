"""Diagnose the frozen rise gate from cached actual motion, without rejudging it."""
import json
from pathlib import Path
import numpy as np
from manual_key4_review import ROOT,W,sha

RUN=ROOT/'runs/contact-carry-handoff-20260924-v1/first-resume'
def main():
    files=[RUN/'physics/frames.ndjson',RUN/'handoff.ndjson']
    physical,gate=[[json.loads(s) for s in p.read_text().splitlines()] for p in files]
    closure=np.array([(r['after_state']['mechanism_qpos'][15]-.37)/.482 for r in physical])
    z=np.array([r['after_state']['bucket_origin_world_m'][2] for r in physical])
    held=next((i for i in range(180,len(closure)) if np.min(closure[i-180:i+1])>=.99),None)
    windows=[]
    for name,end in [('first_99_percent_held_3s',held),('final_3s',len(closure)-1)]:
        if end is None:continue
        start=end-180;v=z[start:end+1]
        windows.append(dict(name=name,frame_start=2328+start,frame_end=2328+end,
            minimum_closure_fraction=float(np.min(closure[start:end+1])),net_origin_rise_m=float(v[-1]-v[0]),
            minimum_rolling_rise_m={str(span/60):float(np.min(v[span:]-v[:-span])) for span in (12,30,60)},
            maximum_drawdown_m=gate[end]['maximum_drawdown_m'],frozen_gate_ready=gate[end]['ready']))
    result=dict(physics_steps_added=0,inputs={str(p):sha(p) for p in files},
        windows=windows,ready_count=sum(r['ready'] for r in gate),
        conclusion='Actual net rise coexists with millimetre-scale brief drawdowns. No gate or frozen result has been changed; no transport handoff executed.')
    (W/'rise-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
