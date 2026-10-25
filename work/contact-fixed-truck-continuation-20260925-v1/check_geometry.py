"""Recheck the frozen remaining route against the current observed terrain."""
import json,time
import numpy as np
from runtime import ROOT,W,SOURCE,TransportRuntime,read,sha
from observation import stationary_terrain
from layered_47_cycle_geometry import TransportClearance

def main():
    started=time.monotonic()
    r=TransportRuntime(SOURCE,ROOT/'runs/contact-fixed-truck-continuation-20260925-v1/geometry')
    terrain,observation=stationary_terrain(r);checker=TransportClearance(r,terrain)
    initial=checker.check(r.arrays['mechanism_qpos'],physical=True)
    pathfile=ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2/plan.npz'
    paths=np.load(pathfile);plan=read(pathfile.parent/'report.json');count=0;stages=[]
    for segment in plan['stages'][1:]:
        name=segment['name'];qs=paths[name+'_qpos'];minimum={}
        for left,right in zip(qs[:-1],qs[1:],strict=True):
            for q in ((left+right)/2,right):
                measured=checker.check(q)
                for key,value in measured.items():minimum[key]=min(minimum.get(key,float('inf')),value)
                count+=1
        stages.append(dict(name=name,minimum=minimum));print('GEOMETRY',name,count,flush=True)
    report=dict(status='updated_scene_route_geometry_passed',physics_steps_added=0,
        sample_count=count,samples_per_policy_frame=2,initial=initial,observation=observation,
        stages=stages,wall_seconds=time.monotonic()-started,
        scope='Frozen reference endpoints and midpoints; no new physical outcome',
        inputs={str(p):sha(p) for p in [SOURCE,Path(str(SOURCE)+'.json'),pathfile,W/'observation.py',W/'runtime.py',Path(__file__),ROOT/'work/contact-first-transport-review-20260925-v1/static_rocks.py']})
    (W/'geometry-review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

from pathlib import Path
if __name__=='__main__':main()
