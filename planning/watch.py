"""Bounded same-host read-only consumer. Does not import or control the simulator."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .phase_observer import PhaseObserver,bucket_shell_points


def write_latest(folder,value):
    temp=folder/'latest.tmp';temp.write_text(json.dumps(value,allow_nan=False)+'\n');temp.replace(folder/'latest.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=30.)
    args=parser.parse_args()
    if not 0<args.duration<=3600:parser.error('duration must be in (0,3600]')
    args.output.mkdir(parents=True,exist_ok=False)
    observer=PhaseObserver(bucket_shell_points(json.loads(args.contract.read_text())))
    start=time.perf_counter();lastkey=None;stages=Counter();timings=[];count=0
    with (args.output/'observations.jsonl').open('w') as trace:
        try:
            while time.perf_counter()-start<args.duration:
                packet=None;now=time.perf_counter()
                try:
                    packet=json.loads(args.source.read_text())
                    # Wall expiry is authoritative here; simulated time is last reported by the source.
                    result=observer.update(packet,packet.get('time',{}).get('processed_sim_s',0.),now)
                except (OSError,ValueError,AttributeError):result=observer.unavailable('source_unreadable')
                done=time.perf_counter();timings.append((done-now)*1000)
                result['evaluated_monotonic_s']=done
                if result['valid'] and done>result['expires_monotonic_s']:result=observer.unavailable('expired_during_processing')
                write_latest(args.output,result)
                key=(result.get('episode'),result.get('source_sequence'),result['stage'])
                if key!=lastkey:
                    trace.write(json.dumps({'input':packet,'output':result},allow_nan=False)+'\n');trace.flush()
                    stages[result['stage']]+=1;count+=1;lastkey=key
                time.sleep(.05)
        finally:
            write_latest(args.output,observer.unavailable('observer_stopped'))
    report={'source':str(args.source),'read_only':True,'commands_emitted':0,'duration_wall_s':time.perf_counter()-start,
            'unique_records':count,'stages':dict(stages),'config':asdict(observer.config),
            'read_and_judge_ms':{'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95))},
            'scope':'live data plumbing and bounded heuristic judgments; not semantic accuracy or full-cycle acceptance'}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
