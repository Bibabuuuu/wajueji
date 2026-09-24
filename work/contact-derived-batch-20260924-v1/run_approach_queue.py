"""Run only after four cycle replays succeed; probe original initialization first."""
import json,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
status=json.loads((W/'cycle-queue-status.json').read_text())
assert len(status)==4 and all(x['returncode']==0 for x in status),'cycle_queue_not_complete'
start=time.monotonic();results=[]
with (W/'approach-initialization-probe.log').open('x') as log:
 probe=subprocess.run([sys.executable,str(W/'probe_approach.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=150)
assert probe.returncode==0,'approach_initialization_probe_failed'
for run in ['manual-contact-20260924-010900-1088617','manual-contact-20260924-005742-1081585','manual-contact-20260924-100857-1387193','manual-contact-20260924-113555-1443279']:
 if (W/'STOP_QUEUE').exists():break
 if time.monotonic()-start>2550:raise RuntimeError('approach_queue_wall_budget')
 print('START',run,flush=True)
 with (W/(run+'.log')).open('x') as log:
  r=subprocess.run([sys.executable,str(W/'replay.py'),'--protocol',str(W/(run+'.protocol.json'))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=630)
 results.append(dict(run=run,returncode=r.returncode));(W/'approach-queue-status.json').write_text(json.dumps(results,indent=2)+'\n')
 print('END',run,r.returncode,flush=True)
 if r.returncode:break
