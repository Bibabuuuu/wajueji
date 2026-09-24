"""Bounded sequential local replay of four previously ground-safe cycle sources."""
import json,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
runs=['manual-contact-20260924-083547-1327233','manual-contact-20260924-110051-1421079','manual-contact-20260924-110932-1426831','manual-contact-20260924-114908-1452582']
start=time.monotonic();results=[]
for run in runs:
 if (W/'STOP_QUEUE').exists():break
 if time.monotonic()-start>2400:raise RuntimeError('cycle_queue_wall_budget')
 print('START',run,flush=True)
 with (W/(run+'.log')).open('x') as log:
  r=subprocess.run([sys.executable,str(W/'replay.py'),'--protocol',str(W/(run+'.protocol.json'))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=630)
 results.append(dict(run=run,returncode=r.returncode));(W/'cycle-queue-status.json').write_text(json.dumps(results,indent=2)+'\n')
 print('END',run,r.returncode,flush=True)
 if r.returncode:break
