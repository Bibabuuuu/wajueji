from pathlib import Path
import json,subprocess,numpy as np
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import TransportClearance
root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent;run=Path((out/'run-directory.txt').read_text().strip())
r=CycleRuntime(run/'source-audit/final.npz',out/'geometry-equivalence')
new=TransportClearance(r)
ns={'__file__':str(root/'src/v4/layered_47_cycle_geometry.py'),'__name__':'frozen_geometry'}
exec(compile(subprocess.check_output(['git','show','39bbd2b:src/v4/layered_47_cycle_geometry.py'],text=True),ns['__file__'],'exec'),ns)
old=ns['TransportClearance'](r)
def result(obj,q):
 try:return ('pass',obj.check(q,physical=True))
 except ValueError as e:return ('reject',str(e))
q=r.arrays['mechanism_qpos'].copy();checks=[]
for i in range(21):
 if i:
  for _ in range(10):r.chain.q=q.copy();q=r.chain.advance(np.array([0.,.01,0.,0.]),1/60)[0]
 a=result(old,q);b=result(new,q);assert a==b,(i,a,b);checks.append(dict(index=i,result=a[0]))
assert {c['result'] for c in checks}=={'pass','reject'}
(out/'geometry-equivalence.json').write_text(json.dumps(dict(exact_same_results=True,cases=checks),indent=2)+'\n');print('21 geometry pass/reject results unchanged')
