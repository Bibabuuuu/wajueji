from pathlib import Path
import json,tarfile,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';C=R/'work/second-scoop-wait-qerror-20261014-v1';D=R/'runs/second-scoop-full-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
states=read(D/'display/states.json')
(C/'cached-state-witness.json').write_text(json.dumps(dict(source=str(D/'display/states.json'),source_sha256=sha(D/'display/states.json'),selected_frames=[0,1,2,3,1560],states=[states[i] for i in [0,1,2,3,1560]]),indent=2))
files={}
def add(path,name):
 assert path.is_file(),path
 files[name]=path
for root in [W,C]:
 for p in root.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.log'):
   add(p,str(p.relative_to(R)))
for p in D.rglob('*'):
 if not p.is_file() or p.stat().st_size>2500000:continue
 rel=p.relative_to(D)
 if rel.parts[0]=='physics' or rel.parts[0]=='display':continue
 if p.suffix in ('.json','.ndjson') or p.name=='stall-reference-window.npz':add(p,'evidence/'+str(rel))
for n in read(W/'record-final-receipt.json'):add(R/n,n)
for n in ['manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']:add(R/'src/v4'/n,'source-snapshots/'+n)
add(R/'work/second-scoop-continuation-20261014-v1/yaw_transfer.py','source-snapshots/yaw_transfer.py')
add(R/'tests/v4/test_manual_action_handoff.py','source-snapshots/test_manual_action_handoff.py')
for name in ['reference_align.npz.json','reference_align.control.json']:add(R/'runs/second-scoop-short-phased-20261014-v1'/name,'evidence/source/'+name)
add(R/'AGENTS.md','project-AGENTS.snapshot.md')
archive=W/'delivery-export.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=sha(archive))))
