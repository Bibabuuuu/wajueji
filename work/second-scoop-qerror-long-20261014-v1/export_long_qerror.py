from pathlib import Path
import json,tarfile,hashlib
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';C=R/'work/second-scoop-wait-qerror-20261014-v1';D=R/'runs/second-scoop-qerror-long-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');audit=read(D/'read-only-audit.json');closure=read(W/'closure.json');n=p['frames'];states=read(D/'display/states.json');indices=sorted(set([0,1,2,3,min(1560,n),n]))
(W/'cached-state-witness.json').write_text(json.dumps(dict(source=str(D/'display/states.json'),source_sha256=sha(D/'display/states.json'),selected_frames=indices,states=[states[i] for i in indices]),indent=2))
files={}
def add(path,name):
 assert path.is_file(),path
 files[name]=path
for root in [W,C]:
 for f in root.rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.md','.json','.log'):add(f,str(f.relative_to(R)))
for f in D.rglob('*'):
 if not f.is_file():continue
 rel=f.relative_to(D)
 if rel.parts[0] in ('physics','display'):continue
 if f.suffix in ('.json','.ndjson'):
  assert f.stat().st_size<40000000,('unexpected_evidence_size',str(f),f.stat().st_size)
  add(f,'evidence/'+str(rel))
for name in read(W/'record-final-receipt.json'):add(R/name,name)
for name in ['manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']:add(R/'src/v4'/name,'source-snapshots/'+name)
add(R/'work/second-scoop-continuation-20261014-v1/yaw_transfer.py','source-snapshots/yaw_transfer.py')
add(R/'tests/v4/test_manual_action_handoff.py','source-snapshots/test_manual_action_handoff.py')
for name in ['reference_align.npz.json','reference_align.control.json']:add(R/'runs/second-scoop-short-phased-20261014-v1'/name,'evidence/source/'+name)
add(R/'AGENTS.md','project-AGENTS.snapshot.md')
archive=W/'delivery-export.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name,f in sorted(files.items()):tar.add(f,arcname=name,recursive=False)
print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=sha(archive))))
