"""Verify completed evidence without rerunning physics or changing frozen inputs."""
from pathlib import Path
import ast,hashlib,json,subprocess
out=Path(__file__).resolve().parent;root=out.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
p=json.loads((out/'protocol.json').read_text());c=json.loads((out/'comparison.json').read_text());r=json.loads((out/'candidate/report.json').read_text())
checks={'frozen_runner':sha(out/'replay.py')==p['runner_sha256'],'frozen_header':sha(out/'bucket_integral.h')==p['header_sha256'],'prior_evidence_bindings':all(sha(k)==v for k,v in p['prior_bindings'].items()),'runtime_source_bindings':all(sha(k)==v for k,v in r['source_bindings'].items()),'analysis_bindings':all(sha(k)==v for k,v in c['bindings'].items()),'full_state_4x23_exact':len(c['exact_state_checks'])==4 and all(len(x)==23 and all(x.values()) for x in c['exact_state_checks'].values()),'prefix1140_trace_exact':all(c['prefix_trace_exact'].values()),'all_original_hard_guards':c['gates']['all_original_hard_guards'],'reported1290_frames':r['frames']==1290}
for name in ['replay.py','analyze.py','finalize.py']:ast.parse((out/name).read_text())
checks['syntax']=True
unit=subprocess.run([str(out/'test_release')],capture_output=True,text=True);checks['unit_tests']=unit.returncode==0
checks['native_compile_and_run']=r['frames']==1290 and (out/'candidate/instrumented-bridge.cc').is_file()
docs=subprocess.run(['python3','scripts/check_docs.py'],cwd=root,capture_output=True,text=True);checks['docs']=docs.returncode==0
diff=subprocess.run(['git','diff','--check','--','docs/CURRENT_STATUS.md','docs/plans/S04-contact-excavation-demonstrations.md','docs/decisions.md','docs/acceptance.md'],cwd=root,capture_output=True,text=True);checks['docs_diff']=diff.returncode==0
pids=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  parts=f.read_bytes().split(b'\0')
  if any(x==str(out/'replay.py').encode() or x==b'work/bucket-integral-release-20260923-v1/replay.py' for x in parts):pids.append(int(f.parent.name))
 except OSError:pass
checks['experiment_process_ended']=not pids
v={'checks':checks,'artifact_verification_pass':all(checks.values()),'physical_protocol_pass':c['all_gates_pass'],'default_enabled':False,'training_eligible':False,'unit_stdout':unit.stdout,'docs_stdout':docs.stdout,'replay_pids':pids}
(out/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2));assert all(checks.values())
files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='MANIFEST.sha256.json'}
(out/'MANIFEST.sha256.json').write_text(json.dumps(files,indent=2)+'\n');print('MANIFEST_FILES',len(files))
