"""CPU preparation only. A collection gate never authorizes SAC or a budget."""
import ast
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from manual_residual_sac_env import environment_contract, sha, write

root=Path('/home/qunwang/wajueji')
out=root/'work/manual-strength-collection-20261013-v2'
out.mkdir(exist_ok=False)
contract=environment_contract()
physical=root/'runs/manual-wait-strength-validation-20261013-v2'
rescored=root/'runs/manual-hold-rescore-20261013-v1'
evidence=[physical/'review.json',physical/'read-only-audit.json',rescored/'review.json']
p=json.loads(evidence[0].read_text());a=json.loads(evidence[1].read_text());r=json.loads(evidence[2].read_text())
checks={'physical_control_audit_passed':a['status']=='passed',
        'corrected_score_evidence_passed':r['status']=='passed' and r['terminal_score_version']==contract['terminal_score_version'],
        'half_full_qualified_under_corrected_score':all(e['reevaluated_success'] for e in r['episodes'] if e['label'] in ['loaded_half','loaded_full']),
        'cpu_only':os.environ.get('CUDA_VISIBLE_DEVICES')==''}
stream=io.StringIO();sys.path.insert(0,str(root/'tests/v4'))
tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(
    unittest.defaultTestLoader.discover(str(root/'tests/v4'),pattern='test_manual*.py'))
(out/'tests.txt').write_text(stream.getvalue());checks['all_cpu_tests_passed']=tests.wasSuccessful()
gate=dict(status='ready_for_collection' if all(checks.values()) else 'failed',
          environment_contract=contract,code=contract['code'],
          source_evidence={str(path):sha(path) for path in evidence},
          runtime_bindings=p['episodes'][0]['result']['runtime_bindings'],
          collection_admission=all(checks.values()),training_admission=False,
          new_dataset_verified=False,permission_scope='CPU receipt only; new physical budget requires user confirmation',
          neural_updates=0,physical_frames=0)
write(out/'collection-admission.json',gate)
assert all(checks.values()),'CPU/evidence preflight failed'
commands=[
 ('collection', [sys.executable,str(root/'src/v4/validate_manual_wait_strength.py'),
     '--output',str(out/'not-created-collection'),'--collect','--collection-admission',str(out/'collection-admission.json'),'--check-only']),
 ('training', [sys.executable,str(root/'src/v4/run_manual_residual_sac.py'),
     '--output',str(out/'not-created-training'),'--waiting-action-mode','error_strength_v1',
     '--alignment-admission',str(out/'collection-admission.json'),'--check-only'])]
cli=[]
for label,command in commands:
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    (out/(label+'-preflight.txt')).write_text(result.stdout+result.stderr)
    data=json.loads(result.stdout) if result.returncode==0 else {}
    checks[label+'_preflight_passed']=result.returncode==0
    checks[label+'_no_output_directory']=not (out/('not-created-'+label)).exists()
    checks[label+'_no_physics_or_updates']=data.get('physical_frames')==data.get('neural_updates')==0
    checks[label+'_expected_stage']=data.get('status')==('ready_for_collection' if label=='collection' else 'awaiting_verified_new_collection')
    cli.append(dict(label=label,returncode=result.returncode,receipt=data))
old=root/'runs/manual-residual-sac-aligned-20261013/reward-transfer-v2/training-admission.json'
for label,command in commands:
    bad=[str(old) if arg==str(out/'collection-admission.json') else arg for arg in command]
    result=subprocess.run(bad,cwd=root,capture_output=True,text=True)
    (out/(label+'-old-admission-rejected.txt')).write_text(result.stdout+result.stderr)
    checks[label+'_old_admission_rejected']=result.returncode!=0 and (
        'environment_contract_mismatch' in result.stderr or 'alignment_admission_code_changed' in result.stderr)
    checks[label+'_rejection_no_output']=not (out/('not-created-'+label)).exists()
sources=['src/v4/manual_residual_sac_env.py','src/v4/run_manual_residual_sac.py',
         'src/v4/validate_manual_wait_strength.py','src/v4/collect_manual_recovery.py',
         'tests/v4/test_manual_contract.py','work/manual-wait-strength-validation-20261013/audit_wait_strength_handoff.py',
         'work/manual-strength-collection-20261013/prepare_strength_collection.py']
hashes={}
for name in sources:
    path=root/name;ast.parse(path.read_text());hashes[name]=sha(path)
    target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
checks['all_sources_syntax_passed']=True
checks['environment_contract_still_current']=environment_contract()==contract
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
checks['gpu_compute_idle']=not gpu.strip()
review=dict(revision_note='v1 passed; v2 freezes corrected source-label counts for new waiting-strength data.',status='passed' if all(checks.values()) else 'failed',checks=checks,cpu_tests=tests.testsRun,
            cli_checks=cli,code=hashes,environment_contract=contract,
            collection_admission_path=str(out/'collection-admission.json'),
            collection_admission_sha256=sha(out/'collection-admission.json'),
            collection_ready=all(checks.values()),training_admission=False,
            new_physics_frames=0,neural_updates=0,new_transition_count=0,
            next_scope='1200 seconds collection/validation only: manual baseline, stop failure, half/full recovery; verify complete datasets before SAC.')
write(out/'review.json',review)
print(json.dumps({k:review[k] for k in ['status','cpu_tests','checks','collection_ready','training_admission','new_physics_frames','neural_updates']}))
sys.exit(0 if all(checks.values()) else 1)
