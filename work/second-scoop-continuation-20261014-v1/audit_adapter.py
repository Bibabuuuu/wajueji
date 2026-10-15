"""Independent read-only review of CPU adapter receipts and rejection gates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,subprocess,shutil,numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sys.path[:0]=[str(W),str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import read,write,sha,MODEL,MODEL_SHA
from yaw_transfer import YawTransferEnv,canonical_values,rotation
p=read(W/'adapter-cpu-v3/report.json');checks={}
def check(k,v):checks[k]=bool(v);assert v,k
check('105 CPU checks all passed',p['status']=='cpu_adapter_passed' and len(p['checks'])==p['check_count']==105 and all(p['checks'].values()))
for name,value in p['code_sha256'].items():check('tested code unchanged '+name,sha(W/name)==value)
check('legacy encoder unchanged',sha(ROOT/'src/v4/manual_residual_sac_env.py')==p['unchanged_legacy_encoder_sha256'])
protocol=read(ROOT/'work/continuous-single-scoop-20261014-v1/protocol.json')
check('all frozen core code unchanged',all(sha(Path(n))==h for n,h in protocol['code_sha256'].items()))
check('frozen model bytes unchanged',sha(MODEL)==MODEL_SHA==p['model_sha256'])
cases=read(W/'adapter-cpu-v3/rotation-cases.json')
check('24 metamorphic cases retained',len(cases)==p['synthetic_rotation_cases']==24)
check('reported observation max independently recomputed',max(c['maximum_observation_error'] for c in cases)==p['maximum_observation_error']<=2e-6)
check('reported action max independently recomputed',max(c['maximum_action_error'] for c in cases)==p['maximum_frozen_action_error']<=2e-6)
check('48 CPU predictions accurately counted',p['cpu_frozen_predict_calls']==2*len(cases)==48)
check('zero physical training updates',p['physics_steps_added']==p['native_initializations']==p['training_updates']==p['neural_updates']==0 and not p['physical_admission'])
check('real zero-angle evidence count',p['zero_angle_real_observations']==12 and p['real_cached_states']==4)
dummy=YawTransferEnv.__new__(YawTransferEnv)
for name in ['reset','step']:
 try:getattr(dummy,name)()
 except RuntimeError:check('candidate '+name+' rejected',True)
 else:raise AssertionError('candidate '+name+' allowed')
for bad in [np.zeros(220),np.r_[np.zeros(220),float('nan')]]:
 try:canonical_values(bad,.1,rotation(.1))
 except ValueError:pass
 else:raise AssertionError('invalid observation accepted')
check('invalid observation shape finite rejected',True)
check('original105 + independent gates do not imply rollout',p['limits'].startswith('Synthetic rotation equality') and not p['physical_admission'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();check('GPU compute idle',not gpu)
snap=W/'adapter-cpu-v3/code-snapshot';snap.mkdir(exist_ok=False)
for n in ['yaw_transfer.py','check_adapter_cpu.py','audit_adapter.py']:shutil.copyfile(W/n,snap/n)
write(W/'adapter-independent-audit.json',dict(status='passed',checks=checks,check_count=len(checks),cpu_frozen_predict_calls=48,physical_steps_added=0,
 training_updates=0,reference_input_adapter='implemented and CPU verified',physical_runner_admitted=False,
 source_contract='physical world unchanged; reference and policy input alone use yaw transformation',
 code_sha256={n:sha(snap/n) for n in ['yaw_transfer.py','check_adapter_cpu.py','audit_adapter.py']},
 limits='No new terrain/second-scoop execution or performance qualification. Original terminal adhesion permission remains.'))
print('passed',len(checks),'independent adapter checks')
