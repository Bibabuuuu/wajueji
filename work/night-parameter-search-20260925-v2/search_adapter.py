"""Install a process-local search adapter; frozen source files stay untouched."""
import sys,os,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
from parameters import validate,modify
CONFIG=Path(os.environ['NIGHT_TRIAL_CONFIG']).resolve()
if not CONFIG.is_relative_to(ROOT/'runs/night-parameter-search-20260925-v2'):raise ValueError('search_config_path')
SPEC=json.loads(CONFIG.read_text());PARAMETERS=validate(SPEC['parameters'])
if SPEC['status']!='authorized_bounded_parameter_search':raise ValueError('search_not_authorized')
sys.path.insert(0,str(ROOT/'work/contact-next-scoop-20260925-v1'))
import candidate_multisource as base
OriginalFeedback=base.LiftFeedback;OriginalRuntime=base.MultisourceRuntime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class SearchFeedback(OriginalFeedback):
 def command(self,**kw):
  raw=np.asarray(kw['bc_action'],float);q=np.asarray(kw['qpos']);closure=float((q[15]-.37)/(.852-.37))
  adjusted=modify(raw,kw['phase'],closure,PARAMETERS)
  request,command,detail=super().command(**{**kw,'bc_action':adjusted})
  detail['parameter_search']=dict(parameters=PARAMETERS,raw_bc=raw.tolist(),scaled_bc=adjusted.tolist(),scope='curl_load below95pct actual stroke')
  return request,command,detail
class SearchRuntime(OriginalRuntime):
 def save(self,path,**kw):
  result=super().save(path,**kw);side=Path(str(path)+'.json');meta=json.loads(side.read_text())
  meta.update(search_bindings={str(p):sha(p) for p in [*W.glob('*.py'),CONFIG]},
   experiment_reset=dict(kind='repeated_same_complete_prepoint',trial_id=SPEC['trial_id'],not_continuous_scoop_evidence=True),search_parameters=PARAMETERS)
  side.write_text(json.dumps(meta,indent=2)+'\n');return result
base.LiftFeedback=SearchFeedback;base.MultisourceRuntime=SearchRuntime
