"""Source-specific frozen contact controllers with bounded shared BC scaling."""
import copy,json,os,sys
from pathlib import Path
import numpy as np
from sources import ROOT,W,RUN,SOURCE_SPECS,sha,read
from parameters import validate,modify,BASELINE
CONFIG=Path(os.environ['NIGHT_TRIAL_CONFIG']).resolve()
if not CONFIG.is_relative_to(RUN):raise ValueError('dual_trial_config_path')
SPEC=read(CONFIG);KIND=SPEC['source_kind'];SOURCE=SOURCE_SPECS[KIND]
PARAMETERS=validate(SPEC['parameters'])
if SPEC['status']!='authorized_bounded_dual_search':raise ValueError('dual_search_not_authorized')
family=ROOT/('work/contact-multisource-lift-20260924-v1' if KIND=='first' else 'work/contact-next-scoop-20260925-v1')
sys.path.insert(0,str(family))
import candidate_multisource as base
OriginalFeedback=base.LiftFeedback;OriginalRuntime=base.MultisourceRuntime;original_save=base.save_control
# CausalFeatures is shared bookkeeping only; the source's initial observation is retained.
sys.path.append(str(ROOT/'work/contact-next-scoop-20260925-v1'))
from control_history import DYNAMIC
class SearchFeedback(OriginalFeedback):
 def command(self,**kw):
  saved={k:copy.deepcopy(getattr(self,k)) for k in (*DYNAMIC,'lift_intent','lift_start_origin_z')}
  self.before_command=saved
  raw=np.asarray(kw['bc_action'],float);q=np.asarray(kw['qpos']);closure=float((q[15]-.37)/(.852-.37))
  adjusted=modify(raw,kw['phase'],closure,PARAMETERS)
  try:request,command,detail=super().command(**{**kw,'bc_action':adjusted})
  except BaseException:
   for k,v in saved.items():setattr(self,k,v)
   raise
  detail['parameter_search']=dict(parameters=PARAMETERS,raw_bc=raw.tolist(),scaled_bc=adjusted.tolist(),source_kind=KIND)
  return request,command,detail
class SearchRuntime(OriginalRuntime):
 def __init__(self,source,output,**kw):
  if Path(source).resolve()!=SOURCE['source'] or sha(source)!=SOURCE['sha']:raise ValueError('dual_exact_prepoint_required')
  super().__init__(source,output,**kw)
  if self.metadata['cycle_index']!=SOURCE['cycle'] or self.truck_enabled!=SOURCE['truck']:
   raise ValueError('dual_source_state_changed')
 def save(self,path,**kw):
  result=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
  meta.update(schema='layered_47_dual_search_contact_v1',search_source_kind=KIND,source_prepoint_sha256=SOURCE['sha'],
    search_bindings={str(p):sha(p) for p in [*W.glob('*.py'),CONFIG]},search_parameters=PARAMETERS,
    experiment_reset=dict(kind='repeated_complete_prepoint',trial_id=SPEC['trial_id'],not_continuous_scoop_evidence=True))
  side.write_text(json.dumps(meta,indent=2)+'\n');return result

def initial_history(runtime):
 if KIND=='first':return [0.]*4,None
 return list(runtime.metadata['transport_controller_state']['previous_command']),list(runtime.bridge.state['equivalent_drive_load'])
def save_control(out,r,feedback,phases,features,previous_action,previous_load):
 features.retain_executed(r.frames)
 if r.frames==0 and feedback.last_frame>0:
  for k,v in feedback.before_command.items():setattr(feedback,k,copy.deepcopy(v))
 original_save(out,r,feedback,phases,features,previous_action,previous_load)

def verify_baseline_prefix(out):
 if not SPEC.get('verify_baseline_prefix'):return
 if PARAMETERS!=BASELINE:raise ValueError('baseline_prefix_requires_identity_parameters')
 reference=SOURCE['gate'];candidate=Path(out)/'periodic-00120.npz'
 if sha(reference)!=SOURCE['gate_sha']:raise ValueError('baseline_reference_changed')
 with np.load(reference) as old,np.load(candidate) as new:
  exact=set(old.files)==set(new.files) and all(np.array_equal(old[k],new[k]) for k in old.files)
 result=dict(passed=bool(exact),source_kind=KIND,reference_sha256=sha(reference),candidate_sha256=sha(candidate),frames=120)
 (Path(out)/'baseline-prefix-review.json').write_text(json.dumps(result,indent=2)+'\n')
 if not exact:raise ValueError('baseline_first120_changed')
base.LiftFeedback=SearchFeedback;base.MultisourceRuntime=SearchRuntime;base.save_control=save_control
