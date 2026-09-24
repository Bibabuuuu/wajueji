"""D154 saved prepoint validation; unchanged D152 physics, D153 control."""
import sys,json
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path.insert(0,str(ROOT/'work/contact-preview-lift-20260924-v1'))
from continuation_runtime import sha
from candidate_runtime import CandidateRuntime
from lift_feedback import LiftFeedback
from control_history import encode,DYNAMIC
class MultisourceRuntime(CandidateRuntime):
    def save(self,path,**kw):
        identity=super().save(path,**kw);p=Path(str(path)+'.json');meta=json.loads(p.read_text())
        meta.update(schema='layered_47_full_response_multisource_lift_candidate_v1',control_bindings={str(p):sha(p) for p in (ROOT/'work/contact-preview-lift-20260924-v1').glob('*.py')},multisource_bindings={str(p):sha(p) for p in W.glob('*.py')},continuation_qualified=False)
        p.write_text(json.dumps(meta,indent=2)+'\n');return identity

def save_control(out,r,feedback,phases,features,previous_action,previous_load):
    data=dict(schema='D154_control_state_v1',frame=r.frames,feedback=encode({k:getattr(feedback,k) for k in (*DYNAMIC,'lift_intent','lift_start_origin_z')}),phase=encode({k:v for k,v in vars(phases.router).items() if k not in ('config','contact')}),contact=encode({k:v for k,v in vars(phases.router.contact).items() if k!='config'}),features=encode(features.rows),feature_last_frame=features.last_frame,last_action=previous_action,last_load=previous_load,continuation_qualified=False)
    (out/'control-state.json').write_text(json.dumps(data,indent=2)+'\n')
