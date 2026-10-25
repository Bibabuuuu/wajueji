"""Log-level whole-trajectory metrics; different paths are not matched-load A/B."""
import sys,json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha,LIMITS
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
def summarize(run,limit=None):
 rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
 a=rows(run/'actions.ndjson');p=rows(run/'physics/frames.ndjson')
 if limit is not None:a=a[:limit];p=p[:limit]
 assert len(a)==len(p)
 cmd=np.array([x['applied_command_si'] for x in a]);q=np.array([a[0]['before_state']['mechanism_qpos']]+[x['after_state']['mechanism_qpos'] for x in p])[:,[7,9,12,15]]
 vel=np.diff(q,axis=0)*60
 statistics={}
 for name,v in [('command_si',cmd),('frame_mean_velocity_si',vel),('endpoint_qvel_si',np.array([x['after_state']['mechanism_qvel'] for x in p])[:,[6,8,11,14]])]:
  changes=np.abs(np.diff(v,axis=0))
  statistics[name]=dict(maximum_abs=np.max(abs(v),axis=0).tolist(),maximum_step=changes.max(axis=0).tolist(),maximum_step_frame=(np.argmax(changes,axis=0)+2).tolist(),step_quantiles=np.quantile(changes,[.5,.95,.99],axis=0).tolist(),total_variation=changes.sum(axis=0).tolist())
 rate=a[0]['feedback'].get('command_rate_per_s');previous=np.vstack([np.zeros(4),cmd[:-1]])
 excess=np.max(abs((cmd-previous)/LIMITS),axis=1)>(rate/60+1e-6) if rate else np.zeros(len(a),bool)
 overrides=[x['frame'] for x in a if x['feedback'].get('slew_safety_override')]
 return dict(run=str(run),frames=len(a),statistics=statistics,safety_override_frames=overrides,rate_exceedance_frames=(np.flatnonzero(excess)+1).tolist(),all_rate_exceedances_have_safety_intervention=all(a[i]['feedback'].get('slew_safety_override') or a[i]['command_projection_reason'] or a[i]['feedback']['preview_scale']!=1 for i in np.flatnonzero(excess)),bindings=[dict(path=str(run/f),sha256=sha(run/f)) for f in ['actions.ndjson','physics/frames.ndjson','report.json']])
result=dict(candidate=summarize(ROOT/sys.argv[1]),baseline=summarize(ROOT/'runs/contact-geometric-probe-20260924-v7/scoop1500'),scope='Different resulting trajectories. Descriptive speed metrics, not isolated physical cause or matched-load proof.',physics_steps_added=0)
result['baseline_equal_frame_budget']=summarize(ROOT/'runs/contact-geometric-probe-20260924-v7/scoop1500',result['candidate']['frames'])
(W/(Path(sys.argv[1]).name+'-speed-review.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['candidate']['all_rate_exceedances_have_safety_intervention']
