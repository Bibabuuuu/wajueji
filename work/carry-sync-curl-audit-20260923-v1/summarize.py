import json
from pathlib import Path
import numpy as np
out=Path('/home/qunwang/wajueji/work/carry-sync-curl-audit-20260923-v1')
rows=[json.loads(x) for x in (out/'trace.ndjson').read_text().splitlines()]
result={'frames':len(rows),'windows':[]}
for first,last in [(481,600),(1014,1340),(1341,1370)]:
 selected=[r for r in rows if first<=r['frame']<=last]
 if not selected:continue
 entry={'frames':[first,last],'count':len(selected)}
 for key in ('projection_cohort_max_m','projection_global_max_m','cohort_max_step_m','cohort_p99_step_m','cohort_max_up_step_m','cohort_p99_rigid_carry_error_m','display_bucket_motion_staleness_p99_m'):
  worst=max(selected,key=lambda r:r[key]);entry[key]={'max':worst[key],'frame':worst['frame']}
 entry['target_actual_bucket_error_mean_m']=float(np.mean([r['controller_target'][3]-r['axis_positions'][3] for r in selected]))
 result['windows'].append(entry)
(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
