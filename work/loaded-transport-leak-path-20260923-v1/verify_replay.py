"""Finalize the preserved replay after its string-array comparison error."""
import json
from pathlib import Path
import numpy as np
work=Path(__file__).resolve().parent;run=work/'replay-v2'
r=json.loads((run/'report.json').read_text());rows=[json.loads(l) for l in (run/'trace.ndjson').read_text().splitlines()]
assert len(rows)==887 and all(all(x['exact'].values()) and all(x['hard'].values()) and x['material_safe'] for x in rows)
a=np.load(run/'terminal-diagnostic-000887.npz');b=np.load('/home/qunwang/wajueji/runs/layered-47-loaded-transport-v1/loaded-transport-20260923-215454-962382/final-000887.diagnostic.npz')
exact={key:bool(np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key])) for key in b.files if key in a.files}
assert all(exact.values()) and set(a.files)==set(b.files)
result={'status':'replay_paths_recorded','frames':887,'all_887_recorded_states_exact':True,'all_hard_guards_passed':True,'terminal_arrays_exact':exact,'terminal_field_sets_equal':True,'raw_runner_report_status':r['status'],'raw_runner_report_failure_stage':'post_physics_final_array_comparison_string_dtype','raw_runner_report_preserved':True,'bindings_unchanged':r['bindings_unchanged'],'wall_seconds':r['wall_seconds'],'original_initialization_failure':'v1 frame1 missing applied_command_si in diagnostic hard guard wrapper; raw failure retained','training_eligible':False,'resume_qualified':False}
(work/'verified-replay.json').write_text(json.dumps(result,indent=2)+'\n')
print('VERIFIED',len(rows),'frames;',len(exact),'terminal arrays exact')
