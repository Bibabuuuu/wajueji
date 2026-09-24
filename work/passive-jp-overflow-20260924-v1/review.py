"""Freeze the bounded overflow fix and continuation evidence, without physics."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/passive-jp-overflow-20260924-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=ROOT/'runs/layered-47-cycle-v1/third-scoop-finish-20260924-094135-1368892/transport-after-retreat/align.npz'
source=np.load(original);candidate=np.load(RUN/'candidate-993/final.npz')
baseline=np.load(RUN/'baseline-24/prefix-24.npz');prefix=np.load(RUN/'candidate-993/prefix-24.npz')
whole=np.load(RUN/'integrated-six/final.npz');resumed=np.load(RUN/'integrated-resume-three/final.npz')
physical=read(OUT/'physical-window-check.json')
def implementation(path):
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))]
    return ast.dump(tree,include_attributes=False)
metadata=read(RUN/'integrated-resume-three/final.npz.json')
checks=dict(original_align_unchanged=sha(original)=='1f5796ab9d13c5320f2ffb19c3bd46f2d143486284354328d060486621df9a2e',
 candidate_993_safe=physical['frames']==993 and all(physical[k] for k in ('all_hard_guards','all_material_guards','all_sweep_guards')),
 prefix_23_other_arrays_exact=all(np.array_equal(baseline[k],prefix[k]) for k in baseline.files if k!='particle_Jp'),
 frozen_Jp_exact_source=bool(np.array_equal(candidate['particle_Jp'],source['particle_Jp'])),
 finite_all_candidate_arrays=all(np.isfinite(candidate[k]).all() for k in candidate.files if np.issubdtype(candidate[k].dtype,np.number)),
 deployed_hook_matches_tested_candidate=implementation(ROOT/'src/v4/soil_passive_jp_history.py')==implementation(OUT/'passive_jp_candidate.py'),
 integrated_six_equals_three_plus_three=all(np.array_equal(whole[k],resumed[k]) for k in whole.files),
 integrated_source_restore_exact=read(RUN/'integrated-six/audit.json')['zero_step']['exact'],
 integrated_resume_restore_exact=read(RUN/'integrated-resume-three/audit.json')['zero_step']['exact'],
 policy_inherited_without_flag=metadata['material_history_policy']=='zero_hardening_frozen_jp_v1',
 policy_source_bound=metadata['runtime_bindings']['soil_passive_jp_history.py']==sha(ROOT/'src/v4/soil_passive_jp_history.py'))
result=dict(schema='passive_jp_overflow_fix_review_v1',passed=all(checks.values()),checks=checks,
 physical_window=physical,cpu_tests=dict(candidate=9,production_hook=9,checkpoint=8,manual=23,material_guard=3,production_total=43),
 baseline_original_comparison=read(OUT/'baseline-prefix-check.json'),
 integrated_final=dict(path=str(RUN/'integrated-six/final.npz'),sha256=sha(RUN/'integrated-six/final.npz')),
 source_bindings={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/v4'/n for n in ('soil_passive_jp_history.py','layered_47_cycle_runtime.py','layered_47_cycle_checkpoint.py','run_layered_47_sparse_manual_contact_episode.py','audit_layered_47_cycle.py')]},
 scope='Zero-hardening history policy only. No invalid-source repair, no strength or contact change. Original full unload and next prepoint remain separate qualifications.')
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));assert result['passed']
