"""Read-only prefix comparison; no resumed physics or training qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=OUT.parent/'manual-soil-loss-replay-20260923-v1'
protocol=json.loads((OUT/'protocol.json').read_text())
report=json.loads((OUT/'report.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(path):
    a=np.load(path);F=a['particle_elastic_strain'].astype(float);Jp=a['particle_Jp']
    sv=np.linalg.svd(F,compute_uv=False)
    nonfinite={k:int((~np.isfinite(a[k])).sum()) for k in a.files if a[k].dtype.kind in 'fc' and not np.isfinite(a[k]).all()}
    qd=a['particle_qd'].astype(float);speed=np.linalg.norm(qd,axis=1)
    return {'snapshot':str(path),'sha256':sha(path),'particles':len(F),'nonfinite_fields':nonfinite,
        'F_singular_min':float(sv.min()),'F_max_singular_p50_p99_max':np.quantile(sv[:,0],[.5,.99,1]).tolist(),
        'F_max_singular_over2':int((sv[:,0]>2).sum()),'F_frobenius_over1000':int((np.linalg.norm(F,axis=(1,2))>1000).sum()),
        'Jp_nonfinite':int((~np.isfinite(Jp)).sum()),'finite_Jp_min_p50_p99_max':np.quantile(Jp[np.isfinite(Jp)],[0,.5,.99,1]).tolist(),
        'particle_speed_p99_max':np.quantile(speed,[.99,1]).tolist()}

comparisons=[]
for frame in [277,540,600]:
    path=OUT/f'candidate-{frame:06d}.npz';old=BASE/(f'first-nonfinite-diagnostic-{frame:06d}.npz' if frame==540 else f'diagnostic-{frame:06d}.npz')
    if path.exists():comparisons.append({'frame':frame,'baseline':audit(old),'candidate':audit(path)})
failure=[]
for path in sorted(OUT.glob('rejected-*.npz')):
    if 'material-input' not in path.name:failure.append(audit(path))
trace=[json.loads(l) for l in (OUT/'trace.ndjson').read_text().splitlines()]
checks={'sources_unchanged':all(sha(p)==h for p,h in protocol['bindings'].items()),
        'restore_exact':report.get('restore_exact',False),
        '600frames_completed':report['frames']==600,
        'all_executed_frame_guards_passed':all(all(r['guards'].values()) for r in trace),
        'candidate_gpu_preflight_passed':all(json.loads((OUT/'preflight-report.json').read_text())['checks'].values()),
        'original_actual_commands_replayed':len(trace)==report['frames'],
        'full_scoop_qualified':False,'late_soil_loss_fix_verified':False}
source=[json.loads(l) for l in (ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279/physics.ndjson').read_text().splitlines()]
checks['original_actual_commands_replayed']=checks['original_actual_commands_replayed'] and all(r['frame']==s['frame'] and r['applied_command_si']==s['applied_command_si'] for r,s in zip(trace,source))
result={'status':report['status'],'checks':checks,'comparisons':comparisons,'rejected_snapshots':failure,
        'meaning':'Same complete initial state, original applied commands and frozen controller bridge; coupled mechanics and soil are allowed to differ after candidate starts. Original600frame baseline already has invalid material history. This is not a later soil-loss or full-scoop acceptance.'}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
