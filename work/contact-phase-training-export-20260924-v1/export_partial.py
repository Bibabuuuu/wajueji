"""Export known causal phase prefix; unknown rows remain explicit, not labels."""
import hashlib,json
from pathlib import Path
import numpy as np
from contact_phase_data import align_phase_labels,PHASES
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-phase-teacher-replay-v1/third-scoop-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((RUN/'report.json').read_text())
assert r['status']=='failed' and r['matched_recorded_frames']==1206
trace=[json.loads(s) for s in (RUN/'phase-trace.ndjson').read_text().splitlines()]
assert len(trace)==1206 and trace[-1]['phase']['phase_for_next_decision']=='lift'
m_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/tensor-manifest.json';m=json.loads(m_path.read_text());e=next(x for x in m['episodes'] if x['run']==r['run'])
p=Path(e['path']);assert sha(p)==e['sha256']
with np.load(p,allow_pickle=False) as d:frames=d['frame'].copy()
valid=frames<=1206
known=align_phase_labels(frames[valid],trace)
ids=np.full(len(frames),-1,dtype=np.int8);ids[valid]=known['phase_id']
one_hot=np.zeros((len(frames),5),dtype=np.float32);one_hot[valid]=known['phase_one_hot']
latest=np.full(len(frames),-1,dtype=np.int64);latest[valid]=known['latest_allowed_evidence_frame']
assert np.count_nonzero(valid)==1203 and np.count_nonzero(~valid)==254
assert not np.any(ids==4)
output=ROOT/'local-data/contact-bc-phase-sidecars-20260924-v1';output.mkdir(exist_ok=True,parents=True)
dest=output/(r['run']+'.partial-phases.npz');assert not dest.exists()
np.savez_compressed(dest,frame=frames,phase_valid=valid,phase_id=ids,phase_one_hot=one_hot,latest_allowed_evidence_frame=latest)
with np.load(dest,allow_pickle=False) as d:
 np.testing.assert_array_equal(d['phase_valid'],valid)
 assert (d['phase_id'][~valid]==-1).all() and (d['phase_one_hot'][~valid]==0).all()
 np.testing.assert_array_equal(d['phase_id'][valid],known['phase_id'])
a_path=ROOT/'work/contact-bc-data-preparation-20260924-v1/pilot-assignment.json';a=json.loads(a_path.read_text())
assert r['run'] in a['validation_runs'] and r['run'] not in a['development_runs']
report=dict(schema='partial_causal_phase_sidecar_v1',status='partial_only_not_complete_five_phase_dataset',
 run=r['run'],source_tensor=e,sidecar=dict(path=str(dest),sha256=sha(dest)),source_report=dict(path=str(RUN/'report.json'),sha256=sha(RUN/'report.json')),
 source_phase_trace=dict(path=str(RUN/'phase-trace.ndjson'),sha256=sha(RUN/'phase-trace.ndjson')),
 phase_order=PHASES,rows=len(frames),known_rows=int(valid.sum()),unknown_rows=int((~valid).sum()),
 phase_counts={name:int(np.count_nonzero(ids==i)) for i,name in enumerate(PHASES)},
 split='validation_runs',assignment_sha256=sha(a_path),training_eligible=False,complete_dataset=False,
 original_tensor_unchanged=sha(p)==e['sha256'],original_annotations_changed=False,
 policy='Unknown rows must be excluded explicitly; -1/zero one-hot is not an approach label. No off-soil confirmation assigned.',
 inputs=[dict(path=str(x),sha256=sha(x)) for x in (Path(__file__),ROOT/'src/v4/contact_phase_data.py',m_path)])
(WORK/'partial-export-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('rows','known_rows','unknown_rows','phase_counts','status')},indent=2))
