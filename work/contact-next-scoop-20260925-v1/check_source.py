"""Read-only next-contact source admission; never imports a physical runtime."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-next-prepoint-20260925-v1/physical'
SOURCE=RUN/'prepoint.npz'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=read(RUN/'report.json');audit=read(ROOT/'work/contact-next-prepoint-20260925-v1/audit-review.json')
    meta=read(str(SOURCE)+'.json');rows=[json.loads(x) for x in (RUN/'physics/frames.ndjson').read_text().splitlines()]
    tail=rows[-30:];physical=np.load(SOURCE,allow_pickle=False)
    checks=dict(completed=report['status']=='prepoint_saved_requires_independent_audit',
        audited=audit['audit_passed'] and audit['endpoint_sha256']==sha(SOURCE),
        schema=meta['schema']=='layered_47_full_response_next_prepoint_v1',
        stage=meta['stage']=='prepoint',cycle=meta['cycle_index']==2,training_reset=meta.get('training_reset') is None,
        tail_length=len(tail)==30,tail_soil=all(r['bucket_mpm_contact_samples']==0 for r in tail),
        tail_rock=all(r['rock_bucket_contact_substeps']==0 for r in tail),
        physical_guards=all(r['hard_guard'] for r in rows),endpoint=np.array_equal(physical['mechanism_qpos'],rows[-1]['after_state']['mechanism_qpos']),
        source_sha=meta['checkpoint_sha256']==sha(SOURCE))
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},source=str(SOURCE),source_sha256=sha(SOURCE),
        source_metadata_sha256=sha(str(SOURCE)+'.json'),physics_steps_added=0,tail_contact_frames=30,
        formal_three_scoop_qualified=False)
    (W/'source-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':main()
