"""Bind the completed real prepoint; only permits zero-step inference initially."""
import json
from pathlib import Path
from check_source import ROOT,W,SOURCE,read,sha
def main():
    review=read(W/'source-review.json')
    if not review['passed'] or review['source_sha256']!=sha(SOURCE):raise ValueError('source_admission_required')
    old=read(ROOT/'work/contact-multisource-lift-20260924-v1/first-protocol.json')
    paths=[Path(x['path']) for x in old['inputs'] if x['path'] not in (old['source_path'],old['source_path']+'.json')]
    paths += [SOURCE,Path(str(SOURCE)+'.json'),W/'source-review.json',W/'early-command-review.json',W/'feature-review.json',
        ROOT/'work/contact-next-prepoint-20260925-v1/audit-review.json',*W.glob('*.py'),W/'run.sh']
    meta=read(str(SOURCE)+'.json')
    for field in ('candidate_bindings','prepoint_bindings','prepoint_dependency_bindings'):paths.extend(Path(p) for p in meta[field])
    old.update(status='zero_preflight_only',source_path=str(SOURCE),source_sha256=sha(SOURCE),source_scoop='continuous_second',
        training_reset_allowed=False,adapter='real_second_scene_full_response_v1',
        inputs=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))])
    (W/'protocol.json').write_text(json.dumps(old,indent=2)+'\n')
    print('BOUND_REAL_SECOND_SOURCE',sha(SOURCE),'zero-step only')
if __name__=='__main__':main()
