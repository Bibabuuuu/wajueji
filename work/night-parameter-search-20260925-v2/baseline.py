"""Exact, explicit three-segment baseline admission; never relabel a failed run."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CONTACT=ROOT/'runs/contact-next-scoop-20260925-v1/candidate'
PREFIX=ROOT/'runs/contact-second-finish-unload-20260925-v4/physical'
TAIL=ROOT/'runs/contact-second-fixed-route-20260925-v1/physical'
REVIEW=ROOT/'work/contact-second-fixed-route-review-20260925-v1'
HANDOFF_SHA='f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d'
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def validate_prefix(contact,prefix,tail):
    c=read(Path(contact)/'report.json');p=read(Path(prefix)/'report.json');t=read(Path(tail)/'report.json')
    # This is a specific normal handoff saved BEFORE a known zero-step planner error.
    return (c['status']=='bounded_probe_completed' and p['status']=='failed'
        and p.get('failure')=="ValueError('cycle_observation_grid_budget')" and p['frames']==439
        and p['handoff']['ready'] and p['handoff']['frame_end']==3164 and not p['stages']
        and p['source_sha256']==sha(Path(contact)/'final.npz')
        and p['handoff_checkpoint']['sha256']==HANDOFF_SHA==sha(Path(prefix)/'handoff.npz')
        and t['source_sha256']==HANDOFF_SHA and t['source_frame_end']==3164)
def readiness():
    issues=[]
    if not (TAIL/'report.json').exists():return ['fixed_route_report_missing']
    t=read(TAIL/'report.json')
    if t['status']!='second_scoop_unload_completed':issues.append('fixed_route_not_completed:'+t['status'])
    if not validate_prefix(CONTACT,PREFIX,TAIL):issues.append('baseline_segment_link_invalid')
    for name in ('contact-audit.json','carry-audit.json','audit-review.json','replay-review.json'):
        p=REVIEW/name
        if not p.exists():issues.append('baseline_review_pending:'+name)
        else:
            result=read(p)
            if not (result.get('audit_passed') is True or name=='replay-review.json' and result.get('status')=='cache_sha_and_boundaries_passed'):
                issues.append('baseline_review_failed:'+name)
            if name in ('contact-audit.json','carry-audit.json'):
                bound=result.get('reviewed_inputs',{})
                if not bound or any(sha(path)!=digest for path,digest in bound.items()):issues.append('baseline_review_input_changed:'+name)
            if name=='audit-review.json' and t['status']=='second_scoop_unload_completed':
                if result.get('endpoint_sha256')!=sha(TAIL/'final.npz') or result.get('frames')!=t['frames']:issues.append('fixed_route_audit_endpoint_changed')
    return issues
def evidence_paths():
    paths=[]
    for directory,endpoint in ((CONTACT,'final.npz'),(PREFIX,'handoff.npz'),(TAIL,'final.npz')):
        paths.extend(directory/name for name in ('report.json','actions.ndjson','physics/frames.ndjson','display/reconstruction.json',endpoint,endpoint+'.json'))
    paths.extend(REVIEW/name for name in ('contact-audit.json','carry-audit.json','audit-review.json','replay-review.json'))
    return paths
