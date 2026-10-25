"""Bind completed contact output for the already authorized continuous tail."""
import json
from pathlib import Path
from runtime import ROOT,W,SOURCE,read,sha
def main():
    report=read(SOURCE.parent/'report.json');meta=read(str(SOURCE)+'.json')
    if report['status']!='bounded_probe_completed':raise ValueError('contact_did_not_complete_normally')
    paths=[*W.glob('*.py'),SOURCE,Path(str(SOURCE)+'.json'),Path(report['phase_contract']['path']),ROOT/'runs/contact-next-scoop-20260925-v1/candidate/control-state.json',
        SOURCE.parent/'report.json',SOURCE.parent/'physics/frames.ndjson',SOURCE.parent/'actions.ndjson']
    from controller import carry,transport,yaw_preview
    import observation,load_tracking,continuous_preview
    paths.extend(Path(m.__file__) for m in (carry,transport,yaw_preview,observation,load_tracking,continuous_preview))
    paths.append(ROOT/'work/contact-first-transport-review-20260925-v1/static_rocks.py')
    for field in ('candidate_bindings','control_bindings','multisource_bindings','contact_bindings','preview_binding'):
        paths.extend(Path(p) for p in meta[field])
    paths.extend(ROOT/'src/v4'/name for name in ('contact_transport_handoff.py','contact_geometric_feedback.py','contact_bc_ground_guard.py','layered_47_cycle_geometry.py'))
    paths.extend((ROOT/'planning').glob('*.py'))
    protocol=dict(status='authorized_continuous_run',source_sha256=sha(SOURCE),maximum_frames=6000,maximum_wall_seconds=2000,
        user_scope='持续保存，第二铲卸料完成或保护退出后统一检查；第三铲和RL今晚不启动',
        inputs={str(p):sha(p) for p in sorted(set(paths))})
    (W/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
