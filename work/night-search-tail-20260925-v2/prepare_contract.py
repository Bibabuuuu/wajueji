"""Bind completed contact output for the already authorized continuous tail."""
import json,os
from pathlib import Path
from runtime import ROOT,W,SOURCE,read,sha,PROTOCOL
def main():
    report=read(SOURCE.parent/'report.json');meta=read(str(SOURCE)+'.json')
    if report['status']!='bounded_probe_completed':raise ValueError('contact_did_not_complete_normally')
    paths=[*W.glob('*.py'),SOURCE,Path(str(SOURCE)+'.json'),Path(report['phase_contract']['path']),SOURCE.parent/'control-state.json',
        SOURCE.parent/'report.json',SOURCE.parent/'physics/frames.ndjson',SOURCE.parent/'actions.ndjson']
    from controller import carry,transport,yaw_preview
    import fixed_geometry,load_tracking,continuous_preview
    paths.extend(Path(m.__file__) for m in (carry,transport,yaw_preview,fixed_geometry,load_tracking,continuous_preview))
    paths.append(ROOT/'work/contact-second-fixed-route-20260925-v1/clearance.py')
    for field in ('candidate_bindings','control_bindings','multisource_bindings','contact_bindings','preview_binding','search_bindings'):
        paths.extend(Path(p) for p in meta[field])
    paths.extend(ROOT/'src/v4'/name for name in ('contact_transport_handoff.py','contact_geometric_feedback.py','contact_bc_ground_guard.py','layered_47_cycle_geometry.py'))
    paths.extend((ROOT/'planning').glob('*.py'))
    protocol=dict(status='authorized_continuous_run',evaluation_mode=os.environ.get('NIGHT_EVALUATION_MODE','dig_only'),source_sha256=sha(SOURCE),maximum_frames=6000,maximum_wall_seconds=2000,
        user_scope='用户已选择8h本地挖掘参数搜索；普通候选到CLEAR_PILE，最佳候选独立重复后仅以固定运输卸料验收；不改神经网络权重或正式验收资格',
        inputs={str(p):sha(p) for p in sorted(set(paths))})
    if PROTOCOL.exists():raise ValueError('immutable_trial_protocol_exists')
    (PROTOCOL).write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
