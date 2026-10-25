"""Bind the approved fixed-route continuation without launching physics."""
from runtime import ROOT,W,SOURCE,SOURCE_SHA,MAX_FRAMES,MAX_SECONDS,read,sha,load_endpoint
import json
def main():
    _,meta=load_endpoint(SOURCE)
    paths=[*W.glob('*.py'),SOURCE,ROOT/'work/contact-second-finish-unload-20260925-v4/protocol.json',
           SOURCE.parent/'report.json',SOURCE.parent/'physics/frames.ndjson',SOURCE.parent/'actions.ndjson']
    from controller import base,yaw_preview
    import load_tracking,continuous_preview
    paths.extend([*map(lambda m:m.__file__,(base,yaw_preview,load_tracking,continuous_preview)),str(SOURCE)+'.json'])
    for field in ('candidate_bindings','tail_bindings'):paths.extend(meta[field])
    paths.extend(ROOT/'src/v4'/p for p in meta['runtime_bindings'])
    paths.extend((ROOT/'planning').glob('*.py'))
    paths.extend(ROOT/'src/v4'/p for p in ('layered_47_cycle_geometry.py','contact_bc_ground_guard.py','contact_geometric_feedback.py'))
    paths.extend(ROOT/p for p in ('configs/v4/layered-47-truck-physics-v1.json','evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json','evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz'))
    protocol=dict(status='authorized_fixed_route_continuation',source_sha256=SOURCE_SHA,
        maximum_frames=MAX_FRAMES,maximum_wall_seconds=MAX_SECONDS,source_frame=3164,
        scope='Resume normal handoff; fixed CLEAR_PILE ALIGN DUMP only; no soil terrain map; no return/third scoop',
        inputs={str(p):sha(p) for p in sorted(set(map(str,paths)))})
    if (W/'protocol.json').exists():raise ValueError('immutable_protocol_exists')
    (W/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
if __name__=='__main__':main()
