"""Read saved baseline arrays/logs only; never import or step a simulator."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
SOURCES={
 'second_prepoint':ROOT/'runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz',
 'contact_endpoint':ROOT/'runs/contact-next-scoop-20260925-v1/candidate/final.npz',
 'curl_handoff':ROOT/'runs/contact-second-finish-unload-20260925-v4/physical/handoff.npz'}
def main():
    result=dict(physical_steps_added=0,snapshots={},independent_control_audit_completed=False)
    for name,path in SOURCES.items():
        with np.load(path,allow_pickle=False) as state:
            q=state['particle_q'];origin=q.min(0)-.08
            ijk=np.floor((q-origin)/.04).astype(int);shape=ijk.max(0)+3
            far=np.flatnonzero(np.linalg.norm(q[:,:2],axis=1)>15)
            result['snapshots'][name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                particle_count=len(q),minimum_m=q.min(0).tolist(),maximum_m=q.max(0).tolist(),
                dense_grid_shape=shape.tolist(),dense_grid_cells=int(np.prod(shape)),
                exceeds_original_12000000_cell_budget=bool(np.prod(shape)>12000000),
                far_particle_indices=far.tolist(),far_particle_positions=q[far].tolist(),
                far_particle_speed_m_s=np.linalg.norm(state['particle_qd'][far],axis=1).tolist())
    tail=SOURCES['curl_handoff'].parent
    rows=[json.loads(line) for line in (tail/'physics/frames.ndjson').read_text().splitlines()]
    report=json.loads((tail/'report.json').read_text())
    result.update(baseline_report_sha256=hashlib.sha256((tail/'report.json').read_bytes()).hexdigest(),
        status=report['status'],failure=report['failure'],frames=report['frames'],wall_seconds=report['wall_seconds'],
        returned_frame_hard_guards=all(row['hard_guard'] for row in rows),
        logged_frames_match=len(rows)==report['frames'],handoff=report['handoff'],
        limitation='Read-only failure triage, not full physical/control qualification. Far particle origin/cause not established; no particle removed or ignored.')
    (W/'baseline-observation-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
