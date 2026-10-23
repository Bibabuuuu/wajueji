"""One-step diagnostic of tiny residual MPM impulse samples; no guard changes."""
import json
import numpy as np
from scipy.spatial import cKDTree
from layered_47_cycle_runtime import CycleRuntime
with CycleRuntime('runs/layered-47-cycle-v1/next-prepoint-physical-v1/prepoint.npz',
                  'runs/layered-47-cycle-v1/prepoint-contact-diagnosis-v1') as r:
    row=r.step([0]*4)
    impulses,positions,ids=r.fixture.solver.collect_collider_impulses(r.fixture.state)
    impulses=impulses.numpy();positions=positions.numpy();ids=ids.numpy()
    mask=np.isin(ids,r.fixture.bucket_collider_ids)&(np.linalg.norm(impulses,axis=1)>1e-12)
    q=r.fixture.state.particle_q.numpy();d,index=cKDTree(q).query(positions[mask])
    result=dict(sample_positions=positions[mask].tolist(),impulse_ns=impulses[mask].tolist(),
        nearest_particle_distance_m=d.tolist(),nearest_particle_indices=index.tolist(),
        nearest_particle_positions=q[index].tolist(),hard_guard=row['hard_guard'])
    (r.output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
