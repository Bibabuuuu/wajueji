from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import transport_plan
from run_layered_47_sparse_manual_contact_episode import _clean
import json
r=CycleRuntime('runs/layered-47-cycle-v1/truck-hold-v1/final.npz','runs/layered-47-cycle-v1/transport-geometry-v7')
segments,checker=transport_plan(r)
for s in segments:print(s['name'],s['path']['steps'],s['minimum_clearance'],flush=True)
(r.output/'plan.json').write_text(json.dumps(_clean(dict(segments=segments,observation=checker.observation)),indent=2))
