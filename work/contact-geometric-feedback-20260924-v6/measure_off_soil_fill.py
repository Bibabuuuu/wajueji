"""Measure mixed fill only on an actually completed causal off-soil endpoint."""
import sys,json
from pathlib import Path
import numpy as np
from live_bucket_fill import MixedFillEstimator,load_config
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
from contact_bc_inference import sha
W=Path(__file__).resolve().parent;ROOT=W.parents[1];run=ROOT/sys.argv[1]
read=lambda p:json.loads(Path(p).read_text())
r=read(run/'report.json');a=read(W/(run.name+'-review.json'))
if r['status']!='bounded_probe_completed' or r['final_phase']!='off_soil' or not a['audit_passed'] or not a['behavior_gate_passed']:
 raise ValueError('no_qualified_off_soil_endpoint')
scene=read(run/'display/scene.json');data=np.load(run/'final.npz')
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
pose=assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket']
config=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json');assert config['occupancy_level_override']==.2
records=scene['records'];e=MixedFillEstimator(config,[v['volume_m3'] for v in records],_cache_rock_vertices(Path(scene['root']),records))
result=e.measure_snapshot(dict(frame=r['frames_completed'],soil=data['particle_q'],rocks=data['rigid_qpos'],pose=pose))
result.update(endpoint=dict(path=str(run/'final.npz'),sha256=sha(run/'final.npz')),off_soil_confirmed=True,not_stable_delivery_qualification=True)
(W/'off-soil-fill.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['valid_estimate']
