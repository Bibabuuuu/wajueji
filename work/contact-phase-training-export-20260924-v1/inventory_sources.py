"""Read-only preparation inventory; no runtime construction or physics."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=ROOT/'work/contact-bc-data-preparation-20260924-v1/tensor-manifest.json'
assignment=ROOT/'work/contact-bc-data-preparation-20260924-v1/pilot-assignment.json'
m=json.loads(manifest.read_text());a=json.loads(assignment.read_text());results=[]
for e in m['episodes']:
 root=ROOT/'runs/layered-47-sparse-manual-contact-v1'/e['run'];report_path=root/'report.json';r=json.loads(report_path.read_text())
 with (root/'physics.ndjson').open() as f:first=json.loads(next(f))
 source=r.get('handoff_checkpoint') or r['source_checkpoint']
 # Cycle runs advertise their full prepoint in source_checkpoint.
 if r.get('cycle_lineage') is not None:source=r['source_checkpoint']
 p=Path(source['path']);assert sha(p)==source['sha256']
 side=Path(str(p)+'.json');meta=json.loads(side.read_text())
 with np.load(p,allow_pickle=False) as d:
  pose_equal=bool(np.array_equal(d['mechanism_qpos'],first['before_state']['mechanism_qpos']))
  velocity_equal=bool(np.array_equal(d['mechanism_qvel'],first['before_state']['mechanism_qvel']))
 split=next(k for k in ('development_runs','validation_runs') if e['run'] in a[k])
 results.append(dict(run=e['run'],split=split,tensor_rows=e['rows'],report=dict(path=str(report_path),sha256=sha(report_path)),
  initial_source=source,metadata=dict(path=str(side),sha256=sha(side)),schema=meta.get('schema'),
  initial_mechanism_position_equal=pose_equal,initial_mechanism_velocity_equal=velocity_equal,
  compatible_cycle_envelope=meta.get('schema')=='layered_47_cycle_checkpoint_v1',
  endpoint_paths=[v['path'] for v in r.get('cycle_checkpoints',[])],phase_export_completed=False,
  note='Header/pose inspection only; no restore qualification, no automatic source conversion'))
result=dict(schema='phase_replay_source_inventory_v1',physics_advanced=False,
 tensor_manifest_sha256=sha(manifest),assignment_sha256=sha(assignment),episodes=results)
(OUT/'source-inventory-v1.json').write_text(json.dumps(result,indent=2)+'\n')
for e in results:print(e['run'],e['schema'],e['initial_mechanism_position_equal'],e['initial_mechanism_velocity_equal'])
