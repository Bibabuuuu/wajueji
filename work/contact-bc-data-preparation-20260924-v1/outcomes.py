"""Uniform user-selected occupancy-level audit; does not rewrite old outcomes."""
from pathlib import Path
import json,hashlib
import numpy as np
from live_bucket_fill import MixedFillEstimator, load_config
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
BUNDLE=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
catalog=read(ROOT/'work/contact-collection-progress-20260924-v5/catalog.json')
assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
rocks=[v for v in read(BUNDLE/'results/layered-mound-physical-preflight-v1/scene-manifest.json')['rocks'] if v['layer_id'] in ('lower','lower_middle','middle','upper')]
vertices=_cache_rock_vertices(BUNDLE,rocks);results=[]
for episode in catalog['episodes']:
 p=Path(episode['snapshot']['path']);require_sha=episode['snapshot']['sha256'];assert sha(p)==require_sha
 r=read(p.parent/'report.json');config=dict(r['live_bucket_fill']['config']) if 'live_bucket_fill' in r else load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json');original=r.get('live_bucket_fill',{}).get('config',{}).get('occupancy_level_override');config['occupancy_level_override']=.2
 estimator=MixedFillEstimator(config,[v['volume_m3'] for v in rocks],vertices)
 with np.load(p) as a:
  pose=assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket']
  fill=estimator.measure_snapshot(dict(frame=episode['frames'],soil=a['particle_q'],rocks=a['rigid_qpos'],pose=pose))
 assert fill['valid_estimate'] and sha(p)==require_sha
 results.append(dict(run=episode['run'],scoop_index=episode['scoop_index'],lineage=episode['lineage_root'],source_sha256=require_sha,original_threshold=original,
                     working_fill=fill,working_fill_meets_95=fill['fill_percent']>=95,recording_review_pass=episode['recording_review_pass'],
                     expert_admission='pending_goal_phase_and_detachment_review',operator_retargeting_review=episode['operator_retargeting_review']))
report=dict(schema='contact_uniform_fill_review_v1',occupancy_level=.2,original_results_unchanged=True,episodes=results,
            at_least_95_count=sum(r['working_fill_meets_95'] for r in results),
            scope='User-selected geometric mixed occupancy including heap; neither formal calibrated truck volume nor automatic expert-success admission.')
(OUT/'outcomes.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([dict(run=e['run'],fill=e['working_fill']['fill_percent']) for e in results],indent=2))
