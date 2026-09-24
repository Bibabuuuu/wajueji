"""Read-only same-calibration endpoint comparison; no physical replay."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from live_bucket_fill import MixedFillEstimator
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
RUN=None
BUNDLE=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
def identity(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 global RUN
 parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();RUN=args.run_dir.resolve()
 r=json.loads((RUN/'report.json').read_text());assert r['status']=='derived_demo_completed_pending_independent_audit'
 original_report=ROOT/'runs/layered-47-sparse-manual-contact-v1'/r['teacher_run']/'report.json'
 teacher=json.loads(original_report.read_text());config=teacher.get('live_bucket_fill',{}).get('config')
 if config is None:
  from live_bucket_fill import load_config
  config=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json')
 config=dict(config,occupancy_level_override=.2)
 manifest=BUNDLE/'results/layered-mound-physical-preflight-v1/scene-manifest.json'
 records=[v for v in json.loads(manifest.read_text())['rocks'] if v['layer_id'] in ('lower','lower_middle','middle','upper')]
 estimator=MixedFillEstimator(config,[v['volume_m3'] for v in records],_cache_rock_vertices(BUNDLE,records))
 assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
 output=dict(schema='derived_fill_comparison_v2',occupancy_level=.2,inputs=[identity(Path(__file__)),identity(original_report),identity(manifest)],training_eligible=False)
 for name,item in [('original',r['original_endpoint']),('derived',r['final_checkpoint'])]:
  path=Path(item['path']);assert identity(path)['sha256']==item['sha256'];output['inputs'].append(identity(path))
  with np.load(path,allow_pickle=False) as a:
   pose=assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket']
   output[name]=estimator.measure_snapshot(dict(frame=r['frames_completed'],soil=a['particle_q'],rocks=a['rigid_qpos'],pose=pose))
  assert output[name]['valid_estimate']
 output['fill_difference_percentage_points']=output['derived']['fill_percent']-output['original']['fill_percent']
 (WORK/(r['teacher_run']+'.fill.json')).write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({k:output[k] for k in ('original','derived','fill_difference_percentage_points')},indent=2))
if __name__=='__main__':main()
